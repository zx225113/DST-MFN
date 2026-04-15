import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import time
import numpy as np

from .backbone import build_backbone
from .token_encoder import build_token_encoder
from .FFM import FFM
class DFGAR(nn.Module):  # 定义一个继承自 nn.Module 的神经网络类 DFGAR
    def __init__(self, args):
        super(DFGAR, self).__init__()

        self.dataset = args.dataset  # 数据集名称
        self.num_class = args.num_activities  # 活动类别数量

        # model parameters
        self.num_frame = args.num_frame  # 输入视频帧数
        self.hidden_dim = args.hidden_dim  # 模型隐藏层维度
        self.num_tokens = args.num_tokens  # Token嵌入数量

        # feature extraction
        self.backbone = build_backbone(args)  # 构建特征提取模块 + 位置编码模块
        self.token_encoder = build_token_encoder(args)  # 构建 Token 嵌入模块
        self.query_embed = nn.Embedding(self.num_tokens, self.hidden_dim)  # 创建嵌入层，用于生成查询向量来获取场景tokens（12,256）
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.token_encoder.d_model, kernel_size=1)
        #  1x1卷积层，用于将backbone输出的特征通道数调整到token encode

        # ---------------------------------------------------------------------
        # 修复核心 1：在 __init__ 中初始化 FFM，确保它能随模型一起转移到 CUDA
        # ---------------------------------------------------------------------
        # 检查是否使用了 optical flow (即 high order motion features)
        use_optical_flow = hasattr(args, 'use_optical_flow') and args.use_optical_flow

        self.ffm = FFM(dim1=self.backbone.num_channels,
                       dim2=self.hidden_dim,
                       res=0,
                       use_high_order=use_optical_flow)
        # ---------------------------------------------------------------------

        if self.dataset == 'volleyball':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            # 使用两个3x3一维卷积层，带padding保持尺寸
        elif self.dataset == 'nba':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            # 使用三个5x5卷积层，不使用padding会逐渐减小时间维度
        else:
            assert False

        self.self_attn = nn.MultiheadAttention(self.token_encoder.d_model, args.nheads_agg, dropout=args.drop_rate)  # 多头自注意力机制，用于token间的信息交互
        self.dropout1 = nn.Dropout(args.drop_rate)  # Dropout层，用于防止过拟合
        self.norm1 = nn.LayerNorm(self.hidden_dim)  # LayerNorm层，用于标准化特征
        self.norm2 = nn.LayerNorm(self.hidden_dim)  # LayerNorm层，用于标准化特征
        self.classifier = nn.Linear(self.hidden_dim, self.num_class)  # 分类层，用于将特征映射到类别空间

        self.relu = F.relu  # 定义ReLU激活函数
        self.gelu = F.gelu  # 定义GELU激活函数

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):  # 检查当前模块是否是线性层 isinstance(m, nn.Linear) 会判断模块 m 是否是 nn.Linear 类的实例 只有线性层才需要进行这种特定的权重初始化
                    nn.init.kaiming_normal_(m.weight)  # 使用Kaiming正态分布初始化方法初始化线性层的权重 适用于ReLU激活函数的网络 防止梯度消失或梯度爆炸问题，保持前向传播和反向传播中信号的方差稳定
                    if m.bias is not None:  # 检查该线性层是否有偏置项（bias）
                        nn.init.zeros_(m.bias)  # 如果存在偏置项，则将其初始化为0

    def forward(self, x, return_features=False, return_attentions=False):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        # x = x.reshape(b * t, 3, h, w)  # 将所有时间帧作为单独的图像批次处理
        x_reshaped = x.reshape(b * t, 3, h, w)  # 添加这行代码定义 x_reshaped

        # src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        src, pos = self.backbone(x_reshaped)
        _, c, oh, ow = src.shape

        ####
        # 提取经过TSSA处理的高阶运动统计量特征
        high_order_features = None
        if hasattr(self.backbone[0], 'extract_high_order_motion_features') and self.backbone[
            0].optical_flow_extractor is not None:
            high_order_features = self.backbone[0].extract_high_order_motion_features(x_reshaped)

        # ---------------------------------------------------------------------
        # 核心 2：使用 self.ffm 调用
        # ---------------------------------------------------------------------
        fused_features = self.ffm(src, src, high_order_features)
        src = fused_features
        # ---------------------------------------------------------------------

        # src = self.input_proj(src)   将 src 的通道数从 c 调整到 token_encoder.d_model
        representations, att = self.token_encoder(src, None, self.query_embed.weight, pos)  # 特征处理
        # [1, B x T, K, F'], [1, B x T, K, H' x W']  K: token数量，F': 特征维度

        representations = representations.reshape(b, t, self.num_tokens, -1)                    # [B, T, K, D]

        if self.dataset == 'volleyball':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)   # 在时间维度上取平均值，聚合时间信息  [B x K, D]
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations  # 设置自注意力机制的查询、键和值
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations = torch.mean(representations, dim=1)  # 在token维度上去取平均值，聚合token信息  [B, D]
        elif self.dataset == 'nba':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = self.conv3(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)                                # [B x K, D]
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations = torch.mean(representations, dim=1)                                # [B, D]

        representations = representations.reshape(b, -1)  # 对representations进行展平，将所有特征向量堆叠成二维向量

        #### 保存用于可视化的特征
        if return_features:
            features = representations.clone().detach()  # .clone()表示复制一份 .detach()表示不进行梯度计算
        ####

        activities_scores = self.classifier(representations)  # 将 representations 输入分类器（通常是一个全连接层 nn.Linear），得到分类得分。   [B, C]

        if return_attentions:
            return activities_scores, features, att
        elif return_features:
            return activities_scores, features  # 返回分类得分和用于可视化的特征
        else:
            return activities_scores


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim

        # feature extraction
        self.backbone = build_backbone(args)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.backbone.num_channels, self.num_class)

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape

        representations = self.avg_pool(src)
        representations = representations.reshape(b, t, c)

        representations = representations.reshape(b * t, self.backbone.num_channels)        # [B, T, F]
        activities_scores = self.classifier(representations)
        activities_scores = activities_scores.reshape(b, t, -1).mean(dim=1)

        return activities_scores