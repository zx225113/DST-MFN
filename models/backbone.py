# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

from spatial_correlation_sampler import SpatialCorrelationSampler

from .position_encoding import build_position_encoding

from .optical_flow import OpticalFlowExtractor
class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        backbone = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation], pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 局部相关性模块
        if self.motion:
            self.layer_channel = [64, 128, 256, 512]

            self.channel_dim = self.layer_channel[self.motion_layer - 1]

            # self.corr_input_proj = nn.Sequential(
            #     nn.Conv2d(self.channel_dim, self.corr_dim, kernel_size=1, bias=False),
            #     nn.ReLU()
            # )

            ####
            # 添加缺失的属性定义
            self.corr_input_proj1 = nn.Sequential(
                nn.Conv2d(self.layer_channel[1], self.corr_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )
            self.corr_input_proj2 = nn.Sequential(
                nn.Conv2d(self.layer_channel[3], self.corr_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )
            ####

            self.neighbor_size = args.neighbor_size
            self.ps = 2 * args.neighbor_size + 1

            #  pyTorch 自定义操作，用于高效的计算两个特征图之间的相关性
            self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                                 stride=1, padding=0, dilation_patch=1)

            # self.corr_output_proj = nn.Sequential(
            #     nn.Conv2d(self.ps * self.ps, self.channel_dim, kernel_size=1, bias=False),
            #     nn.ReLU()
            # )
            ####
            # 添加缺失的输出投影层
            self.corr_output_proj1 = nn.Sequential(
                nn.Conv2d(self.ps * self.ps, self.layer_channel[2], kernel_size=1, bias=False),
                nn.ReLU()
            )
            self.corr_output_proj2 = nn.Sequential(
                nn.Conv2d(self.ps * self.ps, self.layer_channel[3], kernel_size=1, bias=False),
                nn.ReLU()
            )
            ####
        ####
        # 添加光流提取器
        if hasattr(args, 'use_optical_flow') and args.use_optical_flow:
            self.optical_flow_extractor = OpticalFlowExtractor(args)
        else:
            self.optical_flow_extractor = None
        ####
    # def get_local_corr(self, x):   # 计算局部相关性
    #     x = self.corr_input_proj(x)     # [B*T,C,H,W]
    #     # print("1",x.shape)
    #     x = F.normalize(x, dim=1)
    #     # print("2",x.shape)
    #     x = x.reshape((-1, self.num_frames) + x.size()[1:])  # [B,T,C,H,W]
    #     # print("3",x.shape)
    #     b, t, c, h, w = x.shape
    #
    #     x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]
    #     # print("4",x.shape)
    #     # new implementation
    #     x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)        # [B*T,C,H,W]
    #     # print("5",x_pre.shape)
    #     x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)  # [B*T,C,H,W]
    #     # print("6",x_post.shape)
    #     corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
    #     # print("7",corr.shape)
    #     corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
    #     # print("8",corr.shape)
    #     corr = F.relu(corr)
    #     # print("9",corr.shape)
    #     corr = self.corr_output_proj(corr)
    #     # print("10",corr.shape)
    #     return corr
    ####
    def get_local_corr(self, x, idx):
        if idx == 0:
            x = self.corr_input_proj1(x)
        else:
            x = self.corr_input_proj2(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)

        corr = self.correlation_sampler(x_pre, x_post)
        corr = corr.view(-1, self.ps * self.ps, h, w)
        corr = F.relu(corr)

        if idx == 0:
            corr = self.corr_output_proj1(corr)
        else:
            corr = self.corr_output_proj2(corr)
        return corr

    def extract_high_order_motion_features(self, x):
        """
        提取高阶运动特征并经过TSSA处理

        Args:
            x: 输入帧序列 [B*T, C, H, W]

        Returns:
            high_order_features: 经过TSSA处理的高阶运动特征 [B*T, hidden_dim, H, W]
        """
        if self.optical_flow_extractor is None:
            return None

        b, c, h, w = x.shape
        t = self.num_frames
        b_batch = b // t

        # 重塑为 [B, T, C, H, W]
        x_reshaped = x.view(b_batch, t, c, h, w)

        processed_features_list = []

        # 对每个批次计算高阶运动统计量并经过TSSA处理
        for i in range(b_batch):
            frame_seq = x_reshaped[i]  # [T, C, H, W]

            # 计算高阶运动统计量并经过TSSA处理
            processed_features = self.optical_flow_extractor.process_frame_sequence(frame_seq)  # [T, hidden_dim, H, W]

            processed_features_list.append(processed_features)

        if processed_features_list:
            # 合并所有批次的结果 [B, T, hidden_dim, H', W']
            combined_features = torch.stack(processed_features_list, dim=0)

            # ---------------------------------------------------------------
            # 关键修改：修复 RuntimeError (view size not compatible)
            # 1. 使用 flatten(0, 1) 代替 view(b, ...)，自动合并 [B, T] 维度。
            # 2. 自动适应降采样后的新 H', W'，不再依赖原始的 h, w。
            # 3. flatten/reshape 自动处理非连续内存问题。
            # ---------------------------------------------------------------
            combined_features = combined_features.flatten(0, 1)  # [B*T, hidden_dim, H', W']

            return combined_features
        else:
            return None
    ####

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # if self.motion:
        #     if self.motion_layer == 1:
        #         corr = self.get_local_corr(x)
        #         x = x + corr
        #         x = self.layer2(x)
        #         x = self.layer3(x)
        #         x = self.layer4(x)
        #     elif self.motion_layer == 2:
        #         x = self.layer2(x)
        #         corr = self.get_local_corr(x)
        #         x = x + corr
        #         x = self.layer3(x)
        #         x = self.layer4(x)
        #     elif self.motion_layer == 3:
        #         x = self.layer2(x)
        #         x = self.layer3(x)
        #         corr = self.get_local_corr(x)
        #         x = x + corr
        #         x = self.layer4(x)
        #     elif self.motion_layer == 4:
        #         x = self.layer2(x)
        #         x = self.layer3(x)
        #         x = self.layer4(x)
        #         corr = self.get_local_corr(x)
        #         x = x + corr
        #     else:
        #         assert False
        if self.motion:
            if self.motion_layer == 1:
                corr = self.get_local_corr(x, 0)
                x = x + corr
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
            elif self.motion_layer == 2:
                x = self.layer2(x)
                corr = self.get_local_corr(x, 0)
                x = x + corr
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
            elif self.motion_layer == 3:
                x = self.layer2(x)
                x = self.layer3(x)
                corr = self.get_local_corr(x, 0)
                x = x + corr
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
            elif self.motion_layer == 4:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
                corr = self.get_local_corr(x, 1)
                x = x + corr
            else:
                assert False
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x


class MultiCorrBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args):
        super(MultiCorrBackbone, self).__init__()

        backbone = getattr(torchvision.models, args.backbone)(  # getattr 动态获取 torchvision.models 模块中由 args.backbone 指定的预训练模型(默认：resnet18)
            replace_stride_with_dilation=[False, False, args.dilation],  # 在最后一个残差块使用空洞卷积来代替步幅
            pretrained=True)  # pretrained=True 表示加载在 ImageNet 上预训练的权重

        self.num_frames = args.num_frame  # 输入的帧数：18
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048  # 如果backbone 为 resnet18 或 resnet34，则通道数等于 512；否则等于 2048

        # 保存运动相关性计算相关的配置参数
        self.motion = args.motion  # 运动特征开关
        self.motion_layer = args.motion_layer  # 运动相关性计算模块所在层
        self.corr_dim = args.corr_dim  # 运动相关性计算模块的通道数

        # 从预训练的 backbone 中提取各个组件并保存为实例变量，用于构建前向传播过程
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1  # BatchNorm
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 残差块
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer_channel = [64, 128, 256, 512]  # 定义 ResNet 各个阶段的通道数列表 64: layer1的通道数, 128: layer2的通道数, 256: layer3的通道数, 512: layer4的通道数

        self.channel_dim = self.layer_channel[self.motion_layer - 1]  # self.channel_dim 来存储当前运动相关层的通道维度 使用 self.motion_layer - 1 作为索引访问 self.layer_channel 列表

        self.corr_input_proj1 = nn.Sequential(  # 定义输入映射层1
            nn.Conv2d(self.layer_channel[2], self.corr_dim, kernel_size=1, bias=False),
            # 输入通道数: self.layer_channel[2] = 256 输出通道数: self.corr_dim = 64 kernel_size=1: 使用1x1卷积核，只在通道维度进行变换 bias=False: 不使用偏置项
            nn.ReLU()  # 使用 ReLU 激活函数，对输入进行非线性映射
        )
        self.corr_input_proj2 = nn.Sequential(  # 定义输入映射层2
            nn.Conv2d(self.layer_channel[3], self.corr_dim, kernel_size=1, bias=False),
            # 输入通道数: self.layer_channel[3] = 512 输出通道数: self.corr_dim = 64 kernel_size=1: 使用1x1卷积核，只在通道维度进行变换 bias=False: 不使用偏置项
            nn.ReLU()  # 使用 ReLU 激活函数，对输入进行非线性映射
        )

        self.neighbor_size = args.neighbor_size  # 在计算相关性时考虑的邻域范围：5
        self.ps = 2 * args.neighbor_size + 1  # 计算并存储patch size = 2 * 5 + 1 = 11 ，表示 11×11 的邻域窗口

        # 定义空间相关性采样器
        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps, # kernel_size=1: 相关性计算的卷积核大小为1x1 patch_size=self.ps: 邻域窗口大小
                                                             stride=1, padding=0, dilation_patch=1)  # stride=1: 步长为1，padding=0: 不进行填充，dilation_patch=1: 邻域窗口的膨胀系数为1

        self.corr_output_proj1 = nn.Sequential(  # 定义输出映射层1
            nn.Conv2d(self.ps * self.ps, self.layer_channel[2], kernel_size=1, bias=False), # 输入通道数为ps*ps=11×11=121，输出通道数为layer_channel[2]= 256
            nn.ReLU()  # 激活函数ReLU，对卷积结果进行非线性映射
        )
        self.corr_output_proj2 = nn.Sequential(  # 定义输出映射层2
            nn.Conv2d(self.ps * self.ps, self.layer_channel[3], kernel_size=1, bias=False), # 输入通道数为ps*ps=11×11=121，输出通道数为layer_channel[3]= 512
            nn.ReLU()  # 激活函数ReLU，对卷积结果进行非线性映射
        )

        ####
        # 添加光流提取器
        if hasattr(args, 'use_optical_flow') and args.use_optical_flow:
            self.optical_flow_extractor = OpticalFlowExtractor(args)
        else:
            self.optical_flow_extractor = None
        ####

    def get_local_corr(self, x, idx): # 计算视频序列中相邻帧间的局部相关性 idx: 索引参数，用于区分使用哪个投影层（0或1）
        if idx == 0:  # idx为0时，进行映射层1
            x = self.corr_input_proj1(x)  # 256 -> 64
        else:  # idx为1时，进行映射层2
            x = self.corr_input_proj2(x)  # 512 -> 64
        x = F.normalize(x, dim=1)   # 对输入在通道维数上进行归一化处理  [B * T , C , H , W]
        x = x.reshape((-1, self.num_frames) + x.size()[1:])  # x.size()[1:] 表示获取第 1 个维度之后的维度，即 [C,H,W]
        # -1 是 reshape 中的一个特殊值，表示这个维度会被自动推断出来。 + 表示元组拼接，最终构成新的维度形状：(-1, self.num_frames, C, H, W)
        # x.shape(),将原来的张量重塑成新的形状, [B * T , C , H , W] -> [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # .contiguous(): 确保张量在内存中连续存储  [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        # x[:, :, :] 复制所有数据 .permute(0, 2, 1, 3, 4) 重新排列为 [B, T, C, H, W] 格式 .view(-1, c, h, w)展平为 [B*T, C, H, W] 格式
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        # x[:, :, 1:]: 取第2帧到最后一帧 [B, C, T-1, H, W]
        # x[:, :, -1:]: 取最后一帧（复制作为边界处理）[B, C, 1, H, W]
        # torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2) 在时间维度上拼接  [B, C, T, H, W]
        # .permute(0, 2, 1, 3, 4) 重新排列为 [B, T, C, H, W] 格式 .view(-1, c, h, w)展平为 [B*T, C, H, W] 格式

        corr = self.correlation_sampler(x_pre, x_post)  # 使用相关性采样器计算相邻帧间的相关性 # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)   # 重塑相关性张量形状                # [B x T, P * P, H, W]
        corr = F.relu(corr)  # 对相关性结果应用ReLU激活函数，将负值置为0

        if idx == 0:
            corr = self.corr_output_proj1(corr)  # 使用卷积层将相关性结果映射为输出  [B x T, P * P, H, W] -> [B x T, C=self.layer_channel[2], H, W]
        else:
            corr = self.corr_output_proj2(corr)  # 使用卷积层将相关性结果映射为输出  [B x T, P * P, H, W] -> [B x T, C=self.layer_channel[3], H, W]
        return corr

    ####
    def extract_high_order_motion_features(self, x):
        """
        提取高阶运动特征并经过TSSA处理

        Args:
            x: 输入帧序列 [B*T, C, H, W]

        Returns:
            high_order_features: 经过TSSA处理的高阶运动特征 [B*T, hidden_dim, H, W]
        """
        if self.optical_flow_extractor is None:
            return None

        b, c, h, w = x.shape
        t = self.num_frames
        b_batch = b // t

        # 重塑为 [B, T, C, H, W]
        x_reshaped = x.view(b_batch, t, c, h, w)

        processed_features_list = []

        # 对每个批次计算高阶运动统计量并经过TSSA处理
        for i in range(b_batch):
            frame_seq = x_reshaped[i]  # [T, C, H, W]

            # 计算高阶运动统计量并经过TSSA处理
            processed_features = self.optical_flow_extractor.process_frame_sequence(frame_seq)  # [T, hidden_dim, H, W]

            processed_features_list.append(processed_features)

        if processed_features_list:
            # [B, T, hidden_dim, H', W']
            combined_features = torch.stack(processed_features_list, dim=0)
            # ---------------------------------------------------------------
            # 修复核心：使用 flatten(0, 1) 代替 view(b, -1, h, w)
            # ---------------------------------------------------------------
            combined_features = combined_features.flatten(0, 1)  # [B*T, hidden_dim, H', W']
            return combined_features
        else:
            return None
    ####

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #
    #     corr = self.get_local_corr(x, 0)
    #     x = x + corr
    #
    #     x = self.layer4(x)
    #
    #     corr = self.get_local_corr(x, 1)
    #     x = x + corr
    #
    #     return x
    ####
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if self.motion:
            if self.motion_layer == 1:
                corr = self.get_local_corr(x, 0)
                x = x + corr
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
            elif self.motion_layer == 2:
                x = self.layer2(x)
                corr = self.get_local_corr(x, 0)
                x = x + corr
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
            elif self.motion_layer == 3:
                x = self.layer2(x)
                x = self.layer3(x)
                corr = self.get_local_corr(x, 0)
                x = x + corr
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
            elif self.motion_layer == 4:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x, 1)
                x = x + corr
                corr = self.get_local_corr(x, 1)
                x = x + corr
            else:
                assert False
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x
    ####

class Joiner(nn.Sequential):  # 义一个名为 Joiner 的类，继承自 PyTorch 的 nn.Sequential nn.Sequential 是一个容器，可以按顺序存储和执行多个模块
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)  # 前向传播时会按顺序执行这两个模块

    def forward(self, x):
        features = self[0](x)  # self[0] 指向序列中的第一个模块 通过骨干网络处理输入 x，提取特征
        pos = self[1](features).to(x.dtype)  # self[1] 指向序列中的第二个模块 将提取的特征传给位置编码模块生成位置信
        # .to(x.dtype) 将位置编码的数据类型转换为与输入 x 相同的类型
        return features, pos

def build_backbone(args):  # 定义一个名为 build_backbone 的函数，构建整个骨干网络模型的工厂函数
    position_embedding = build_position_encoding(args)  # 调用 build_position_encoding 函数
    # 返回值： pos:[b,PE,h,w]
    if args.multi_corr:
        backbone = MultiCorrBackbone(args)  # 创建一个 MultiCorrBackbone 实例，一个具有多个相关性计算层的骨干网络
    else:
        backbone = Backbone(args)  # 创建一个 Backbone 实例，一个基本的骨干网络，只在指定层计算一次运动相关性
    model = Joiner(backbone, position_embedding)  # 创建一个 Joiner 实例，将骨干网络和位置编码器连接起来 作用是在前向传播时先通过骨干网络提取特征，然后为这些特征生成位置编码
    model.num_channels = backbone.num_channels  #  将骨干网络的通道数属性复制到组合模型上 这样外部代码可以通过 model.num_channels 访问骨干网络的输出通道数
    return model
