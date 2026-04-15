# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
# ------------------------------------------------------------------------

import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        # num_pos_fears=N_steps,temperature=10000:温度参数，控制位置编码的频率,normalize=True,scale=None:归一化时的缩放因子
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")  # 如果normalize为False，则不允许设置缩放因子
        if scale is None:
            scale = 2 * math.pi  # 计算位置编码的缩放因子，通常为2π
        self.scale = scale

    def forward(self, x):
        bs, c, h, w = x.shape  # 获取输入张量的维度信息

        y_embed = torch.arange(1, h + 1, device=x.device).unsqueeze(0).unsqueeze(2)
        # torch.arange(1, h + 1, device=x.device)：创建从1到h的序列，放在相同设备上
        # .unsqueeze(0)：在第0维增加一个维度
        # .unsqueeze(2)：在第2维增加一个维度  [1,h,1]
        y_embed = y_embed.repeat(bs, 1, w)  # 将y坐标扩展到完整的batch和width维度 repeat(bs, 1, w) 表示在各维度上重复的次数 [bs,h,w]
        x_embed = torch.arange(1, w + 1, device=x.device).unsqueeze(0).unsqueeze(1)
        # torch.arange(1, w + 1, device=x.device)：创建从1到w的序列，放在相同设备上
        # .unsqueeze(0)：在第0维增加一个维度
        # .unsqueeze(1)：在第1维增加一个维度  [1,1,w]
        x_embed = x_embed.repeat(bs, h, 1)  # # 将x坐标扩展到完整的batch和height维度 repeat(bs, h, 1) 表示在各维度上重复的次数 [bs,h,w]

        if self.normalize:
            eps = 1e-6  # 防止除零的小数值
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 每一行都除以最后一列，并乘以缩放因子
            # y_embed[:, -1:, :]：高级索引切片 ‘:’：选择所有batch ‘-1:’：选择最后一行（注意冒号，表示保持维度）‘:’：选择所有列
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 每一列都除以最后一列，并乘以缩放因子
            # x_embed[:, :, -1:]：高级索引切片 ‘:’：选择所有batch ‘:’：选择所有行‘-1:’：选择最后一列（注意冒号，表示保持维度）

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # torch.arange(self.num_pos_feats)：创建从0到self.num_pos_feats-1的序列 dtype=torch.float32：指定数据类型为32位浮点数 device=x.device：指定张量所在的设备
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        # torch.div(a, b, rounding_mode='floor')：整数除法，向下取整 **：幂运算符

        pos_x = x_embed[:, :, :, None] / dim_t  # x_embed[:, :, :, None]：在最后一个维度添加新维度
        pos_y = y_embed[:, :, :, None] / dim_t  # y_embed[:, :, :, None]：在最后一个维度添加新维度
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 生成正弦/余弦位置编码
        # 0::2：从索引0开始，步长为2（选取偶数索引：0,2,4...） 1::2：从索引1开始，步长为2（选取奇数索引：1,3,5...）
        # torch.stack(..., dim=4)：在第4维堆叠两个张量 .flatten(3)：将第3维及之后的所有维度展平
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 生成正弦/余弦位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # torch.cat((pos_y, pos_x), dim=3)：在第3维拼接y和x的位置编码 .permute(0, 3, 1, 2)：重新排列维度顺序
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)  # 行位置嵌入，大小为50×num_pos_feats
        self.col_embed = nn.Embedding(50, num_pos_feats)  # 列位置嵌入，大小为50×num_pos_feats
        self.reset_parameters()  # 调用参数初始化方法，初始化嵌入层的权重参数

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        # nn.init.uniform_：PyTorch提供的均匀分布初始化函数
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]   # 获取输入张量的最后两个维度
        i = torch.arange(w, device=x.device)  # 创建从0到w-1的列索引序列
        j = torch.arange(h, device=x.device)  # 创建从0到h-1的行索引序列
        x_emb = self.col_embed(i)  # 将列索引序列映射为列位置编码  [w, num_pos_feats]
        y_emb = self.row_embed(j)  # 将行索引序列映射为行位置编码  [h, num_pos_feats]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),  # [h,w,num_pos_feats]
            # x_emb.unsqueeze(0) : 在第0维（最前面）增加一个维度
            # ：repeat(h, 1, 1)表示第0维重复h次，第1维重复1次，第2维重复1次
            y_emb.unsqueeze(1).repeat(1, w, 1), # [h,w,num_pos_feats]
            # y_emb.unsqueeze(1) : 在第1维（中间）增加一个维度
            # ：repeat(1, w, 1)表示第0维重复1次，第1维重复w次，第2维重复1次
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # dim=-1表示在最后一个维度拼接 [h, w, 2*num_pos_feats]
        # ：permute(2, 0, 1)表示原来的第2维变为第0维，第0维变为第1维，第1维变为第2维 [2*num_pos_feats, h, w]
        # .unsqueeze(0) ：在第0维增加一个维度 [1, 2*num_pos_feats, h, w]
        # .repeat(x.shape[0], 1, 1, 1) 表示第0维重复batch_size次 [batch_size, 2*num_pos_feats, h, w]
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2  # 计算位置编码的步数，通常是隐藏维度的一半  256 // 2 = 128
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)  # 使用正弦和余弦函数生成位置编码，这是Transformer中经典的位置编码方式
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)  # 使用可学习的位置编码器，这是DeformableDETR中的位置编码方式
    else:
        raise ValueError(f"not supported {args.position_embedding}")  # 捕获异常

    return position_embedding



def build_index_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.index_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.index_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
