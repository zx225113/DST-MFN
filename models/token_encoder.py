# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/transformer.py
# ------------------------------------------------------------------------
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6,  # 特征维度=512，8个注意力头，6个解码层
                 dim_feedforward=2048, dropout=0.1,  # 前馈神经网络的维度=2048，dropout概率0.1
                 activation="relu", normalize_before=False,  # 激活函数为relu 前向传播时是否先对输入进行归一化
                 return_intermediate_dec=False):  # 是否返回中间层的输出
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)  # 创建一个TransformerDecoderLayer实例，用于构建解码器层
        decoder_norm = nn.LayerNorm(d_model)  # 创建LayerNorm层，用于对解码器输出进行归一化 输入维度为 d_model（512）
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)  # 创建TransformerDecoder层，用于生成预测结果

        self._reset_parameters()  # 调用_reset_parameters方法，对模型进行参数初始化，确保模型的权重按特定方式初始化，从而提高训练效果。

        self.d_model = d_model  # 保存d_model参数的值
        self.nhead = nhead  # 保存nhead参数的值

    def _reset_parameters(self):
        for p in self.parameters():  # 遍历模型的所有参数
            if p.dim() > 1:  # 如果参数的维度大于1
                nn.init.xavier_uniform_(p)  # 对参数进行Xavier均匀分布初始化，Xavier 初始化有助于避免在训练初期梯度消失或爆炸的问题。

    def forward(self, src, mask, query_embed, pos_embed):
    #src: 输入特征图，通常是编码器提取的视觉特征 mask: 掩码张量，用于标记填充位置 query_embed: 查询嵌入，用于解码器的查询向量 pos_embed: 位置编码，提供空间位置信息
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) # src.flatten(2): 将特征图从 [B, C, H, W] 展平为 [B, C, H×W]，将空间维度合并
        # .permute(2, 0, 1): 重新排列维度为 [H×W, B, C]，将空间位置作为序列的第一个维度
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # pos_embed.flatten(2): 将位置编码从 [B, C, H, W] 展平为 [B, C, H×W]，将空间维度合并
        # .permute(2, 0, 1): 重新排列维度为 [H×W, B, C]，将空间位置作为序列的第一个维度
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # .unsqueeze(1): 在第1维添加一个维度 [num_quries,1,d_model]
        # .repeat(1, bs, 1): 在第一维重复bs次 [num_quries,bs,d_model]
        tgt = torch.zeros_like(query_embed)  # 创建与 query_embed 形状相同的全零张量作为目标序列的初始值
        hs, att = self.decoder(tgt, src, memory_key_padding_mask=mask,
                               pos=pos_embed, query_pos=query_embed)
        # 返回解码器输出 hs[1, num_queries, batch_size, d_model] 和注意力权重 (batch_size, num_heads, num_queries, num_keys)
        return hs.transpose(1, 2), att  # hs.transpose(1, 2): 交换第1和第2维度，调整输出格式为[1, batch_size, num_queries, d_model]


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)  # 使用 _get_clones 函数复制 num_layers 个 decoder_layer，创建解码器的所有层
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt  # 初始化输出为目标序列
        att = None  # 初始化注意力权重为 None

        # 创建列表用于存储中间层输出和注意力权重（如果需要）
        intermediate = []
        intermediate_att = []

        for layer in self.layers:  # 遍历每（6）层解码层
            output, att = layer(output, memory, tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                pos=pos, query_pos=query_pos)
            if self.return_intermediate:  # 是否返回中间结果
                intermediate.append(self.norm(output))
                intermediate_att.append(att)

        if self.norm is not None:  # 是否归一化
            output = self.norm(output)
            if self.return_intermediate:  # 是否返回中间结果
                intermediate.pop()  # 删除最后一个元素
                intermediate.append(output)  # 添加归一化后的结果

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_att)  # 如果需要返回中间输出，将列表堆叠成张量并返回

        return output.unsqueeze(0), att.unsqueeze(0) # 返回最终输出和注意力权重，添加一个维度以保持一致的输出格式


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # 初始化自注意力层，计算每个元素的加权和
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # 初始化交叉注意力层，计算每个元素的加权和
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 创建前馈网络的第一层线性变换 输入维度 d_model，输出维度 dim_feedforward（2048）
        self.dropout = nn.Dropout(dropout)  # 创建Dropout层，用于前馈网络中
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 创建前馈网络第二层线性变换 输入维度 dim_feedforward，输出维度 d_model

        self.norm1 = nn.LayerNorm(d_model)  # 自注意力的归一化层
        self.norm2 = nn.LayerNorm(d_model)  # 交叉自注意力归一化层
        self.norm3 = nn.LayerNorm(d_model)  # 前馈网络归一化层
        self.dropout1 = nn.Dropout(dropout)  # 自注意力的残差连接层
        self.dropout2 = nn.Dropout(dropout)  # 交叉自注意力的残差连接层
        self.dropout3 = nn.Dropout(dropout)  # 前馈网络残差连接层

        self.activation = _get_activation_fn(activation)  # 获取激活函数
        self.normalize_before = normalize_before  # 是否在计算前向传播时对输入进行归一化 True: 在注意力之前归一化（Pre-norm） False: 在注意力之后归一化（Post-norm）

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos  # 如果pos=None，则返回tensor，否则返回tensor加上pos


    def forward_post(self, tgt, memory,   # 定义后置归一化（Post-norm）的前向传播函数
                     # tgt：目标序列（解码器的输入序列）[bs,tgt_seq_len,d_model] memory：编码器输出（源序列）[bs,tgt_seq_len,d_model]
                     tgt_mask: Optional[Tensor] = None,  # 目标序列的掩码矩阵[bs,tgt_seq_len,tgt_seq_len]
                     memory_mask: Optional[Tensor] = None,  # 源序列的掩码矩阵[bs,tgt_seq_len,src_seq_len]
                     tgt_key_padding_mask: Optional[Tensor] = None,  # 目标序列的填充掩码矩阵[bs,tgt_seq_len]
                     memory_key_padding_mask: Optional[Tensor] = None,  # 源序列的填充掩码矩阵[bs,src_seq_len]
                     pos: Optional[Tensor] = None,  # 目标序列的位置编码矩阵[bs,tgt_seq_len,d_model]
                     query_pos: Optional[Tensor] = None):  # 源序列的位置编码矩阵[bs,src_seq_len,d_model]
                     # Optional[Tensor] = None 表示参数的数据类型可以是Tensor或者None

        q = k = self.with_pos_embed(tgt, query_pos)  # 创建一个 query（查询） 和 key（键）调用with_pos_embed函数将位置编码函数到目标序列
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,  # 调用self_attn函数进行自注意力计算
                              key_padding_mask=tgt_key_padding_mask)[0]
        # nn.MultiheadAttention()的输出是一个元组，（attn_output,attn_output_weights）
        # attn_output为自注意力输出 [L(目标序列长度),B（批次大小）,E（特征维度）]， attn_output_weights为自注意力权重 [B（批次大小）,L（目标序列长度）,S（源序列长度）]
        tgt = tgt + self.dropout1(tgt2)  # self.dropout1(tgt2)对自注意力结果进行随机丢弃防止过拟合，之后进行残差连接：将将自注意力输出与原始输入相加
        tgt = self.norm1(tgt)  # 对残差连接后的结果进行归一化，稳定训练过程，帮助加速收敛。
        tgt2, att = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  # 调用交叉注意力机制，计算计算目标序列(tgt)与记忆序列(memory)之间的注意力和注意力权重。
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)  # 将注意力输出 tgt2 经过dropout处理后与原始目标序列 tgt 进行残差连接
        tgt = self.norm2(tgt)  # 对残差连接后的结果进行层归一化处理
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 实现前馈神经网络(FFN)：   # self.linear1(tgt): 通过第一个线性层将维度从 d_model 扩展到 dim_feedforward
                                # self.activation(): 应用激活函数(如ReLU)
                                # self.dropout(): 应用dropout防止过拟合
                                # self.linear2(): 通过第二个线性层将维度从 dim_feedforward 压缩回 d_model
        tgt = tgt + self.dropout3(tgt2)  # 将前馈网络输出 tgt2 经过dropout处理后与之前的结果进行残差连接
        tgt = self.norm3(tgt)  # 对最终的残差连接结果进行层归一化
        return tgt, att  # 返回处理后的目标序列 tgt 和交叉注意力权重 att

    def forward_pre(self, tgt, memory,  # 定义使用前置归一化的前向传播函数
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)  # 对输入目标序列 tgt 进行第一层归一化处理
        q = k = self.with_pos_embed(tgt2, query_pos)  # 将位置编码添加到归一化后的目标序列，作为自注意力的查询(Q)和键(K)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,  # 计算自注意力
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)  # 将自注意力输出经过dropout后与原始输入进行残差连接
        tgt2 = self.norm2(tgt)  # 对残差连接后的结果进行第二层归一化
        tgt2, att = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),  # 计算交叉注意力和注意力权重
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)  # 将交叉注意力输出经过dropout后与残差连接后的结果进行残差连接
        tgt2 = self.norm3(tgt)  # 对残差连接后的结果进行第三层归一化
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))  # FFN
        tgt = tgt + self.dropout3(tgt2)  # 将FFN输出经过dropout后与残差连接后的结果进行残差连接
        return tgt, att  # 返回残差连接后的结果和注意力权重

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:  # 检查 normalize_before 属性，决定使用哪种归一化策略
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)  # 使用前置归一化的前向传播函数
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)  # 使用后置归一化的前向传播函数


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    # [copy.deepcopy(module) for i in range(N)]： 深拷贝N个module 复制对象及其所有嵌套子对象，确保每个副本完全独立，互不影响。
    # nn.ModuleList([...])：调用 PyTorch 的模块列表容器，传入一个列表 [...]，能自动注册子模块用于模型训练。

def build_token_encoder(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.drop_rate,
        nhead=args.nheads,
        dim_feedforward=args.ffn_dim,
        num_decoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
