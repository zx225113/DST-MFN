import torch
import torch.nn as nn
from einops import rearrange


class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)  # For q, k, and v

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        # Compute q, k, v with linear layer and rearrange
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        b, h, N, d = w.shape

        # Normalize w
        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        # Rearrange back and pass through the output layer
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


if __name__ == "__main__":
    # 参数设置
    batch_size = 1  # batch size
    seq_len = 512  # 序列长度
    dim = 64  # 特征维度
    num_heads = 8  # 注意力头数

    # 创建 AttentionTSSA 模块
    attention_tssa = AttentionTSSA(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
    print(attention_tssa)

    # 生成随机输入张量 [batch_size, seq_len, dim]
    x = torch.randn(batch_size, seq_len, dim)

    # 打印输入和输出张量的形状
    print("Input shape:", x.shape)

    # 前向传播计算输出
    output = attention_tssa(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)
