import torch
import torch.nn as nn
import torch.nn.functional as F


class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out


class FFM(nn.Module):
    def __init__(self, dim1, dim2, res, use_high_order=False):
        """
        Args:
            dim1: 输入特征通道数
            dim2: 隐藏层维度
            res: 保留接口参数
            use_high_order: 是否使用了高阶运动特征(Optical Flow)
        """
        super().__init__()
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        self.concat = nn.Conv2d(dim2 * 2, dim2, 1)

        # 根据是否使用高阶特征动态调整输入通道数
        fusion_in_channels = dim2 * 5 if use_high_order else dim2 * 4

        self.fusion = nn.Sequential(
            IDSC(fusion_in_channels, dim2),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
            DSC(dim2, dim2),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
            nn.Conv2d(dim2, dim2, 1),
            nn.BatchNorm2d(dim2),
            nn.GELU()
        )

    def forward(self, x, y, high_order_features=None):
        # 1. 初始转换
        x = self.trans_c(x)
        if y.dim() == 3:
            # 如果y是3D的，需要重塑回4D。假设顺序与x一致
            b_y, n_y, c_y = y.shape
            h_x, w_x = x.shape[2], x.shape[3]
            y = y.permute(0, 2, 1).reshape(b_y, c_y, h_x, w_x)
        else:
            y = self.trans_c(y)

        # 2. 动态 Padding 处理
        # 解决 RuntimeError: shape invalid 问题
        # 确保 H 和 W 都能被 4 整除
        b, c, h, w = x.shape
        ws = 4  # Window size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        # 如果需要，进行 Padding
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            y = F.pad(y, (0, pad_w, 0, pad_h))

        # 获取 Padding 后的新尺寸
        B, C, H, W = x.shape  # H, W 现在是 4 的倍数

        # 3. 计算权重和初步融合
        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        # 4. Attention 计算 (使用填充后的 H, W)
        # 此时 H 和 W 均可被 4 整除，reshape 不会报错
        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, (H // 4) * (
                    W // 4), 8, 16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, (H // 4) * (
                    W // 4), 8, 16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, (H // 4) * (
                    W // 4), 8, 16, C // 8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, W // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)

        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, (H // 4) * (
                    W // 4), 8, 16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, (H // 4) * (
                    W // 4), 8, 16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, (H // 4) * (
                    W // 4), 8, 16, C // 8)

        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, W // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)

        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)

        # 5. 处理高阶运动特征
        if high_order_features is not None:
            # 必须插值到当前 Padding 后的尺寸 (H, W)，否则无法拼接
            if high_order_features.shape[-2:] != (H, W):
                high_order_features = F.interpolate(
                    high_order_features,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            combined_features = torch.cat([x, y, out1, out2, high_order_features], dim=1)
        else:
            combined_features = torch.cat([x, y, out1, out2], dim=1)

        out = self.fusion(combined_features)

        # 6. 裁剪回原始尺寸 (Crop)
        # 去掉之前填充的部分，恢复到 [b, c, h, w]
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out