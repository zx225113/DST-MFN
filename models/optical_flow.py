# 文件: models/optical_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from .TSSA import AttentionTSSA
from .position_encoding import build_position_encoding


class DeepFlowProcessor:
    """
    DeepFlow光流处理器
    """

    def __init__(self):
        # 初始化DeepFlow算法
        try:
            self.deepflow = cv2.optflow.createOptFlow_DeepFlow()
        except AttributeError:
            self.deepflow = None

    def compute_flow(self, prev_frame, next_frame):
        """
        计算两帧之间的光流场 (包含降采样加速)

        Args:
            prev_frame: 前一帧图像 (H, W, 3) numpy array
            next_frame: 后一帧图像 (H, W, 3) numpy array

        Returns:
            flow: 光流场 (H, W, 2) numpy array (已恢复到原始尺寸)
        """
        # --- 优化开始：降采样 ---
        h, w = prev_frame.shape[:2]
        # 将长边缩放到 256 像素，大幅减少计算量
        target_size = 256.0
        scale = min(target_size / h, target_size / w)

        # 如果图像本来就小，就不缩放了
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            prev_small = cv2.resize(prev_frame, (new_w, new_h))
            next_small = cv2.resize(next_frame, (new_w, new_h))
        else:
            prev_small = prev_frame
            next_small = next_frame
            scale = 1.0

        # 转换为灰度图
        if len(prev_small.shape) == 3:
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_small, cv2.COLOR_RGB2GRAY)
        else:
            prev_gray = prev_small
            next_gray = next_small

        # 计算光流 (小图上计算，速度快)
        if self.deepflow is not None:
            flow_small = self.deepflow.calc(prev_gray, next_gray, None)
        else:
            flow_small = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # --- 优化结束：恢复尺寸 ---
        if scale < 1.0:
            # 将光流图放大回原始尺寸
            flow = cv2.resize(flow_small, (w, h))
            # 光流的值表示位移像素数，图像放大了，位移值也要按比例放大
            flow = flow * (1.0 / scale)
        else:
            flow = flow_small

        return flow

    def compute_bidirectional_flows(self, frames):
        """
        计算双向光流场
        """
        forward_flows = []
        backward_flows = []

        # 计算前向光流 (t -> t+1)
        for i in range(len(frames) - 1):
            flow = self.compute_flow(frames[i], frames[i + 1])
            forward_flows.append(flow)

        # 计算后向光流 (t -> t-1)
        for i in range(len(frames) - 1, 0, -1):
            flow = self.compute_flow(frames[i], frames[i - 1])
            backward_flows.append(flow)

        return forward_flows, backward_flows


class OpticalFlowExtractor(nn.Module):
    """
    光流特征提取器，用于计算高阶运动统计量
    """

    def __init__(self, args):
        super(OpticalFlowExtractor, self).__init__()
        self.num_frames = args.num_frame
        self.deepflow_processor = DeepFlowProcessor()

        # 添加位置编码
        self.position_embedding = build_position_encoding(args)

        # 添加TSSA注意力机制处理高阶运动统计量
        self.tssa = AttentionTSSA(dim=args.hidden_dim, num_heads=args.nheads, attn_drop=args.drop_rate,
                                  proj_drop=args.drop_rate)

        # 投影层，将6通道高阶运动统计量投影到hidden_dim通道
        self.high_order_proj = nn.Sequential(
            nn.Conv2d(6, args.hidden_dim, kernel_size=1),
            nn.ReLU()
        )

    def numpy_to_tensor(self, flow):
        """
        将numpy光流转换为tensor
        """
        return torch.from_numpy(flow).float().permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)

    def tensor_to_numpy(self, flow_tensor):
        """
        将tensor光流转换为numpy
        """
        return flow_tensor.permute(1, 2, 0).cpu().numpy()  # (2, H, W) -> (H, W, 2)

    def compute_acceleration_from_flows(self, flow_forward, flow_backward):
        """
        从光流计算加速度场
        A(t) = V(t,t+Δt) + V(t,t-Δt)
        """
        # 转换为numpy进行计算后再转回tensor
        flow_fwd_np = self.tensor_to_numpy(flow_forward)
        flow_bwd_np = self.tensor_to_numpy(flow_backward)

        # 计算加速度
        acceleration_np = flow_fwd_np + flow_bwd_np

        # 转换回tensor
        acceleration = self.numpy_to_tensor(acceleration_np)
        return acceleration

    def compute_jerk_from_accelerations(self, acc_forward, acc_backward):
        """
        从加速度计算急动度场
        J(t) = A(t,t+Δt) - A(t-Δt,t)
        """
        acc_fwd_np = self.tensor_to_numpy(acc_forward)
        acc_bwd_np = self.tensor_to_numpy(acc_backward)

        # 计算急动度
        jerk_np = acc_fwd_np - acc_bwd_np

        # 转换回tensor
        jerk = self.numpy_to_tensor(jerk_np)
        return jerk

    def compute_snap_from_jerks(self, jerk_forward, jerk_backward):
        """
        从急动度计算加加速度场
        S(t) = J(t,t+Δt) + J(t,t-Δt)
        """
        jerk_fwd_np = self.tensor_to_numpy(jerk_forward)
        jerk_bwd_np = self.tensor_to_numpy(jerk_backward)

        # 计算加加速度
        snap_np = jerk_fwd_np + jerk_bwd_np

        # 转换回tensor
        snap = self.numpy_to_tensor(snap_np)
        return snap

    def process_frame_sequence(self, frame_sequence):
        """
        处理帧序列，计算高阶运动统计量并经过TSSA处理
        """
        # 获取设备信息
        device = frame_sequence.device

        # 确保输入数据在正确的设备上
        if frame_sequence.device != device:
            frame_sequence = frame_sequence.to(device)

        # 转换为numpy格式以便使用OpenCV
        frames_np = []
        for i in range(frame_sequence.shape[0]):
            # 反归一化并转换为uint8
            frame = frame_sequence[i].cpu().numpy()
            frame = np.transpose(frame, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frames_np.append(frame)

        # 计算光流
        forward_flows, backward_flows = self.deepflow_processor.compute_bidirectional_flows(frames_np)

        # 转换光流为tensor格式
        forward_flows_tensors = [self.numpy_to_tensor(flow) for flow in forward_flows]
        backward_flows_tensors = [self.numpy_to_tensor(flow) for flow in backward_flows]

        # 计算高阶运动统计量
        accelerations = []
        jerks = []
        snaps = []

        # 计算加速度场
        min_len = min(len(forward_flows_tensors), len(backward_flows_tensors))
        for i in range(min_len):
            acc = self.compute_acceleration_from_flows(
                forward_flows_tensors[i],
                backward_flows_tensors[len(backward_flows_tensors) - 1 - i]
            )
            accelerations.append(acc)

        # 计算急动度场
        for i in range(len(accelerations) - 1):
            if i < len(accelerations) - 1:
                jerk = self.compute_jerk_from_accelerations(
                    accelerations[i],
                    accelerations[len(accelerations) - 1 - i]
                )
                jerks.append(jerk)

        # 计算加加速度场
        for i in range(len(jerks) - 1):
            if i < len(jerks) - 1:
                snap = self.compute_snap_from_jerks(
                    jerks[i],
                    jerks[len(jerks) - 1 - i]
                )
                snaps.append(snap)

        # 选择合适的高阶统计量进行后续处理
        if snaps and len(snaps) > 0:
            mid_idx = len(snaps) // 2
            selected_snap = snaps[mid_idx]
            selected_jerk = jerks[mid_idx] if jerks and len(jerks) > mid_idx else torch.zeros_like(selected_snap)
            selected_acc = accelerations[mid_idx] if accelerations and len(
                accelerations) > mid_idx else torch.zeros_like(selected_snap)
        elif jerks and len(jerks) > 0:
            mid_idx = len(jerks) // 2
            selected_jerk = jerks[mid_idx]
            selected_acc = accelerations[mid_idx] if accelerations and len(
                accelerations) > mid_idx else torch.zeros_like(selected_jerk)
            selected_snap = torch.zeros_like(selected_jerk)
        elif accelerations and len(accelerations) > 0:
            mid_idx = len(accelerations) // 2
            selected_acc = accelerations[mid_idx]
            selected_jerk = torch.zeros_like(selected_acc)
            selected_snap = torch.zeros_like(selected_acc)
        else:
            _, h, w = frame_sequence.shape[1:]
            selected_acc = torch.zeros(2, h, w)
            selected_jerk = torch.zeros(2, h, w)
            selected_snap = torch.zeros(2, h, w)

        # 拼接高阶运动统计量 (加速度、急动度、加加速度)
        combined_stats = torch.cat([selected_acc, selected_jerk, selected_snap], dim=0)  # [6, H, W]

        # 确保combined_stats在正确设备上
        if combined_stats.device != device:
            combined_stats = combined_stats.to(device)

        # 添加批次维度
        combined_stats = combined_stats.unsqueeze(0).repeat(frame_sequence.shape[0], 1, 1, 1)  # [T, 6, H, W]

        # -------------------------------------------------------------------------
        # 下采样以避免显存溢出 (OOM)
        # -------------------------------------------------------------------------
        combined_stats = F.interpolate(combined_stats, scale_factor=0.0625, mode='bilinear', align_corners=False)

        # 投影到隐藏维度
        projected_features = self.high_order_proj(combined_stats)  # [T, hidden_dim, H', W']

        # 添加位置编码
        pos_embed = self.position_embedding(projected_features)  # [T, hidden_dim, H', W']
        features_with_pos = projected_features + pos_embed

        # 重塑为序列格式以应用TSSA注意力
        t, c, h, w = features_with_pos.shape
        features_seq = features_with_pos.view(t, c, -1).permute(0, 2, 1)

        # 应用TSSA线性注意力机制
        attended_features = self.tssa(features_seq)  # [T, H'*W', hidden_dim]

        # 重塑回图像格式
        processed_features = attended_features.permute(0, 2, 1).view(t, c, h, w)

        return processed_features
