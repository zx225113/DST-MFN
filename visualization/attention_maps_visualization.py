import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms  # 如果需要使用transforms.Normalize的默认参数

def visualize_attention_maps(attention_maps, first_batch_images, vis_save_path, args):
    """
    可视化注意力热力图并叠加到原始图像上

    参数:
    - attention_maps: 注意力图张量 [B, num_heads, num_queries, seq_len]
    - first_batch_images: 第一批图像张量 [B, C, H, W]
    - vis_save_path: 可视化结果保存路径
    - args: 命令行参数
    """
    if attention_maps is None or first_batch_images is None:
        print("No attention maps or images to visualize.")
        return

    print(f"Attention map shape: {attention_maps.shape}")
    print(f"First batch images shape: {first_batch_images.shape}")

    # 创建保存目录
    att_save_path = os.path.join(vis_save_path, 'attention_maps')
    os.makedirs(att_save_path, exist_ok=True)

    # 遍历样本、头、查询
    for sample_idx in range(min(5, first_batch_images.size(0))):
        # 获取原始图像（已反归一化）
        image_tensor = first_batch_images[sample_idx]  # [C, H, W]
        image = denormalize(image_tensor)  # [H, W, C]（0-255）

        # 获取注意力图
        sample_att = attention_maps[sample_idx]  # [num_heads, num_queries, seq_len]

        # 推断特征图尺寸
        seq_len = sample_att.size(-1)
        h_feat = int(np.sqrt(seq_len))
        w_feat = seq_len // h_feat

        # 遍历注意力头和查询
        for head_idx in range(min(4, sample_att.size(0))):
            for query_idx in range(min(4, sample_att.size(1))):
                # 提取注意力图并重塑
                att_map = sample_att[head_idx, query_idx]  # [seq_len]
                att_map = att_map[:h_feat * w_feat].view(h_feat, w_feat)  # [h_feat, w_feat]

                # 插值到原图尺寸
                att_map = F.interpolate(
                    att_map.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, h_feat, w_feat]
                    size=(image.shape[0], image.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()  # [H, W]

                # 归一化到 0-1
                att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)

                # 生成热力图（jet 配色）
                cmap = plt.get_cmap('jet')
                heatmap_rgba = cmap(att_map)  # [H, W, 4]（RGBA）
                heatmap_rgb = heatmap_rgba[..., :3]  # [H, W, 3]（RGB 0-1）
                heatmap = (heatmap_rgb * 255).astype(np.uint8)  # [H, W, 3]（0-255）

                # 叠加热力图（纯 Python 实现）
                alpha = 0.6
                overlay = (image * (1 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)

                # 绘图保存
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                plt.title('Original Image')
                plt.imshow(image)
                plt.axis('off')

                plt.subplot(132)
                plt.title('Attention Map')
                plt.imshow(att_map, cmap='jet')
                plt.axis('off')

                plt.subplot(133)
                plt.title('Overlay')
                plt.imshow(overlay)
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(
                    os.path.join(att_save_path, f'sample_{sample_idx}_head_{head_idx}_query_{query_idx}.png'),
                    bbox_inches='tight',
                    dpi=300
                )
                plt.close()

    # 释放内存
    del attention_maps, first_batch_images
    torch.cuda.empty_cache()

# ------------------ 新增：反归一化函数 ------------------
def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将归一化后的张量还原为原始图像（RGB 0-255）"""
    # 检查图像维度
    if image.dim() == 3:  # [C, H, W]
        image = image.permute(1, 2, 0).numpy()
    elif image.dim() == 4:  # [T, C, H, W] 或 [B, C, H, W]
        # 取第一帧或第一个样本
        image = image[0].permute(1, 2, 0).numpy()
    elif image.dim() == 5:  # [T, Cam, C, H, W]
        # 取第一帧的第一个视角
        image = image[0, 0].permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unsupported image dimension: {image.dim()}")

    # 反归一化并确保数值在0-1范围内
    image = (image * std + mean).clip(0, 1)
    return (image * 255).astype(np.uint8)