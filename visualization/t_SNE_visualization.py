import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def plot_tsne_visualization(features, labels, class_names=None, save_path='tsne_visualization.png'):
    """
    对特征进行t-SNE降维并可视化

    Args:
        features: 提取的特征向量 (n_samples, n_features)
        labels: 真实标签 (n_samples,)
        class_names: 类别名称列表
        save_path: 保存路径
    """
    print("Performing t-SNE visualization...")
    print(f"Original features shape: {features.shape}")
    print(f"Original labels shape: {labels.shape}")

    # 确保features和labels长度一致
    if len(features) != len(labels):
        print(f"Warning: features length ({len(features)}) != labels length ({len(labels)})")
        # 取较小的长度，确保两者一致
        min_length = min(len(features), len(labels))
        features = features[:min_length]
        labels = labels[:min_length]
        print(f"Truncated to {min_length} samples for consistency")

    print(f"Final features shape: {features.shape}")
    print(f"Final labels shape: {labels.shape}")

    # 执行t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)

    # 创建可视化图表
    plt.figure(figsize=(10, 8))

    # 再次确保长度一致（以防万一）
    if len(features_2d) != len(labels):
        print(f"Warning: features_2d length ({len(features_2d)}) != labels length ({len(labels)})")
        min_length = min(len(features_2d), len(labels))
        features_2d = features_2d[:min_length]
        labels = labels[:min_length]

    # 如果有类别名称，则使用它们，否则使用数字标签
    if class_names is not None:
        unique_labels = sorted(list(set(labels)))
        colors = plt.cm.get_cmap('Set1', len(unique_labels))
        # 创建标签值到索引的映射
        label_to_index = {label_val: idx for idx, label_val in enumerate(unique_labels)}

        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)[0]
            # 过滤掉超出范围的索引
            valid_indices = indices[indices < len(features_2d)]

            if len(valid_indices) > 0:  # 只有当存在有效索引时才绘制
                # 使用映射获取正确的索引
                class_index = label_to_index[label]
                if class_index < len(class_names):
                    plt.scatter(features_2d[valid_indices, 0], features_2d[valid_indices, 1],
                                c=[colors(i)], label=class_names[class_index], alpha=0.7, s=50)
                else:
                    plt.scatter(features_2d[valid_indices, 0], features_2d[valid_indices, 1],
                                c=[colors(i)], label=f'Class {label}', alpha=0.7, s=50)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=labels, cmap='Set1', alpha=0.7, s=50)
        plt.colorbar(scatter)

    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"t-SNE visualization saved to {save_path}")
