import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_confusion_matrix(confusion_matrix, class_names=None, save_path=None, title='Confusion Matrix'):
    """
    可视化混淆矩阵

    Args:
        confusion_matrix: 混淆矩阵数组
        class_names: 类别名称列表
        save_path: 保存路径（可选）
        title: 图表标题
    """
    # 创建图表
    plt.figure(figsize=(10, 8))

    # 如果没有提供类别名称，则使用默认编号
    if class_names is None:
        class_names = [f'Class {i}' for i in range(confusion_matrix.shape[0])]

    # 使用seaborn绘制热力图
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                cbar=True)

    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # 旋转标签以提高可读性
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 如果指定了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    plt.close()


def plot_normalized_confusion_matrix(confusion_matrix, class_names=None, save_path=None,
                                     title='Normalized Confusion Matrix'):
    """
    可视化归一化混淆矩阵（显示百分比）

    Args:
        confusion_matrix: 混淆矩阵数组
        class_names: 类别名称列表
        save_path: 保存路径（可选）
        title: 图表标题
    """
    # 归一化混淆矩阵（按行归一化）
    normalized_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # 创建图表
    plt.figure(figsize=(10, 8))

    # 如果没有提供类别名称，则使用默认编号
    if class_names is None:
        class_names = [f'Class {i}' for i in range(normalized_cm.shape[0])]

    # 使用seaborn绘制热力图（显示百分比）
    sns.heatmap(normalized_cm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                cbar=True,
                vmin=0,
                vmax=1)

    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # 旋转标签以提高可读性
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 如果指定了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Normalized confusion matrix saved to {save_path}")

    plt.show()
    plt.close()
