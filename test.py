import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.distributions import Categorical

import os
import copy
import time
import random
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import models.models as models
from visualization.confusion_visualization import *
from visualization.t_SNE_visualization import *
from visualization.attention_maps_visualization import *
from util.utils import *
from dataloader.dataloader import read_dataset

parser = argparse.ArgumentParser(description='Detector-Free Weakly Supervised Group Activity Recognition')

# Dataset specification
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')
parser.add_argument('--data_path', default='./Dataset/', type=str, help='data path')
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=18, type=int, help='number of frames for each clip')
parser.add_argument('--num_total_frame', default=72, type=int, help='number of total frames for each clip')
parser.add_argument('--num_activities', default=6, type=int, help='number of activity classes in volleyball dataset')

# Model parameters
parser.add_argument('--base_model', action='store_true', help='average pooling base model')
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')
parser.add_argument('--dilation', action='store_true', help='use dilation or not')
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

# Motion parameters
parser.add_argument('--motion', action='store_true', help='use motion feature computation')
parser.add_argument('--multi_corr', action='store_true', help='motion correlation block at 4th and 5th')
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')

# Transformer parameters
parser.add_argument('--nheads', default=4, type=int, help='number of heads')
parser.add_argument('--enc_layers', default=6, type=int, help='number of encoder layers')
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_tokens', default=12, type=int, help='number of queries')

# Aggregation parameters
parser.add_argument('--nheads_agg', default=4, type=int, help='number of heads for partial context aggregation')

# Training parameters
parser.add_argument('--batch', default=4, type=int, help='Batch size')
parser.add_argument('--test_batch', default=4, type=int, help='Test batch size')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')

# GPU
parser.add_argument('--device', default="0, 1", type=str, help='GPU device')

# Load model
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

args = parser.parse_args()
best_mca = 0.0
best_mpca = 0.0
best_mca_epoch = 0
best_mpca_epoch = 0


def main():
    global args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_DFGAR_<%s>' % (args.dataset, time_str)
    save_path = './result/%s' % exp_name

    _, test_set = read_dataset(args)

    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=8, pin_memory=True)

    if args.base_model:
        model = models.BaseModel(args)
    else:
        model = models.DFGAR(args)
    model = torch.nn.DataParallel(model).cuda()

    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # define loss function and optimizer
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # validate(train_loader, model)
    acc, mean_acc, confusion, features, labels, attention_maps, first_batch_images = validate(test_loader, model)
    print('Accuracy is %.2f' % acc)
    print('Mean accuracy is %.2f' % mean_acc)
    print(confusion)

    # 添加混淆矩阵可视化
    # 定义类别名称（根据你的数据集进行修改）
    if args.dataset == 'volleyball':
        class_names = ['r_set', 'r_spike', 'r_pass', 'r_winpoint', 'l_set', 'l_spike', 'l_pass', 'l_winpoint']
    elif args.dataset == 'nba':
        class_names = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']
    else:
        class_names = None  # 使用默认编号

    # 创建保存目录
    vis_save_path = os.path.join(save_path, 'visualization')
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    # 可视化混淆矩阵
    plot_confusion_matrix(confusion, class_names=class_names,
                          save_path=os.path.join(vis_save_path, 'confusion_matrix.png'))

    # 可视化归一化混淆矩阵
    plot_normalized_confusion_matrix(confusion, class_names=class_names,
                                     save_path=os.path.join(vis_save_path, 'normalized_confusion_matrix.png'))

    # 如果有特征，则进行t-SNE可视化
    if features is not None:
        plot_tsne_visualization(features, labels, class_names=class_names,
                                save_path=os.path.join(vis_save_path, 'tsne_visualization.png'))

    # 注意力可视化
    if attention_maps is not None and first_batch_images is not None:
        visualize_attention_maps(attention_maps, first_batch_images, vis_save_path, args)
@torch.no_grad()
def validate(test_loader, model):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    accuracies = AverageMeter()
    true = []
    pred = []
    features_list = []  # 用于存储特征
    labels_list = []  # 用于存储标签
    attention_maps_list = []  # 新增：存储注意力图
    first_batch_images = None  # 保存第一批图像用于可视化
    # switch to eval mode
    model.eval()

    # 创建进度条对象
    test_loader_with_progress = tqdm(test_loader, leave=False, desc="Testing", ncols=100, unit="batch")

    for i, (images, activities) in enumerate(test_loader_with_progress):
        images = images.cuda()
        activities = activities.cuda()

        # 保存第一批图像用于可视化
        if first_batch_images is None:
            first_batch_images = images.cpu().clone()

        num_batch = images.shape[0]
        num_frame = images.shape[1]
        activities_in = activities[:, 0].reshape((num_batch,))

        # compute output
        # 检查模型是否有return_features参数支持
        try:
            score, features, att = model(images, return_features=True, return_attentions=True)
            features_list.append(features.cpu().numpy())
            labels_list.append(activities_in.cpu().numpy())
            attention_maps_list.append(att.detach().cpu())  # att 需从 model 前向中传递
        except TypeError:
            # 如果不支持return_features参数，则只获取预测结果
            score = model(images)

        true = true + activities_in.tolist()
        pred = pred + torch.argmax(score, dim=1).tolist()

        # measure accuracy and record loss
        group_acc = accuracy(score, activities_in)
        accuracies.update(group_acc, num_batch)

        # 更新进度条显示
        test_loader_with_progress.set_postfix({
            'Acc': f'{group_acc * 100:.2f}%',            # 当前批次准确率
            'mean_acc': f'{accuracies.avg * 100:.2f}%'   # 平均准确率
        })

    acc = accuracies.avg * 100.0
    confusion = confusion_matrix(true, pred)
    mean_acc = np.mean([confusion[i, i] / confusion[i, :].sum() for i in range(confusion.shape[0])]) * 100.0

    # 合并所有特征
    if features_list and attention_maps_list:
        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.array(true)
        all_attention_maps = torch.cat(attention_maps_list, dim=0)
        return acc, mean_acc, confusion, all_features, all_labels,all_attention_maps,first_batch_images

    return acc, mean_acc, confusion, None, None, None, None


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


def accuracy(output, target):
    output = torch.argmax(output, dim=1)
    correct = torch.sum(torch.eq(target.int(), output.int())).float()
    return correct.item() / output.shape[0]

if __name__ == '__main__':
    main()
