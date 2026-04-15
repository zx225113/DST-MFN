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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import models.models as models
from util.utils import *
from dataloader.dataloader import read_dataset

parser = argparse.ArgumentParser(description='Detector-Free Weakly Supervised Group Activity Recognition')  # 参数解释器对象

# Dataset specification 数据集配置
# parser.add_argument() 用来向命令行参数器添加一个参数
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')  # 数据集的名字 help是提示参数
parser.add_argument('--data_path', default='/home/data/', type=str, help='data path')  # 指定数据集的路径
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')  # 调整输入图面的宽度 1280
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')  # 调整输入图面的高度 720
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')  # 随机采样
parser.add_argument('--num_frame', default=18, type=int, help='number of frames for each clip')  # 每个片段采样的帧数 (采样后)
parser.add_argument('--num_total_frame', default=72, type=int, help='number of total frames for each clip')  # 原始视频片段的总帧数 (采样前)
parser.add_argument('--num_activities', default=6, type=int, help='number of activity classes in volleyball dataset')  # 指定数据集中不同活动类别的总数

# Model parameters
parser.add_argument('--base_model', action='store_true', help='average pooling base model')  # 带平均池化的简单模型
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')  # 特征提取主干网络
parser.add_argument('--dilation', action='store_true', help='use dilation or not')  # 空洞卷积
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')  # 设置 Transformer 模型隐藏层维度（通道数）

# Motion parameters
parser.add_argument('--motion', action='store_true', help='use motion feature computation')  # 运动特征开关 调用光流估计模块 视频动作识别 视频目标检测
parser.add_argument('--multi_corr', action='store_true', help='motion correlation block at 4th and 5th')  # 运动相关性计算模块
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')  # 指定在 backbone 的第几层计算运动相关性
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')  # 控制运动相关性计算前的特征投影维度，本质是降维操作
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')  # 定义运动相关性计算的局部搜索窗口大小，控制模型对运动范围的感知能力

# Transformer parameters
parser.add_argument('--nheads', default=4, type=int, help='number of heads')  # 设置多头自注意力的数量
parser.add_argument('--enc_layers', default=6, type=int, help='number of encoder layers')  # 设置模型的编码器层数
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')  # 前置归一化
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')  # 前馈神经网络的隐藏层维度
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')  # 选择位置编码的方式
parser.add_argument('--num_tokens', default=12, type=int, help='number of queries')  # 设置模型中查询的数量

# Aggregation parameters
parser.add_argument('--nheads_agg', default=4, type=int, help='number of heads for partial context aggregation')  # 控制模型局部上下文聚合操作中使用的注意力头数

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')  # 控制程序随机种子，确保实验的可重复性
parser.add_argument('--epochs', default=30, type=int, help='Max epochs')  # 控制训练轮次
parser.add_argument('--test_freq', default=2, type=int, help='print frequency')  # 控制测试的执行频率
parser.add_argument('--batch', default=4, type=int, help='Batch size')  # 控制输入批量大小的训练参数
parser.add_argument('--test_batch', default=4, type=int, help='Test batch size')  # 控制测试阶段批量大小的参数
parser.add_argument('--lr', default=1e-6, type=float, help='Initial learning rate')  # 初始学习率
parser.add_argument('--max_lr', default=1e-4, type=float, help='Max learning rate')  # 最大学习率上限，用于动态学习率调度策略
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')  # 学习率调度器的步长，控制学习率衰减节奏
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')  # 循环学习率调度器中的下降步长参数，专门用于控制学习率在周期性调整中的下降阶段持续时间。
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')  # 控制模型权重衰减强度，防止过拟合
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')  # Dropout率，优化
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')  # 梯度裁剪，防止梯度爆炸
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')  # 梯度裁剪的最大范数阈值，控制梯度裁剪的强度

# GPU
parser.add_argument('--device', default="0, 1", type=str, help='GPU device')  # 设备

# Load model
parser.add_argument('--load_model', action='store_true', help='load model')  # 模型加载开关参数，用于控制是否从保存的检查点恢复模型训练或推理
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')  # 指定预训练模型路径的命令行参数，是模型迁移学习和断点续训的核心配置

parser.add_argument('--use_optical_flow', action='store_true', help='use optical flow for high-order motion statistics')

args = parser.parse_args()
best_mca = 0.0  # 记录验证集上的最佳Mean Class Accuracy（平均类别准确率）
best_mpca = 0.0  # 记录验证集上的最佳Mean Pixel-wise Class Accuracy（平均像素级类别准确率）
best_mca_epoch = 0  # 达到best_mca时的训练epoch数
best_mpca_epoch = 0  # 达到best_mpca时的训练epoch数


def main():
    global args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 设置CUDA设备的排序方式 ，这里是按照PCI总线ID排序 默认是按性能排序
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device    # 设置程序可见的设别

    # 设置保存的时间、名字和地址
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_DFGAR_<%s>' % (args.dataset, time_str)
    save_path = './result/%s' % exp_name

    # set random seed
    random.seed(args.random_seed)   # 设置Python内置random模块的随机种子
    np.random.seed(args.random_seed)    # 设置NumPy的随机种子
    torch.manual_seed(args.random_seed)  # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed(args.random_seed)    # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(args.random_seed)    # 设置所有GPU的随机种子（多GPU时）
    torch.backends.cudnn.deterministic = True   # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False   # 关闭CuDNN的自动优化（避免随机性）

    train_set, test_set = read_dataset(args)  # 读取训练集和测试集的数据

    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
    #  加载训练数据集的                          控制每批的数据量：4        打乱数据顺序   启用8个进程加速数据加载  将数据锁到GPU内存以提升传输效率
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=8, pin_memory=True)

    # 选择模型
    if args.base_model:
        model = models.BaseModel(args)
    else:
        model = models.DFGAR(args)
    model = torch.nn.DataParallel(model).cuda()  # 将模型包装为数据并行模式，实现多GPU训练，自动将输入数据分割并分配到不同GPU上并行计算

    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')  # 调用print_log()函数,打印args
    print_log(save_path, parameters)  # 调用print_log()函数,打印args

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()    # 创建交叉熵损失函数，通过.cuda()将损失函数转移到GPU上计算，以加速训练过程
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)
    # 使用Adam算法优化模型参数  model.parameters()：要优化的模型参数  args.lr：初始学习率  betas=(0.9, 0.999)：动量参数，参数更新保持之前的方向趋势，自适应调节每个参数的学习率
    # eps=1e-8：数值稳定项，安全系数 weight_decay：L2正则化系数
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.max_lr, step_size_up=args.lr_step,
                                                  step_size_down=args.lr_step_down, mode='triangular2',
                                                  cycle_momentum=False)
    # 创建学习率调度器 optimizer: 优化的目标 args.lr: 初始学习率 args.max_lr: 最大学习率 step_size_up = args.lr_step: 学习率从最小值增长到最大值需要的步数（迭代次数）
    # step_size_down = args.lr_step_down: 学习率从最大值下降到最小值需要的步数
    # args.mode: triangular：标准三角循环（振幅恒定） triangular2（当前选择）：每个周期振幅减半，逐步收敛 exp_range：指数变化（需额外参数)

    # 加载预训练模型
    if args.load_model:
        checkpoint = torch.load(args.model_path)  # 加载检查点文件
        model.load_state_dict(checkpoint['state_dict'])  # 加载模型权重：将检查点中的模型参数加载到当前模型
        scheduler.load_state_dict(checkpoint['scheduler'])  # 恢复学习率调度器状态：确保学习率变化从上次中断的位置继续
        optimizer.load_state_dict(checkpoint['optimizer'])  # 恢复优化器状态：包括动量缓存等优化器内部变量
        start_epoch = checkpoint['epoch'] + 1   # 设置起始 epoch：从上次训练的 epoch+1 开始
    else:
        start_epoch = 1  # 默认从头开始训练：设置起始 epoch 为 1

    # training phase
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- %s at epoch #%d' % ("Train", epoch))
        train_log = train(train_loader, model, criterion, optimizer, epoch)  # 调用train函数完成一个epoch的训练
        # 返回包含以下信息的字典： group_acc：当前epoch的准确率 loss：平均损失值 time：训练耗时 epoch: 训练轮次
        print_log(save_path, 'Accuracy: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                  (train_log['group_acc'], train_log['loss'], train_log['time']))  # 打印返回的信息
        print('Current learning rate is %f' % scheduler.get_last_lr()[0])  # 打印当前学习率
        scheduler.step()  # 调用学习率调度器执行step更新

        if epoch % args.test_freq == 0:  # 训练轮次整除测试的执行频率时，就进行一次测试
            print_log(save_path, '----- %s at epoch #%d' % ("Test", epoch))  # 打印测试执行的轮次信息
            test_log = validate(test_loader, model, criterion, epoch)  # 调用validate函数进行一次测试
            print_log(save_path, 'Accuracy: %.2f%%, Mean-ACC: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                      (test_log['group_acc'], test_log['mean_acc'], test_log['loss'], test_log['time']))
            # 打印返回的信息： group_acc/test_acc：整体准确率  mean_acc：类别平均准确率 loss: 平均损失函数值 time:测试用时间
            print_log(save_path, '----------Best MCA: %.2f%% at epoch #%d.' %
                      (test_log['best_mca'], test_log['best_mca_epoch']))  # 打印最好的MCA对应的epoch
            print_log(save_path, '----------Best MPCA: %.2f%% at epoch #%d.' %
                      (test_log['best_mpca'], test_log['best_mpca_epoch']))  # 打印最好的MPCA对应的epoch

            if epoch == test_log['best_mca_epoch'] or epoch == test_log['best_mpca_epoch']:
                #  如果当前epoch是最好的MCA或MPCA对应的epoch，则保存模型
                state = {
                    'epoch': epoch,  # 当前epoch
                    'state_dict': model.state_dict(),   # 模型参数（权重）
                    'optimizer': optimizer.state_dict(),    # 优化器的状态信息（如动量、历史梯度等），便于恢复训练时继续使用
                    'scheduler': scheduler.state_dict(),    # 学习率调度器的状态信息，确保学习率调度也能够从上次中断的地方继续
                }
                result_path = save_path + '/epoch%d_%.2f%%.pth' % (epoch, test_log['group_acc'])  # 创建保存模型文件的路径和名称
                torch.save(state, result_path)  # 调用 PyTorch 的 torch.save() 方法，将 state 中的内容保存为 .pth 文件


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    epoch_timer = Timer()  # 创建计时器对象
    losses = AverageMeter()  # 创建损失值和准确率计算的对象
    accuracies = AverageMeter()  # 创建准确率计算对象

    # switch to train mode
    model.train()   # 将模型设置为训练模式

    # 创建进度条对象
    train_loader_with_progress = tqdm(train_loader, desc=f'Epoch {epoch} Train', leave=False)

    # 遍历训练数据加载器train_loader，每次获取一个批次的图像和对应的活动标签。i是批次索引，images是输入的视频帧数据，activities是对应的活动标签。
    for i, (images, activities) in enumerate(train_loader_with_progress):
        # 将数据从CPU转移到GPU，加速计算
        images = images.cuda()                                      # [B, T, 3, H, W]
        activities = activities.cuda()                              # [B, T]

        num_batch = images.shape[0]  # 获取批次大小B
        num_frame = images.shape[1]  # 获取当前批次中每个样本的帧数T

        activities_in = activities[:, 0].reshape((num_batch, ))
        # : 表示选择所有批次的样本。 0 表示选择每个样本的第一个帧的标签。 activities[:, 0] 的结果是一个形状为 [B] 的一维张量，表示每个样本取第一个帧的标签作为当前批次的训练目标。
        # reshape 是用来改变张量形状的操作 (num_batch, ) 表示将张量重塑为一个一维张量，长度为 num_batch（即 B）确保该张量的长度为 B。
        # 最终 activities_in 是一个形状为 [B] 的张量，表示每个样本的标签（这里取的是每个样本第一个帧的标签）。
        # 在视频分类任务中，有时每个视频片段包含多帧，但标签是整个片段的类别（即每帧的标签相同）。在这种情况下，只需要取任意一帧的标签作为整个片段的标签即可。

        # compute output
        score = model(images)                                       # 模型输出结果：[B, C]，其中 B 是批次大小，C 是类别数。

        # calculate loss
        loss = criterion(score, activities_in)  # 使用交叉熵损失函数criterion计算模型输出score与真实标签activities_in之间的损失值。

        # measure accuracy and record loss
        group_acc = accuracy(score, activities_in)  # 调用accuracy函数计算当前批次的准确率
        losses.update(loss, num_batch)  # 更新损失值的平均值
        accuracies.update(group_acc, num_batch)  # 更新准确率的平均值

        # compute gradient and do SGD step
        optimizer.zero_grad()  # 清除优化器中所有参数的梯度，以避免梯度累积。
        loss.backward()  # 执行反向传播，计算梯度
        if args.gradient_clipping:  # 判断是否启用梯度裁剪（防止梯度爆炸）
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)  # 对模型参数的梯度进行裁剪，使其范数不超过args.max_norm。
        optimizer.step()    # 更新模型参数，完成一次优化步骤

        # 更新进度条显示
        train_loader_with_progress.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg * 100:.2f}%'
        })

    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': accuracies.avg * 100.0,
    }

    return train_log


@torch.no_grad()    # 装饰器，表示该函数不进行梯度计算（节省内存和计算资源），即不进行反向传播
def validate(test_loader, model, criterion, epoch):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch  # 声明全局变量，用于更新最佳准确率记录
    epoch_timer = Timer()   # 创建计时器对象
    losses = AverageMeter()  # 创建损失值计算对象
    accuracies = AverageMeter()  # 创建准确率计算对象
    true = []  # 创建一个空列表，用于存储真实标签
    pred = []  # 创建一个空列表，用于存储预测标签

    # switch to eval mode
    model.eval()    # 将模型设置为评估模式

    # 添加进度条
    test_loader_with_progress = tqdm(test_loader, desc=f'Epoch {epoch} Test', leave=False)
    # 遍历训练数据加载器train_loader，每次获取一个批次的图像和对应的活动标签。i是批次索引，images是输入的视频帧数据，activities是对应的活动标签。
    for i, (images, activities) in enumerate(test_loader_with_progress):
        # 将数据从CPU转移到GPU，加速计算
        images = images.cuda()  # [B, T, 3, H, W]
        activities = activities.cuda()  # [B, T]

        num_batch = images.shape[0]  # 获取批次大小B
        num_frame = images.shape[1]  # 获取当前批次中每个样本的帧数T
        activities_in = activities[:, 0].reshape((num_batch,))  # [B]

        # compute output
        score = model(images)  # 模型输出结果：[B, C]，其中 B 是批次大小，C 是类别数。

        true = true + activities_in.tolist()  # 将真实标签添加到true列表中
        pred = pred + torch.argmax(score, dim=1).tolist()  # torch.argmax(score, dim=1)：取预测得分最高的类别索引
        # .tolist()：将预测标签添加到pred列表

        # calculate loss
        loss = criterion(score, activities_in)  # criterion：交叉熵损失函数，输入： score：模型输出 [B, C]；activities_in：真实标签 [B]。

        # measure accuracy and record loss
        group_acc = accuracy(score, activities_in)  # 计算当前批次的分类准确率
        losses.update(loss, num_batch)  # 更新损失值的平均值
        accuracies.update(group_acc, num_batch)  # 更新准确率的平均值

        # 更新进度条显示
        test_loader_with_progress.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg * 100:.2f}%'
        })

    acc = accuracies.avg * 100.0  # 总体准确率
    confusion = confusion_matrix(true, pred)  # 生成混淆矩阵（行：真实标签，列：预测标签）。
    mean_acc = np.mean([confusion[i, i] / confusion[i, :].sum() for i in range(confusion.shape[0])]) * 100.0  # 平均类别准确率

    # 更新最佳准确率记录
    if acc > best_mca:
        best_mca = acc
        best_mca_epoch = epoch
    if mean_acc > best_mpca:
        best_mpca = mean_acc
        best_mpca_epoch = epoch

    test_log = {
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': acc,
        'mean_acc': mean_acc,
        'best_mca': best_mca,
        'best_mpca': best_mpca,
        'best_mca_epoch': best_mca_epoch,
        'best_mpca_epoch': best_mpca_epoch,
    }

    return test_log


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()  # 初始化方法，将属性val、avg、sum和count都设置为0

    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 样本数量

    def update(self, val, n=1):
        self.val = val  # 记录当前值
        self.sum += val * n  # n 是样本数量，当不显式指定 n 时，默认 n=1，表示 val 是单个样本的值（或已按批次均值处理）。
        self.count += n  # 样本数量
        self.avg = self.sum / self.count    # 更新后的平均值


class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time = time.time()  # 初始化时记录当前时间戳

    def timeit(self):
        old_time = self.last_time  # 记录上一次调用 timeit() 方法的时间戳
        self.last_time = time.time()  # 记录当前时间戳
        return self.last_time - old_time  # 返回两次调用 timeit() 方法的时间间隔


def accuracy(output, target):
    output = torch.argmax(output, dim=1)  # 沿着 dim=1(类别维度) 轴上，返回 output 中每个元素对应的最大索引。
    correct = torch.sum(torch.eq(target.int(), output.int())).float()
    # torch.eq() 逐元素比较两个张量是否相等
    # .int()确保数据类型一致
    # torch.sum() 统计正确预测的总数
    # .float() 转换为浮点数以便后续除法运算
    return correct.item() / output.shape[0]  # .item() 作用是将单元素张量([5])转换为 Python 的基础数据类型（如 float[5.0]或 int[5]）


if __name__ == '__main__':
    main()
