import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image

ACTIVITIES = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']


def read_ids(path):  # path = ./dataset("nba","volleyball")/train_video_ids
    file = open(path)   # 默认以只读的方式打开对应路径的文件
    values = file.readline()  # 读取文件中的第一行数据（包括换行符）
    values = values.split(',')[:-1]  # split(',')：按逗号分割字符  [:-1]：去掉列表中的最后一个元素（通常空字符串或换行符）
    values = list(map(int, values))  # map(int, values)：将字符串values中的元素列表转换为整数迭代器 list(): 将迭代器转换为列表

    return values   # 返回一个values列表


def nba_read_annotations(path, seqs):
    labels = {}
    # 将分组名称（或活动类别）映射为数字ID
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:  # path/21801058/annotations.txt
            for line in f.readlines():  # 15\t2p-fail.-def.\n
                values = line[:-1].split('\t')  # ['15','2p-fail.-def.']
                file_name = values[0]  # '15'
                fid = int(file_name.split('.')[0])  # 不会进行分割 15

                activity = group_to_id[values[1]]   # 将分组名称映射为数字ID 2

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }   # { 15 : { 'file_name'：'15' ， 'group_activity'： 2 }, …… }
            labels[sid] = annotations  # { 21801058 : { 15 : { 'file_name'：'15' ， 'group_activity'：2 }, …… } }

    return labels  # 返回一个三重字典labels


def nba_all_frames(labels):
    frames = []

    for sid, anns in labels.items():  # sid: 21801058 , anns: { 15 : { 'file_name:'15',group_activity: 2 }, …… }
        for fid, ann in anns.items():  # { 'file_name:'15',group_activity: 2 } fid:15
            frames.append((sid, fid))  # [(21801058, 15),……]

    return frames


class NBADataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(NBADataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.transform = transforms.Compose([  # 开始定义图像预处理流水线，使用transforms.Compose组合多个变换操作。
            transforms.Resize((args.image_height, args.image_width)),  # 调整图像尺寸变换：将图像调整为指定的高度和宽度。
            transforms.ToTensor(),  # 转换变换：将PIL图像或numpy数组转换为PyTorch张量，并将像素值从[0,255]归一化到[0,1]。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 这是PyTorch中的图像标准化变换，用于将图像数据转换为标准正态分布形式。
            # mean=[0.485, 0.456, 0.406] 这是ImageNet数据集上计算得到的RGB三个通道的均值 R通道均值：0.485 G通道均值：0.456 B通道均值：0.406
            # std=[0.229, 0.224, 0.225] 这是ImageNet数据集上计算得到的RGB三个通道的标准差 R通道标准差：0.229 G通道标准差：0.224 B通道标准差：0.225
            # 对每个像素进行如下计算：normalized_pixel = (original_pixel - mean) / std
            # 作用：预训练模型兼容性：这些值来自ImageNet数据集，如果使用在ImageNet上预训练的模型（如ResNet），需要使用相同的标准化参数
            # 加速收敛：标准化后的数据更接近标准正态分布，有助于神经网络更快收敛
            # 数值稳定性：避免梯度消失或爆炸问题
        ])

    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])  # 根据索引idx获取对应的帧数据，并通过select_frames方法进行帧选择处理
        samples = self.load_samples(frames)  # 将处理后的帧数据加载为样本

        return samples

    def __len__(self):
        return len(self.frames)  # 返回self.frames列表的长度

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        vid, sid = frame  # vid: 21801058 ， …… , sid: 15 ， ……

        if self.is_training:  # 判断当前是否处于训练模式
            if self.random_sampling:  # 判断是否进行随机采样
                sample_frames = random.sample(range(72), self.num_frame)
                # 从0到71（共72帧）中随机选择 self.num_frame 个帧。random.sample 确保选择的帧不重复。
                sample_frames.sort()  # 对选中的帧按升序排序，确保帧的顺序是连续的。
            else:
                segment_duration = self.num_total_frame // self.num_frame  # 计算每段的持续时间（即每个采样段包含多少帧）。72/18=4.0
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
                # list(range(self.num_frame))：生成一个从0到self.num_frame-1的连续整数列表
                # np.multiply(list(range(self.num_frame)), segment_duration)：将上述列表中的每个元素都乘以segment_duration，表示每个采样段的起始位
                # np.random.randint(segment_duration, size=self.num_frame)：生成一个包含self.num_frame个随机整数的列表，每个整数都在[0, segment_duration)。表示在每个段内的随机偏移量
        else:
            if self.num_frame == 6:
                # [6, 18, 30, 42, 54, 66]
                sample_frames = list(range(6, 72, 12))  # 如果需要选择6帧，则按固定间隔选择帧：从第6帧开始，每隔12帧选一帧。
            elif self.num_frame == 12:
                # [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]
                sample_frames = list(range(4, 72, 6))  # 如果需要选择12帧，则按固定间隔选择帧：从第4帧开始，每隔6帧选一帧
            elif self.num_frame == 18:
                # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
                sample_frames = list(range(2, 72, 4))  # 如果需要选择18帧，则按固定间隔选择帧：从第2帧开始，每隔4帧选一帧
            else:
                segment_duration = self.num_total_frame // self.num_frame  # 计算每段的持续时间（即每个采样段包含多少帧）。72/18=4.0
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + segment_duration // 2
                # 从视频中等间隔采样帧的策略
        return [(vid, sid, fid) for fid in sample_frames]  # 返回一个包含所选帧信息的列表。每个元素是一个三元组 (vid, sid, fid)，其中 fid 是选定的帧ID。

    def load_samples(self, frames):
        images, activities = [], []  # 初始化两个空列表：images 用于存储加载的图像，activities 用于存储对应的活动标签

        for i, (vid, sid, fid) in enumerate(frames):  # 遍历 frames 列表中的每个元素。enumerate 提供索引 i 和对应的三元组 (vid, sid, fid)。
            fid = '{0:06d}'.format(fid)  # 将帧ID转换为6位数字字符串，不足6位前面补0  “000015”
            img = Image.open(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))  # 使用 PIL.Image.open 打开指定路径的图像文件。
            img = self.transform(img)  # 对加载的图像应用预定义的变换操作（调整大小、转换为张量、标准化等）

            images.append(img)  # 将处理后的图像添加到 images 列表中
            activities.append(self.anns[vid][sid]['group_activity'])  # 从注释字典中获取对应的群体活动标签，并将其添加到 activities 列表中。

        images = torch.stack(images)  # 使用 torch.stack 将图像列表转换为一个张量。假设有 n 张图像，每张图像的形状为 (C, H, W)，则最终的张量形状为 (n, C, H, W)。
        activities = np.array(activities, dtype=np.int32)  # 将活动标签列表转换为 NumPy 数组，数据类型为 32 位整数。

        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()  # 将 NumPy 数组转换为 PyTorch 张量，并将其数据类型设置为长整型（long）。

        return images, activities
