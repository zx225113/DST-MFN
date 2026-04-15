import torch

import os


def print_log(result_path, *args):  # 传入 save_path = result_path = './result/%s (args.dataset, time_str)
    os.makedirs(result_path, exist_ok=True)  # 创建目录路径，exist_ok=True表示目录已存在则不报错

    print(*args)  # 将传入的的参数输出到控制台
    file_path = result_path + '/log.txt'  # 将 result_path 和 /log.txt 组合成文件路径
    if file_path is not None:
        with open(file_path, 'a') as f:  # 以追加的模式（‘a’）打开日志文件，确保多次调用不会覆盖之前的内容 使用with语句自动管理文件资源，确保操作后正确关闭文件
            print(*args, file=f)  # 将*args参数内容同时输出到文件（通过file=f参数实现文件写入）
