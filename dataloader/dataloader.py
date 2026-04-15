from .volleyball import *
from .nba import *

TRAIN_SEQS_VOLLEY = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_SEQS_VOLLEY = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_SEQS_VOLLEY = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


def read_dataset(args):
    if args.dataset == 'volleyball':
        data_path = args.data_path + args.dataset
        image_path = data_path #+ "/videos"

        train_data = volleyball_read_annotations(image_path, TRAIN_SEQS_VOLLEY + VAL_SEQS_VOLLEY, args.num_activities)
        train_frames = volleyball_all_frames(train_data)

        test_data = volleyball_read_annotations(image_path, TEST_SEQS_VOLLEY, args.num_activities)
        test_frames = volleyball_all_frames(test_data)

        train_set = VolleyballDataset(train_frames, train_data, image_path, args, is_training=True)
        test_set = VolleyballDataset(test_frames, test_data, image_path, args, is_training=False)

    elif args.dataset == 'nba':
        data_path = args.data_path + 'NBA_dataset'  # 数据集的路径 = 基础路径 + “NBA_dataset”
        image_path = data_path + "/videos"  # 视频文件存储的路径

        train_id_path = data_path + "/train_video_ids"  # 训练集视频ID文件路径 = 数据集路径 + “train_video_ids”
        test_id_path = data_path + "/test_video_ids"    # 测试集视频ID文件路径 = 数据集路径 + “test_video_ids”

        train_ids = read_ids(train_id_path)  # 调用read_ids函数从指定路径读取训练集视频ID [21801058,,21801127,……,21800968]
        test_ids = read_ids(test_id_path)   # 调用read_ids函数从指定路径读取测试集视频ID  [21800974,21801076,……,21801129]

        train_data = nba_read_annotations(image_path, train_ids)    # 调用 nba_read_annotations 函数，读取并解析NBA数据集的训练集标注文件。
        # 返回值：{ 21801058 :15 : { 'file_name'：'15' ， 'group_activity'： 2 }，…… }
        train_frames = nba_all_frames(train_data)   # 调用 nba_all_frames 函数，从 train_data 中提取所有有效视频帧的序列信息。
        #  返回值：# [(21801058, 15),……]

        test_data = nba_read_annotations(image_path, test_ids)
        test_frames = nba_all_frames(test_data)

        train_set = NBADataset(train_frames, train_data, image_path, args, is_training=True)
        # train_set：变量名，用于存储创建的训练数据集对象
        # NBADataset：这是自定义的数据集类（在nba.py文件中定义），继承自PyTorch的data.Dataset
        # train_frames：参数1，包含所有训练帧的标识信息，格式为[(视频ID, 帧ID), ...]
        # train_data：参数2，包含所有训练数据的详细标注信息，是一个嵌套字典结构
        # image_path：参数3，图像文件的根路径，用于定位和加载图像文件
        # args：参数4，包含各种配置参数的对象（如图像尺寸、采样策略等）
        # is_training = True：参数5，布尔值，指示这是训练模式，会影响数据增强和采样策略
        test_set = NBADataset(test_frames, test_data, image_path, args, is_training=False)
        # is_training = False：参数5，布尔值，指示这是测试模式，会使用不同的数据处理策略（如不进行随机采样）
    else:
        assert False

    print("%d train samples and %d test_command samples" % (len(train_frames), len(test_frames)))

    return train_set, test_set
