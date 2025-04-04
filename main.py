import argparse
import os

from torch import gru

from C3D import C3DModel
from LSTM import LSTMModel
from convnext import ConvNeXtV2Temporal
from gru import GRUModel
from informer import InformerModel
from interimage import InternImageTemporal
from real_image import load_labels_for_feature_prediction
from segformer import SegFormerTemporal
from snow import get_tensors_from_images
from tcnn import TCN
from timesformer import TimeSformer
from unet import UNetTransformer
from vit1 import ViTModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from train import CustomDataset, val, train_model1, train_model2, TimeSeriesDataset, validate_model
from model2 import MyRestormerModel, PredictModule
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from uu.loss import Fusionloss
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

GPU_number = os.environ['CUDA_VISIBLE_DEVICES']



def main(params, model, loss_train_mean=None, criterion=None, epoch=None, generate_img=None, real_snow_images=None,):
    path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/'  # 返回seg的绝对路径

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate used for train')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    # parser.add_argument('--num_classes', type=int, default=3, help='num of object classes (with void)')
    parser.add_argument('--model_name', type=str, default=None, help='path to model')
    parser.add_argument('--base_dir', type=str, default=path, help='project directory')
    parser.add_argument('--train_dir', type=str, default=path, help='train directory')
    parser.add_argument('--val_dir', type=str, default=path, help='val directory')
    parser.add_argument('--mode', type=str, default='train', help='train/demo')
    parser.add_argument('--data_format', type=str, default='voc', help='voc/coco/snow')
    parser.add_argument('--data_name', type=str, default=None, help='data name')
    parser.add_argument('--imgsz', nargs='+', type=int, default=64, help='inference size h,w')
    parser.add_argument('--output_path', type=str, default=None, help='save demo image directory')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--weights', type=str, default=None, help='weights path')
    parser.add_argument('--timestamp_csv', type=str, default=None, help='Path to the timestamp csv file')
    parser.add_argument('--generate_img_freq', type=int, default=10)  # 添加图像生成频率
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device to use for training')
    args = parser.parse_args(params)
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('-cuda', type=str, default='0', help='choose GPU ID')
    args_ = parser_.parse_args()

    if args_.cuda != None:
        try:
            gpu_id = [int(i) for i in args_.cuda.split(',')]
        except:
            raise Exception(f'"{args_.cuda}"中分隔符没有用","')
        os.environ['CUDA_VISIBLE_DEVICES'] = args_.cuda

    assert args.model_name is not None, 'Please name model.'
    assert args.mode in ['train', 'val'], 'Please choose between train and demo.'

    #创建cddfuse的数据集和数据加载器
    path_dict1 = {}
    path_dict1['train_path'] = args.train_dir if args.train_dir else args.base_dir + f'data/train/'  # 训练集图片存放地址
    path_dict1['val_path'] = args.val_dir if args.val_dir else args.base_dir + f'data/val/'  # 验证集图片存放地址

    model1 = MyRestormerModel().to(args.device)
    model1 = model1.cuda()
    #model2 = LSTMModel(input_dim=262144, hidden_dim=64, num_layers=3, fusion_dim=512)
    #model2 = PredictModule(input_dim=64, hidden_dim=[256, 256, 256], kernel_size=[(3, 3), (3, 3), (3, 3)], num_layers=3)
    #model2 = TCN(input_size=64,output_channels=1, num_channels=[64, 128, 128], kernel_size=3, dropout=0.2)
    #model2 = TimeSformer(img_size=64, patch_size=16, num_frames=3, num_classes=1, dim=128,depth=6, heads=8, mlp_dim=256, channels=3, dropout=0.1)
    model2 = ConvNeXtV2Temporal(num_frames=160, channels=64)
    #model2 = C3DModel(input_channels=64)
    #model2 = SegFormerTemporal(num_frames=3, channels=64, num_classes=1)
    #model2 = InformerModel(input_dim=64, output_dim=64, seq_length=3)
    #model2 = GRUModel(hidden_dim=64, input_dim=64, output_dim=64, num_layers=3, seq_length=3)
    #model2 = UNetTransformer()
    #model2 = ViTModel(input_dim=64)
    real_snow_images = load_labels_for_feature_prediction(base_dir=r"C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\lbl", num_folders=1, num_images_per_folder=390)

    file_path = r'F:\1snow\MMIF-CDDFuse-main\time\time\concatenated_features.pt'

    # 加载保存的特征
    fused_features = torch.load(file_path)
    output1 = fused_features
    # 打印加载的数据的形状和类型以确认加载成功
    #print(f"加载的特征形状: {fused_features.shape}")
    #print(f"加载的特征类型: {type(fused_features)}")

    #print(output1.shape)
    #label_path = r"C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train\lbl_patches.h5"

    dataset_val2 = TimeSeriesDataset(output1, real_snow_images, time_steps=10, label_steps=1)
    dataloader_val2 = DataLoader(dataset_val2, batch_size=10, shuffle=False, num_workers=0)

    if not os.path.exists(os.path.join(args.base_dir, 'checkpoints', args.data_name, args.model_name)):
        os.makedirs(os.path.join(args.base_dir, 'checkpoints', args.data_name, args.model_name))
    path_dict1['ckpt_path'] = os.path.join(args.base_dir, 'checkpoints', args.data_name, args.model_name)

    # 使用 generate_img 进行后续操作，例如训练模型2
    if output1 is not None:
        print("Using generate_img to train model2")
        #print(output1.shape)

    # 创建数据集
    dataset = TimeSeriesDataset(output1, real_snow_images, time_steps=10, label_steps=1)
    print(f"Dataset length: {len(dataset)}")  # 验证数据集长度
    dataloader_second_model = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

    if args.mode == 'train':
        train_model2(args, model2, dataloader_second_model, dataloader_val2)

    #elif args.mode == 'val':
       # val(args, model, dataloader_val1, label_tensors, epoch, loss_train_mean, criterion, path_dict1)
    #    val(args, model, label_tensors, epoch, loss_train_mean, criterion, path_dict1)
    else:
        raise ValueError("Invalid mode. Please specify 'train' or 'val'.")


    #return args, model1, model2, dataloader_train, path_dict1,output1
    return args, model1, model2, path_dict1, output1

root_dir = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\val\lbl' # 替换为你的文件夹路径

white_threshold = 150
label_tensors = get_tensors_from_images(root_dir)


if __name__ == '__main__':
    try:
        print('==========Main==========')
        args, model1, model2,  dataloader_val, path_dict1, start_epoch, epoch_gap = main([])  # 如果没有命令行参数，可以传递空列表

    except Exception as e:
        print(f"An error occurred: {e}")


    # 创建一个包含所有模型的字典或其他结构
    params_1 = [
        '--learning_rate', '1e-5',
        '--num_workers', '0',
        '--batch_size', '1',
        # '--data_format', 'coco',
        '--imgsz', '128',  # 推理时的图像大小
        '--data_name', 'data',  # 数据集的名称
        '--train_dir', r"C:\Users\ZYX\Desktop\CDDFuse-main\code\data\train",  # 训练集的路径

        '--val_dir', r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\val',  # 验证集的路径
        '--base_dir', r'C:\Users\ZYX\Desktop\CDDFuse-main\code',  # 基础目录的路径

        '--mode', 'train',  # 模式选择为训练模式
        '--model_name', 'MyCombinedModel'  # 模型的名称
    ]

    main(params_1, MyRestormerModel)
