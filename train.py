import h5py
import numpy as np
import torch.fft as fft
from matplotlib.colors import ListedColormap
from torchmetrics.functional import structural_similarity_index_measure as ssim
import os
import pandas as pd
from C3D import C3DModel
from LSTM import LSTMModel
from convnext import ConvNeXtV2Temporal
from gru import GRUModel
from informer import InformerModel
from interimage import InternImageTemporal
from segformer import SegFormerTemporal
from tcnn import TCN
from timesformer import TimeSformer
from unet import UNetTransformer
from vit1 import ViTModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from net2 import ConvLSTM
from snow import get_tensors_from_images
from uu.standard import SegmentationMetric
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import kornia
import os
import torch.nn as nn
from PIL import Image
from uu.loss import Fusionloss, cc
from model.net import Restormer_opt_Encoder, Restormer_opt_Decoder, Restormer_sar_Encoder, Restormer_sar_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from dataprocessing import read_images_in_batches
from model2 import MyRestormerModel, PredictModule
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F

torch.cuda.empty_cache()

class CustomDataset(Dataset):
    def __init__(self, data_dir, image_size, mode,):
        self.data_dir = data_dir
        self.image_size = image_size
        self.opt_data = []
        self.sar_data = []
        self.lbl_data = []
        self.mode = mode
       # 定义光学和sar图像的目录路径
        opt_dir = os.path.join(data_dir, 'opt')
        sar_dir = os.path.join(data_dir, 'sar')
        lbl_dir = os.path.join(data_dir, 'lbl')
       # 获取光学和sar图像的文件名列表
        #opt_files = os.listdir(opt_dir)
        #sar_files = os.listdir(sar_dir)
        #lbl_files = os.listdir(lbl_dir)
        self.regions = os.listdir(opt_dir)  # 假设每个文件夹代表一个区域
        results = read_images_in_batches(opt_dir, sar_dir, lbl_dir)

        opt_images, sar_images, lbl_images = results
        transform_opt = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.47347586, 0.47310773, 0.47944347, 0.48842097],
                                 std=[0.27851925, 0.27133262, 0.26311616, 0.25469215]),
        ])

        transform_sar = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3328936012585958],
                                 std=[0.15350237081855972]),
        ])
        self.opt_data = []
        self.sar_data = []
        self.lbl_data = []

        for i in range(len(opt_images)-1):
            opt_path = opt_images[i]
            sar_path = sar_images[i]
            lbl_path = lbl_images[i+1]

            opt_image = Image.open(opt_path).convert('RGBA')  # 保持四通道
            sar_image = Image.open(sar_path).convert('L')  # 转换为灰度图像
            lbl_image = Image.open(lbl_path)     #读取标签图像

            transformed_opt_image = transform_opt(opt_image)
            transformed_sar_image = transform_sar(sar_image)
            transformed_lbl_image = transforms.ToTensor()(lbl_image.convert('L'))

            self.opt_data.append(transformed_opt_image)
            self.sar_data.append(transformed_sar_image)
            self.lbl_data.append(transformed_lbl_image)
            print(f"Loaded opt image: {opt_path}")
            print(f"Loaded sar image: {sar_path}")
            print(f"Loaded lbl image: {lbl_path}")
        print(f"Loaded {len(self.opt_data)} data points.")


    def __len__(self):
        # 确保所有数据列表长度一致
        if len(self.opt_data) != len(self.sar_data) or len(self.opt_data) != len(self.lbl_data):
            raise ValueError("The lengths of opt_data, sar_data, and lbl_data must be the same.")
       # return len(self.opt_data)

        return len(self.regions)

    def __getitem__(self, idx):
        if idx >= len(self.lbl_data):
            raise IndexError(f"Index {idx} is out of range for lbl_data with length {len(self.lbl_data)}")
        opt_image = self.opt_data[idx]
        sar_image = self.sar_data[idx]
        label = self.lbl_data[idx]

        return opt_image, sar_image, label


clip_grad_norm_value = 0.5
coeff_mse_loss_VF = 0.05  # alpha1
coeff_mse_loss_IF = 0.05
coeff_decomp = 1e-6  # alpha2 and alpha4
coeff_tv = 10
alpha = 100
step = 1
prev_time = time.time()

lr = 1e-4
weight_decay = 0
optim_step = 20
optim_gamma = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型并移动到设备上
DIDF_opt_Encoder = Restormer_opt_Encoder().to(device)
DIDF_opt_Decoder = Restormer_opt_Decoder().to(device)
DIDF_sar_Encoder = Restormer_sar_Encoder().to(device)
DIDF_sar_Decoder = Restormer_sar_Decoder().to(device)
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=16).to(device)
DetailFuseLayer = DetailFeatureExtraction(num_layers=2).to(device)  # 增加 num_layers

def train_model1(args, model1, train_loader1, val_loader1, path_dict1, self=None, model_optim1=None, setting1=None, num_epochs=5):
    all_fused_features = []
    # 定义优化器
    optimizers = {
        "opt_encoder": torch.optim.Adam(DIDF_opt_Encoder.parameters(), lr=lr, weight_decay=weight_decay),
        "opt_decoder": torch.optim.Adam(DIDF_opt_Decoder.parameters(), lr=lr, weight_decay=weight_decay),
        "sar_encoder": torch.optim.Adam(DIDF_sar_Encoder.parameters(), lr=lr, weight_decay=weight_decay),
        "sar_decoder": torch.optim.Adam(DIDF_sar_Decoder.parameters(), lr=lr, weight_decay=weight_decay),
        "base_fuse": torch.optim.Adam(BaseFuseLayer.parameters(), lr=1e-4, weight_decay=weight_decay),
        "detail_fuse": torch.optim.Adam(DetailFuseLayer.parameters(), lr=1e-4, weight_decay=weight_decay),
    }

    schedulers = {
        key: ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)
        for key, opt in optimizers.items()
    }

    criteria_fusion = Fusionloss()
    device = next(model1.parameters()).device

    model1 = MyRestormerModel()
    model1 = model1.to(device)  # 将模型移动到 GPU

   # model2 = PredictModule(input_dim=64, hidden_dim=[256, 256, 256], kernel_size=[(3, 3), (3, 3), (3, 3)], num_layers=3)
   # model2 = TCN(input_size=64, output_channels=1, num_channels=[64, 128, 128], kernel_size=3, dropout=0.2)
    #model2 = TimeSformer(img_size=64, patch_size=16, num_frames=3, num_classes=1, dim=128, depth=6, heads=8, mlp_dim=256, channels=3, dropout=0.1)
    #model2 = LSTMModel(input_dim=262144, hidden_dim=64, num_layers=3, fusion_dim=512)
    #model2 = TCN(input_size=64, output_channels=1, num_channels=[64, 128, 128], kernel_size=3, dropout=0.2)
    #model2 = TimeSformer(img_size=64, patch_size=16, num_frames=3, num_classes=1, dim=128, depth=6, heads=8,
     #                    mlp_dim=256, channels=3, dropout=0.1)
    model2 = ConvNeXtV2Temporal(num_frames=160, channels=64)
    #model2 = SegFormerTemporal(num_frames=3, channels=64, num_classes=1)
    #model2 = InformerModel(input_dim=64, output_dim=1, seq_length=3)
    #model2 = GRUModel(hidden_dim=64, input_dim=64, output_dim=64, num_layers=3, seq_length=3)
    #model2 = UNetTransformer()
    #model2 = ViTModel(input_dim=64, output_dim=64)
    #model2 = C3DModel(input_channels=64)
    model2 = model2.to(device)

    # 训练模型
    for epoch in range(num_epochs):
        model1.train()
        epoch_loss = 0
        print(f"Entering Epoch {epoch + 1}/{args.num_epochs}")
        for i, (opt, sar, label) in enumerate(train_loader1):
            region_idx = i % len(train_loader1.dataset.regions)  # 获取当前区域的索引
            region_data = (opt, sar, label)  # 获取当前批次的数据

            opt, sar, label = opt.to(device), sar.to(device), label.to(device)
            print("entering phase1")
            len_train = len(train_loader1)
            for opt_name, optimizer in optimizers.items():
                optimizer.zero_grad()
            loss_record2 = []
            epoch_loss = 0  # 当前epoch的总损失

            feature_opt_B, feature_opt_D, feature_opt = model1.opt_encoder(opt)  # 提取特征
            feature_sar_B, feature_sar_D, feature_sar = model1.sar_encoder(sar)

            opt_base_freq = fft.fft2(feature_opt_B)
            sar_base_freq = fft.fft2(feature_sar_B)
            opt_detail_freq = fft.fft2(feature_opt_D)  # 光学细节特征频域变换
            sar_detail_freq = fft.fft2(feature_sar_D)
            # 确保频域特征为实数类型（使用实部或模长）
            opt_base_freq_real = opt_base_freq.real  # 仅取实部
            sar_base_freq_real = sar_base_freq.real  # 仅取实部

            opt_base_freq_real = (opt_base_freq_real - opt_base_freq_real.mean()) / (opt_base_freq_real.std() + 1e-6)
            sar_base_freq_real = (sar_base_freq_real - sar_base_freq_real.mean()) / (sar_base_freq_real.std() + 1e-6)
            opt_detail_freq_real = opt_detail_freq.real  # 仅取实部
            sar_detail_freq_real = sar_detail_freq.real  # 仅取实部
            # 融合频域特征
            fused_base_freq = model1.fusion_layer_base(opt_base_freq_real, sar_base_freq_real)  # 在频域内融合
            fused_detail_freq = model1.fusion_layer_detail(opt_detail_freq_real, sar_detail_freq_real)  # 在频域内融合

            # 频域转换回时域
            fused_base_features = torch.real(fft.ifft2(fused_base_freq))  # 频域特征转换回时域
            fused_detail_features = torch.real(fft.ifft2(fused_detail_freq))  # 频域特征转换回时域

            # 解码融合后的特征
            data_Fuse, fused_img_features = model1.opt_decoder(opt, fused_base_features, fused_detail_features)
            # 将当前区域的结果保存到列表中
            all_fused_features.append(data_Fuse)  # 或者选择合适的特征保存

            cc_loss_B = cc(feature_opt_B, feature_sar_B)
            cc_loss_D = cc(feature_opt_D, feature_sar_D)
            loss_decomp = (cc_loss_D) ** 2 / (alpha + cc_loss_B + 1e-8)
            fusionloss, _, _ = criteria_fusion(opt, sar, fused_img_features)

            fusionloss = 0.1 * fusionloss
            l2_reg = torch.norm(fused_img_features, 2)

            fusionloss += 1e-5 * l2_reg
            fusionloss_weight = 0.1
            loss = fusionloss_weight * fusionloss + coeff_decomp * loss_decomp + 1e-5 * l2_reg


            # 反向传播和梯度裁剪
            loss.backward()
            for opt_name, optimizer in optimizers.items():
                nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=clip_grad_norm_value)

            # 参数更新
            for opt_name, optimizer in optimizers.items():
                optimizer.step()

            epoch_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len_train}], Loss: {loss.item()}')

        # 调度器步进
        for sched_name, scheduler in schedulers.items():
            scheduler.step(epoch_loss)

    file_path = r'C:\Users\ZYX\Desktop\DDcGAN-master\concatenated_features.pt'

    # 加载保存的特征
    fused_features = torch.load(file_path)
    # 扩展维度，模拟多个时间步（例如，time_steps=5）
    time_steps = 30  # 假设有5个时间步

    output1 = fused_features.unsqueeze(1).repeat(1, time_steps, 1, 1, 1)  # [3, 30, 64, 64, 64]

    # 打印形状检查
    #print(f"输入数据的形状: {output1.shape}")
    return output1

# 定义第二个模型的 Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, output1, real_snow_images, time_steps=10, label_steps=1):
        self.output1 = output1  # 形状: [13, 64, 64, 64]（time_steps, height, width, channels）
        self.real_snow_images = real_snow_images  # 形状: [1, 13, 1, 64, 64]
        self.time_steps = time_steps  # 输入的时间步数
        self.label_steps = label_steps  # 输出的时间步数（未来的标签）

        # 确保时间维度足够长
        self.total_time_steps = self.output1.shape[0]  # 13，时间步数
        if self.total_time_steps < self.time_steps + self.label_steps:
            raise ValueError(f"Total time steps ({self.total_time_steps}) is less than required "
                             f"time_steps ({self.time_steps}) + label_steps ({self.label_steps}).")

        # 计算可用的滑动窗口数量
        self.samples_per_region = self.total_time_steps - self.time_steps - self.label_steps + 1

    def __len__(self):
        return self.samples_per_region

    def __getitem__(self, index):
        # 当前滑动窗口的时间步起始索引
        time_idx = index  # 使用索引作为时间步的起始

        # 获取输入数据（连续的 time_steps 个时间步的特征）
        input_data = self.output1[time_idx:time_idx + self.time_steps, :, :, :]  # 取出连续的 time_steps 个时间步的特征
        #print(f"Input data shape (before squeeze): {input_data.shape}")
        #real_images = self.real_snow_images.squeeze(0)
        real_images = torch.tensor(self.real_snow_images)
        real_images = real_images.permute(1, 0, 2, 3, 4)

        # 获取标签数据（标签数据从 `time_idx + self.time_steps` 开始）
        label_data = real_images[time_idx + self.time_steps: time_idx + self.time_steps + self.label_steps, 0, :, :, :]

        #print(f"Label data shape: {label_data.shape}")

        return input_data, label_data



def train_model2(args, model, train_loader, val_loader):
    model = model.to(args.device)

    #optimizer = torch.optim.Adam([
    #    {'params': model.convlstm.parameters(), 'lr': 0.001},  # ConvLSTM 层
    #    {'params': model.conv_decoder.parameters(), 'lr': 0.01}  # 解码器层
   # ], weight_decay=1e-4)

    #optimizer = torch.optim.Adam([
    #    {'params': model.cnn3d.parameters(), 'lr': 0.001},  # ConvLSTM 层
    #    {'params': model.conv_decoder.parameters(), 'lr': 0.01}  # 解码器层
    #], weight_decay=1e-4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    epoch_rmse = []
    epoch_mae = []
    epoch_max_error = []
    train_losses, val_losses = [], []

    def combined_loss(output, label, delta=0.5):
        label = label.squeeze(1)  # 变为 [batch_size, 1, 64, 64]

        # 计算单步 Huber Loss
        loss = F.smooth_l1_loss(output, label, beta=delta)
        return loss

    criterion = combined_loss
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        for input_seq, label in train_loader:
            input_seq = input_seq.float().to(args.device)
            input_seq = input_seq * 1000

            label = label.float().to(args.device)
            optimizer.zero_grad()
            output = model(input_seq)
            mse_loss= combined_loss(output, label)
            loss = mse_loss

            loss.backward()  # 反向传播

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            running_loss += loss.item()
            # 记录预测值和真实值
            all_targets.extend(label.cpu().detach().numpy().flatten())
            all_predictions.extend(output.cpu().detach().numpy().flatten())

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        # 计算 RMSE, MAE, Max Error
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        mae = np.mean(np.abs(all_predictions - all_targets))
        max_error_value = np.percentile(np.abs(all_predictions - all_targets), 95)

        epoch_rmse.append(rmse)
        epoch_mae.append(mae)
        epoch_max_error.append(max_error_value)

        # **确保所有数组长度一致**
        max_len = max(len(epoch_rmse), len(epoch_mae), len(epoch_max_error))
        #epoch_rmse += [np.nan] * (max_len - len(epoch_rmse))
        #epoch_mae += [np.nan] * (max_len - len(epoch_mae))
        #epoch_max_error += [np.nan] * (max_len - len(epoch_max_error))

        print(f"Length Check - RMSE: {len(epoch_rmse)}, MAE: {len(epoch_mae)}, Max Error: {len(epoch_max_error)}")

        # 创建 DataFrame 并保存
        df = pd.DataFrame({
            "Epoch": list(range(1, max_len + 1)),
            "RMSE": epoch_rmse,
            "MAE": epoch_mae,
            "Max_Error": epoch_max_error
        })
        df.to_csv(r"C:\Users\ZYX\Desktop\CDDFuse-main\code\output\time.csv", index=False)
        print(
            f"Epoch {epoch + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, Max Error={max_error_value:.4f}")

        scheduler.step(rmse)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')

        val_loss = validate_model(args, model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')

        scheduler.step(val_loss)

        #if (epoch + 1) % 10 == 0:
            #save_path = r"C:\Users\ZYX\Desktop\CDDFuse-main\predict"
            #with torch.no_grad():  # 禁用梯度计算，减少显存占用

               # visualize_sequence_predictions(output, label, save_path, epoch=epoch + 1, threshold=0.5)


    # 绘制曲线
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df["RMSE"], 'r-', marker='v', label='RMSE')
    plt.plot(df["Epoch"], df["MAE"], 'g--', marker='o', label='MAE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE / MAE Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(df["Epoch"], df["Max_Error"], color='lightblue', alpha=0.6, label='Max Error')
    plt.xlabel('Epochs')
    plt.ylabel('Max Error Value')
    plt.title('Max Error Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def validate_model(args, model, val_loader, criterion, device):
    model.eval()  # 将模型置于评估模式
    running_val_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for data in val_loader:
            # 确保 data 包含至少两个元素: input_seq 和 label
            if len(data) >= 2:
                input_seq, label = data[0], data[1]
            else:
                raise ValueError("Data from val_loader2 does not contain enough elements for input_seq and label.")

            input_seq, label = input_seq.to(args.device), label.to(args.device)
            # 前向传播
            output = model(input_seq)
            label = label / 255.0
            # 计算验证损失
           # mse_loss, ssim_loss = criterion(output, label)
            mse_loss = criterion(output, label)
            #loss = 0.6 * mse_loss + 0.4 * ssim_loss  # 计算总损失
            loss = mse_loss
            #loss = criterion(output, label)
            running_val_loss += loss.item()

    # 返回平均验证损失
    return running_val_loss / len(val_loader)

root_dir = r'C:\Users\ZYX\Desktop\CDDFuse-main\code\data\val\lbl'  # 替换为你的文件夹路径

label_tensors = get_tensors_from_images(root_dir)


def visualize_sequence_predictions(output, label, save_path, epoch=None, threshold=0.5):

    os.makedirs(save_path, exist_ok=True)
    num_samples = output.shape[1]

    # 创建自定义颜色映射
    colors = ['peachpuff', 'cornflowerblue']
    cmap = ListedColormap(colors)

    for idx in range(num_samples):
        # 获取当前样本的预测结果和标签
        output_np = output[idx, 0, 0, :, :].detach().cpu().numpy()  # 取第一个时间步和通道
        label_np = label[idx, 0, 0, :, :].detach().cpu().numpy()  # 取标签的第一个时间步和通道

        binary_output = (output_np > threshold).astype(np.float32)
        binary_label = (label_np > threshold).astype(np.float32)

        plt.figure(figsize=(15, 5))

        # 预测结果图像
        plt.subplot(1, 3, 1)
        # 使用自定义颜色映射，蓝色表示积雪，深绿色表示非积雪
        plt.imshow(binary_output, cmap=cmap, vmin=0, vmax=1)  # 0表示非积雪区域（深绿色），1表示积雪区域（蓝色）
        plt.title(f"Predicted Snow Cover")
        plt.axis('off')

        # 标签图像
        plt.subplot(1, 3, 2)
        # 使用自定义颜色映射，蓝色表示积雪，深绿色表示非积雪
        plt.imshow(binary_label, cmap=cmap, vmin=0, vmax=1)  # 0表示非积雪区域（深绿色），1表示积雪区域（蓝色）
        plt.title(f"True Snow Cover")
        plt.axis('off')

        # 差异图像
        plt.subplot(1, 3, 3)

        # 创建差异图
        diff_image = np.zeros_like(binary_output, dtype=np.float32)

        # 白色：预测与实际一致（预测和实际都为 1 或者预测和实际都为 0）
        diff_image[(binary_output == binary_label)] = 1  # 白色区域（预测正确）

        # 红色：预测与实际不一致（预测和实际不同）
        diff_image[(binary_output != binary_label)] = 2  # 红色区域（预测错误）

        # 显示差异图（白色和红色）
        plt.imshow(diff_image, cmap='coolwarm', vmin=0, vmax=2)  # 白色为一致，红色为不一致
        plt.title("Difference")
        plt.axis('off')

        # 显示图像
        plt.show()

        print(f"Sample {idx + 1}:")
        print(f"  Output range: {output_np.min():.4f} to {output_np.max():.4f}")
        print(f"  Mean value: {output_np.mean():.4f}")
        print(f"  Snow-covered pixels (value > {threshold}): {(output_np > threshold).sum()}")
        print(f"  Total pixels: {output_np.size}")

def val(args, model, val_loader, label_tensors, epoch, loss_train_mean, criterion, path_dict):

    #model= MyRestormerModel()

    start = time.time()
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        # 检查 label_tensors 列表的长度
        if len(label_tensors) < len(val_loader):
            print(
                f"Warning: label_tensors length ({len(label_tensors)}) is less than val_loader length ({len(val_loader)})")
            # 可以选择跳过或重复最后一个标签张量来匹配长度
            label_tensors += label_tensors[-1:] * (len(val_loader) - len(label_tensors))

        criterion = nn.CrossEntropyLoss()
        metric = SegmentationMetric(3)    # 2表示有2个分类，有几个分类就填几
        for i, (opt, sar, label) in enumerate(val_loader):
            opt, sar, label = opt.to(device), sar.to(device), label.to(device)
            label_tensor = label_tensors[i].to(device)

            generate_img = model(opt, sar)

            probs = torch.softmax(generate_img, dim=1)
            preds = torch.argmax(probs, dim=1)
            imgPredict = preds.cpu().numpy()  # 选择最可能的类别

            imgLabel = label_tensor.numpy()  # 可直接换成标注图片
            metric.addBatch(imgPredict, imgLabel)
            pa = metric.pixelAccuracy()
            #cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            print('pa is : %f' % pa, 'mpa is : %f' % mpa, 'mIoU is : %f' % mIoU)