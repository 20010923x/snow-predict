import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from model.net import Restormer_opt_Encoder, Restormer_opt_Decoder, Restormer_sar_Encoder, Restormer_sar_Decoder
from net2 import ConvLSTM
import torch.fft as fft

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import TransformerBlock
from net2 import ConvLSTM
from vision_transformer import PatchEmbed, GELU, Block
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyRestormerModel(nn.Module):
    def __init__(self):
        super(MyRestormerModel, self).__init__()
        # 光学和SAR的编码器
        self.opt_encoder = Restormer_opt_Encoder(inp_channels=4, out_channels=4, dim=64)
        self.sar_encoder = Restormer_sar_Encoder(inp_channels=1, out_channels=1, dim=64)

        # 光学的解码器
        self.opt_decoder = Restormer_opt_Decoder(inp_channels=4, out_channels=4, dim=64)
        self.sar_decoder = Restormer_sar_Decoder(inp_channels=1, out_channels=1, dim=64)
        # 融合层，带通道注意力
        self.fusion_layer_base = FusionLayerWithAttention(in_channels=64)  # 基础特征融合
        self.fusion_layer_detail = FusionLayerWithAttention(in_channels=64)  # 细节特征融合

    def forward(self, opt, sar):
        n = opt.size(0)   # 假设opt和sar的batch_size是n
        region_fusion_results = []  # 用于存储每个区域的融合结果

        # 对每个区域进行独立的多模态融合
        for i in range(n):
            opt_single_region = opt[i].unsqueeze(0)  # 提取当前区域的光学图像 (1, channels, height, width)
            sar_single_region = sar[i].unsqueeze(0)

            # 1. 编码光学和SAR的基础和细节特征
            opt_base_features, opt_detail_features, _ = self.opt_encoder(opt_single_region)  # 光学特征
            sar_base_features, sar_detail_features, _ = self.sar_encoder(sar_single_region)  # SAR特征

            # 2. 将光学和SAR特征转换到频域
            opt_base_freq = fft.fft2(opt_base_features)  # 光学基础特征频域变换
            sar_base_freq = fft.fft2(sar_base_features)  # SAR基础特征频域变换

            opt_detail_freq = fft.fft2(opt_detail_features)  # 光学细节特征频域变换
            sar_detail_freq = fft.fft2(sar_detail_features)  # SAR细节特征频域变换

            # 确保频域特征为实数类型（使用实部或模长）
            opt_base_freq_real = opt_base_freq.real  # 仅取实部
            sar_base_freq_real = sar_base_freq.real  # 仅取实部

            opt_detail_freq_real = opt_detail_freq.real  # 仅取实部
            sar_detail_freq_real = sar_detail_freq.real  # 仅取实部

            # 融合频域特征
            fused_base_freq = self.fusion_layer_base(opt_base_freq_real, sar_base_freq_real)  # 在频域内融合
            fused_detail_freq = self.fusion_layer_detail(opt_detail_freq_real, sar_detail_freq_real)  # 在频域内融合

            # 频域转换回时域
            fused_base_features = torch.real(fft.ifft2(fused_base_freq))  # 频域特征转换回时域
            fused_detail_features = torch.real(fft.ifft2(fused_detail_freq))  # 频域特征转换回时域

            # 解码融合后的特征
            fused_img_features = self.opt_decoder(opt_single_region, fused_base_features, fused_detail_features)
            if isinstance(fused_img_features, tuple):
                fused_img_features = fused_img_features[0]
            region_fusion_results.append(fused_img_features)
            #fused_img_features = fused_img_features[0]  # 只取出第一个元素作为 Tensor
            final_fusion_result = torch.cat(region_fusion_results, dim=0)  # 在batch维度拼接

            return final_fusion_result


class FeatureExtractor(nn.Module):
    """
    共享的特征提取器，用于提取光学和SAR特征
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.opt_encoder = Restormer_opt_Encoder(inp_channels=4, out_channels=4, dim=64)
        self.sar_encoder = Restormer_sar_Encoder(inp_channels=1, out_channels=1, dim=64)

    def forward(self, opt, sar):
        opt_features, _, _ = self.opt_encoder(opt)  # [B, C, H, W]
        sar_features, _, _ = self.sar_encoder(sar)  # [B, C, H, W]
        return opt_features, sar_features


class FusionLayerWithAttention(nn.Module):
    """
    多模态融合层
    """
    def __init__(self, in_channels):
        super(FusionLayerWithAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, opt_features, sar_features):
        combined = torch.cat((opt_features, sar_features), dim=1)  # [B, 2*C, H, W]
        attention_weights = self.attention(combined)  # [B, C, H, W]
        fused_features = opt_features * attention_weights + sar_features * (1 - attention_weights)
        return fused_features


class PredictModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(PredictModule, self).__init__()

        # 时间特征提取器：ConvLSTM
        self.convlstm = ConvLSTM(input_dim=64,
                                 hidden_dim=[256, 256, 256],
                                 kernel_size=[(3, 3), (3, 3), (3, 3)],
                                 num_layers=3,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)

        # 使用 1x1 卷积层将多通道映射到单通道
        self.conv_decoder = nn.Conv2d(256, 1, kernel_size=1)

        self.residual_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input_seq):
        # ConvLSTM 提取时序特征
        residual = self.residual_conv(input_seq[:, -1, :, :, :])  # 最后一个时间步作为残差
        convlstm_output, _ = self.convlstm(input_seq)
        convlstm_output = convlstm_output[0]

        # 假设你想要最后一个时间步的输出
        convlstm_output = convlstm_output[:, -1, :, :, :]  # 选择最后时间步
        convlstm_output = F.relu(convlstm_output + residual)  # 添加残差连接
        #convlstm_output = F.relu(convlstm_output + residual)  # 添加残差连接

        output = self.conv_decoder(convlstm_output)  # 最终输出单通道
        # 如果需要归一化到 [0, 1]
        output = (output - output.min()) / (output.max() - output.min() + 1e-5)
        # output = output * 10
        return output

