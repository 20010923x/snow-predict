from model.net import Restormer_opt_Encoder, Restormer_opt_Decoder, Restormer_sar_Encoder, Restormer_sar_Decoder
import torch.fft as fft
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d
from net2 import ConvLSTM

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
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, fusion_dim):
        super(PredictModule, self).__init__()

        # 时间特征提取器：ConvLSTM
        self.convlstm = ConvLSTM(input_dim=64,
                                 hidden_dim=[128, 128, 128,128],
                                 kernel_size=[(3, 3), (3, 3), (3, 3),(3, 3)],
                                 num_layers=4,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)

        # 3D-CNN支路
        self.conv3d_1 = Conv3d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3d_2 = Conv3d(in_channels=64, out_channels=4096, kernel_size=3, stride=1, padding=1)
        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.conv_decoder1 = nn.Conv2d(128, 1, kernel_size=1)
        # 残差连接卷积层
        self.residual_conv = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),  # 增加特征通道
            nn.ReLU(),
        )

    def forward(self, input_seq):
        # ConvLSTM支路：提取时序特征
        residual = self.residual_conv(input_seq[:, -1, :, :, :])  # 最后一个时间步作为残差
        convlstm_output, _ = self.convlstm(input_seq)
        convlstm_output = convlstm_output[0]
        convlstm_output = convlstm_output[:, -1, :, :, :]  # 选择最后时间步
        convlstm_output = F.relu(convlstm_output + residual)  # 添加残差连接
        convlstm_output = self.conv_decoder1(convlstm_output)  # 最终输出单通道
        # 3D-CNN支路：处理时空特征
        # 输入的形状是 (batch_size, time_steps, channels, height, width)
        # 在3D-CNN中，我们需要考虑time_steps作为一个维度，所以输入形状应该是(batch_size, channels, time_steps, height, width)
        input_seq_3d = input_seq.permute(0, 2, 1, 3, 4)  # 转换为 (batch_size, channels, time_steps, height, width)
        x = self.pool3d(F.relu(self.conv3d_2(F.relu(self.conv3d_1(input_seq_3d)))))  # 3D卷积处理

        # 输出最后的空间维度
        x = x.view(x.size(0), x.size(1), -1)  # 展平
        x = x.mean(dim=2)  # 对时间步维度进行平均，得到一个2D特征图(batch_size, 128)

        # 将ConvLSTM和3D-CNN的输出融合
        combined_output = torch.cat((convlstm_output, x.view(x.size(0), 1, 64, 64)), dim=1)  # 拼接

        # 经过1x1卷积层得到最终输出
        output = self.conv_decoder(combined_output)  # 最终输出单通道
        #output = torch.sigmoid(output)  # 使用sigmoid限制在0到1之间
        output = output * 100  # 调整输出值的尺度

        return output
