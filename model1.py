import torch
import torch.nn as nn
from model.net import Restormer_opt_Encoder, Restormer_opt_Decoder, Restormer_sar_Encoder, Restormer_sar_Decoder
from net2 import ConvLSTM
from TimesNet import TimesNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyRestormerModel(nn.Module):
    def __init__(self):
        super(MyRestormerModel, self).__init__()
        # 光学和SAR的编码器
        self.opt_encoder = Restormer_opt_Encoder(inp_channels=4, out_channels=4, dim=64)
        self.sar_encoder = Restormer_sar_Encoder(inp_channels=1, out_channels=1, dim=64)

        # 光学的解码器
        self.opt_decoder = Restormer_opt_Decoder(inp_channels=4, out_channels=4, dim=64)

        # 融合层，带通道注意力
        self.fusion_layer_base = FusionLayerWithAttention(in_channels=64)  # 基础特征融合
        self.fusion_layer_detail = FusionLayerWithAttention(in_channels=64)  # 细节特征融合

    def forward(self, opt, sar):
        # 1. 编码光学和SAR的基础和细节特征
        opt_base_features, opt_detail_features, _ = self.opt_encoder(opt)  # 光学特征
        sar_base_features, sar_detail_features, _ = self.sar_encoder(sar)  # SAR特征

        # 2. 多模态融合 - 基础特征
        fused_base_features = self.fusion_layer_base(opt_base_features, sar_base_features)
        # 3. 多模态融合 - 细节特征
        fused_detail_features = self.fusion_layer_detail(opt_detail_features, sar_detail_features)

        # 4. 解码融合特征，生成最终的光学图像特征
        fused_img_features = self.opt_decoder(opt, fused_base_features, fused_detail_features)

        return fused_img_features

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
            nn.Conv2d(in_channels, 5, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, opt_features, sar_features):
        combined = torch.cat((opt_features, sar_features), dim=1)  # [B, 2*C, H, W]
        attention_weights = self.attention(combined)  # [B, C, H, W]
        fused_features = opt_features * attention_weights + sar_features * (1 - attention_weights)
        return fused_features


class PredictModule(nn.Module):
    #def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, fusion_dim):
    def __init__(self, configs):
        super(PredictModule, self).__init__()
        # 空间特征提取器：3D-CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.global_pool = nn.AdaptiveAvgPool1d(256)  # 降维到 256
        # 特征融合模块：加入注意力机制
        self.fusion_attention = nn.Sequential(
            nn.Linear(512, 128),  # [512 -> fusion_dim]
            nn.ReLU(),
            nn.Linear(128, 512),  # [fusion_dim -> 512]
            nn.Sigmoid()
        )

        # 解码模块
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128 // 2),
            nn.ReLU(),
            nn.Linear(128, 256 * 256),
            nn.Tanh()
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Step 1: 使用 TimesNet 提取时序特征
        # 假设 x_enc 是时序输入，x_mark_enc 是时间标记输入
        time_features = self.timesnet(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Step 2: 提取空间特征（例如通过 3D CNN 提取）
        spatial_features = self.cnn3d(x_dec)  # 通过解码部分（如图像输入）提取空间特征

        # Step 3: 融合时序和空间特征
        # 假设 time_features 和 spatial_features 已经对齐，可以直接拼接
        combined_features = torch.cat([time_features, spatial_features], dim=2)

        # Step 4: 加入特征融合和注意力机制
        attention_weights = self.fusion_attention(combined_features.view(-1, combined_features.size(-1)))
        attention_weights = attention_weights.view(combined_features.size(0), combined_features.size(1), -1)
        fused_features = combined_features * attention_weights
        # Step 5: 解码融合后的特征
        output = self.decoder(fused_features.view(-1, fused_features.size(-1)))
        output = (output + 1) / 2  # Tanh 输出 [-1, 1] 映射到 [0, 1]
        output = output.view(fused_features.size(0), fused_features.size(1), 1, 256, 256)

        return output

