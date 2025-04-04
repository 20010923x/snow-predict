import torch
import torch.nn as nn
import torch.nn.functional as F

class C3DModel(nn.Module):
    def __init__(self, input_channels, num_classes=1):

        super(C3DModel, self).__init__()

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
            nn.Linear(32768, 128),  # [512 -> fusion_dim]
            nn.ReLU(),
            nn.Linear(128, 32768),  # [fusion_dim -> 512]
            nn.Sigmoid()
        )

        # 解码模块
        self.decoder = nn.Sequential(
            nn.Linear(32768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64),
            nn.ReLU()
        )

    def forward(self, x):


        x = x.permute(0, 2, 1, 3, 4)  # 将 seq_len 调整为 depth
        # 提取空间特征
        spatial_features = self.cnn3d(x)  # 通过 3D-CNN 提取空间特征

        # 展平特征以用于全局池化
        spatial_features = spatial_features.view(spatial_features.size(0), spatial_features.size(1), -1)
        spatial_features = self.global_pool(spatial_features)  # 全局池化

        # 注意力机制融合特征
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        attention_weights = self.fusion_attention(spatial_features)
        fused_features = spatial_features * attention_weights

        # 解码输出
        output = self.decoder(fused_features)
        #output = (output + 1) / 2  # 将 Tanh 输出的 [-1, 1] 映射到 [0, 1]
        output = output.view(output.size(0), 1, 64, 64)  # 调整为目标输出形状
        output = (output - output.min()) / (output.max() - output.min() + 1e-5)
        return output
