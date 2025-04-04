import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXt V2 Block: 深度可分离卷积 + GELU + LayerNorm"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度可分离卷积
        self.norm = nn.LayerNorm(dim)  # 修正 LayerNorm 维度
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # 调整维度适配 LayerNorm
        x = self.norm(x)  # LayerNorm 归一化
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # 调整回 CNN 格式
        return x + residual  # 残差连接

class ConvNeXtV2(nn.Module):
    """ConvNeXt V2 with Spatial Output"""
    def __init__(self, in_channels=640, num_classes=1, dim=768):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=4, stride=1),
            nn.BatchNorm2d(96)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, dim, kernel_size=3, padding=1),
            nn.GELU()
        )

        # **修改解码部分，使其直接输出 64x64**
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch_size, dim, 1, 1]
            nn.Flatten(),  # [batch_size, dim]
            nn.Linear(dim, 64 * 64),  # 线性变换回 64x64
            nn.Tanh(),
            nn.Unflatten(1, (4, 64, 64))  # 变回 [batch_size, 1, 64, 64]
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_block(x)
        x = self.decoder(x)  # **输出 [batch_size, 1, 64, 64]**
        return x



class ConvNeXtV2Temporal(nn.Module):
    """处理 5D 输入 [B, T, C, H, W]"""
    def __init__(self, num_frames=10, channels=64, dim=768):
        super().__init__()
        self.in_channels = num_frames * channels  # 10 * 64 = 640
        self.convnext = ConvNeXtV2(in_channels=self.in_channels, dim=dim)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        # 合并 T 和 C 维度 -> [B, T*C, H, W]
        x = x.view(batch_size, num_frames * channels, height, width)
        output = self.convnext(x)
        # 归一化到 [0, 1]
        output = (output - output.min()) / (output.max() - output.min() + 1e-5)
        return output  # 输出 [B, 1, 64, 64]

