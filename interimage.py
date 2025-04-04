import torch
import torch.nn as nn

class DWConv(nn.Module):
    """ 深度可分离卷积 """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class InternImageBlock(nn.Module):
    """ InternImage Block，包含动态卷积 """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = DWConv(dim)  # 深度可分离卷积
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Point-wise 1
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Point-wise 2

    def forward(self, x):
        residual = x
        x = self.dwconv(x)  # 先进行深度可分离卷积
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        return x + residual  # 残差连接

class InternImage(nn.Module):
    """InternImage 模型"""
    def __init__(self, in_channels=192, num_classes=1, dim=768):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=4, stride=1),
            nn.BatchNorm2d(96)
        )

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[InternImageBlock(96) for _ in range(3)],
                nn.Conv2d(96, 96, kernel_size=2, stride=2) if i < 3 else nn.Identity()
            )
            self.stages.append(stage)

        # **修正 Decoder 结构**
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # **将 `H×W` 变为 `1×1`**
            nn.Flatten(),  # **变成 `[batch_size, dim]`**
            nn.Linear(96, 64 * 64),  # **确保 `dim=96` 和 `Linear` 匹配**
            nn.Tanh(),
            nn.Unflatten(1, (1, 64, 64))  # **恢复 `[batch_size, 1, 64, 64]`**
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.decoder(x)  # **直接输出 [batch_size, 1, 64, 64]**
        return x



class InternImageTemporal(nn.Module):
    """ 适用于时序数据的 InternImage """
    def __init__(self, num_frames=3, channels=64, num_classes=1):
        super().__init__()
        in_channels = num_frames * channels  # 计算输入通道数 (192)
        self.internimage = InternImage(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size, num_frames * channels, height, width)  # 变为 4D 输入
        output = self.internimage(x)  # 直接输出 [batch_size, 1]
        x = (output - output.min()) / (output.max() - output.min() + 1e-5)
        return x
