import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=4, padding=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/4, W/4]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.norm(x)
        return x, (H, W)

# Transformer Encoder Block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

# MLP Decoder
class MLPDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.linear = nn.Linear(embed_dim, out_channels)

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = self.linear(x)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, H, W)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x

# SegFormer 模型
class SegFormer(nn.Module):
    def __init__(self, in_channels=192, embed_dim=256, num_heads=8, num_classes=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        self.encoder = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads) for _ in range(4)
        ])
        self.decoder = MLPDecoder(embed_dim, num_classes)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        for enc in self.encoder:
            x = enc(x)
        x = self.decoder(x, hw_shape)
        return x

class SegFormerTemporal(nn.Module):
    def __init__(self, num_frames=3, channels=64, num_classes=1):
        super().__init__()
        in_channels = num_frames * channels  # 计算输入通道数
        self.segformer = SegFormer(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size, num_frames * channels, height, width)  # 转换为 4D
        x = self.segformer(x)  # 输出 [batch_size, 1, 64, 64]
        output = (x - x.min()) / (x.max() - x.min() + 1e-5)
        return output
