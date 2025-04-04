import torch
import torch.nn as nn
from einops import rearrange

class TimeSformer(nn.Module):
    def __init__(self, img_size, patch_size, num_frames, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1):
        super(TimeSformer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.token_dim = dim

        # Patch Embedding
        self.patch_embedding = nn.Conv3d(
            in_channels=channels,
            out_channels=dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # Positional Embedding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_frames * self.num_patches, dim)
        )

        # Transformer blocks
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        # 解码模块
        self.decoder = nn.Sequential(
            nn.Linear(dim, 64 * 64),  # 将最后的特征映射为二维图像
            nn.Tanh(),
            nn.Unflatten(1, (1, 64, 64))  # 恢复为 [batch_size, 1, 64, 64]
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape

        # Patch Embedding
        #x = rearrange(x, "b t c h w -> b c t h w")
        x = self.patch_embedding(x)
        x = rearrange(x, "b d t h w -> b t (h w) d")
        x = rearrange(x, "b t p d -> b (t p) d")

        # Adjust positional encoding if necessary
        if x.size(1) != self.positional_encoding.size(1):
            self.positional_encoding = nn.Parameter(
                torch.randn(1, x.size(1), self.token_dim, device=x.device)
            )

        x += self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)
        x = self.temporal_transformer(x)
        x = x.mean(dim=0)

        # Decoder
        x = self.decoder(x)
        output = (x - x.min()) / (x.max() - x.min() + 1e-5)
        return output
