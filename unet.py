import torch.nn.functional as F
import torch.nn as nn

class UNetTransformer(nn.Module):
    def __init__(self, input_channels=3, output_dim=1, img_size=64, d_model=64):
        super(UNetTransformer, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 线性变换到 Transformer 的 d_model
        self.input_proj = nn.Linear(img_size * img_size, d_model)

        # Transformer 处理特征
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=2, num_encoder_layers=1, num_decoder_layers=1
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        if x.dim() == 5:  # (batch_size, channels, seq_len, height, width)
            batch_size, channels, seq_len, height, width = x.shape
            x = x[:, :, -1, :, :]  # 取最后一个时间步

        x = self.encoder(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1)  # 变成 (batch_size, 64, height*width)
        x = x.permute(2, 0, 1)
        x = self.transformer(x, x)  # Transformer 编码
        x = x.permute(1, 2, 0)
        x = x.view(b, 64, h, w)  # 变回 (batch_size, 64, height, width)
        x = self.decoder(x)  # (batch_size, 1, height, width)
        x = (x - x.min()) / (x.max() - x.min() + 1e-5)
        return x
