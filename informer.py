import torch
import torch.nn as nn


class InformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length):
        super(InformerModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(12288, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.time_projection = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        if x.dim() == 5:  # 如果输入是 (batch_size, channels, seq_len, height, width)
            batch_size, channels, seq_len, height, width = x.shape
            x = x.view(batch_size, seq_len, -1)  # 重新调整形状，变成 3 维 (batch_size, seq_len, input_dim)

        x = self.encoder(x)  # 送入编码器
        # 调整形状，适配 `MultiheadAttention`
        x = x.permute(1, 0, 2)  # 变成 (seq_len, batch_size, embed_dim)

        # 通过注意力机制
        x, _ = self.attention(x, x, x)

        # 还原形状
        x = x.permute(1, 0, 2)  # 变成 (batch_size, seq_len, 256)

        # 通过解码器
        output = self.decoder(x)  # (batch_size, seq_len, 64*64)

        # 重新调整形状为目标 (batch_size, 1, 64, 64)
        output = output.view(output.shape[0], 1, 64, 64)
        output = (output - output.min()) / (output.max() - output.min() + 1e-5)
        return output

