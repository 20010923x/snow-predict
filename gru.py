import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, num_layers=2):
        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        # GRU 层
        self.gru = nn.GRU(input_size=12288, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True)

        # 解码层，将 GRU 的输出映射到 64x64
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64),  # 变成 64x64 维度
            nn.ReLU()
        )

    def forward(self, x):
        # 处理输入格式
        if x.dim() == 5:  # (batch_size, channels, seq_len, height, width)
            batch_size, channels, seq_len, height, width = x.shape
            x = x.view(batch_size, seq_len, -1)  # 变成 (batch_size, seq_len, input_dim * height * width)

        # 通过 GRU
        x, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)

        # 取最后一个时间步的输出
        x = x[:, -1, :]  # (batch_size, hidden_dim)

        # 通过解码层
        output = self.decoder(x)  # (batch_size, 64*64)
        output = output.view(output.shape[0], 1, 64, 64)  # 变成 (batch_size, 1, 64, 64)
        output = (output - output.min()) / (output.max() - output.min() + 1e-5)
        return output
