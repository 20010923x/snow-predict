import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.net = nn.Sequential(
            self.conv1,
            nn.BatchNorm1d(out_channels),
            self.relu,
            self.dropout,
            self.conv2,
            nn.BatchNorm1d(out_channels),
            self.relu,
            self.dropout
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        # Ensure time dimensions match for residual connection
        if res.size(2) != out.size(2):
            res = F.interpolate(res, size=out.size(2), mode='nearest')

        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, output_channels, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size - 1) * dilation, dropout=dropout)]
        self.network = nn.Sequential(*layers)

        # 解码器，将时间序列特征映射到二维图像
        self.conv_decoder = nn.Sequential(
            nn.Conv1d(num_channels[-1], 64, kernel_size=3, padding=1),  # 转换到 64 通道
            nn.ReLU(),
            nn.Conv1d(64, 64 * 64, kernel_size=1),  # 映射到 64x64 输出
        )

    def forward(self, x):


        x = x.mean(dim=[-2, -1])  # 对 height 和 width 取均值 -> [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)
        y = self.network(x)  # [batch_size, num_channels[-1], seq_len]
        y = self.conv_decoder(y)  # [batch_size, 64*64, seq_len]
        y = y.mean(dim=-1)  # 对 seq_len 聚合，得到 [batch_size, 64*64]
        y = y.view(y.size(0), 1, 64, 64)  # 调整形状为 [batch_size, 1, 64, 64]
        return y

