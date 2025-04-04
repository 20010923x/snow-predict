import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, fusion_dim):
        super(LSTMModel, self).__init__()

        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # 卷积层解码器
        self.conv1 = nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_decoder = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # 上采样
        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        """
        input_seq: [batch_size, seq_len, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = input_seq.size()

        # 展平空间维度：从 (channels, height, width) 到 (channels * height * width)
        input_seq = input_seq.view(batch_size, seq_len, -1)  # 变为 (batch_size, seq_len, input_dim)

        # 通过 LSTM 处理时序特征
        lstm_out, _ = self.lstm(input_seq)  # 输出: (batch_size, seq_len, hidden_dim)

        # 假设我们只使用最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # 选择最后一个时间步: (batch_size, hidden_dim)

        # 将 LSTM 输出调整为 4D 张量，适配卷积层
        lstm_out = lstm_out.unsqueeze(2).unsqueeze(3)  # (batch_size, hidden_dim, 1, 1)

        # 上采样还原到目标空间分辨率
        lstm_out = self.upsample(lstm_out)  # (batch_size, hidden_dim, 64, 64)

        # 通过卷积层映射到单通道输出
        output = self.conv_decoder(lstm_out)  # (batch_size, 1, 64, 64)
        output = (output - output.min()) / (output.max() - output.min() + 1e-5)
        return output
