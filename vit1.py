
import torch.nn as nn
import torch.nn.functional as F
from vit import VisionTransformer


class ViTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size, num_layers, num_heads, img_size=256):
        super(ViTModel, self).__init__()

        # ViT 作为特征提取器
        self.vit = VisionTransformer(
            patch_size=patch_size,
            num_heads=num_heads,
            num_classes=hidden_dim  # 输出与 hidden_dim 一致
        )

        # 使用 1x1 卷积层映射到单通道
        self.conv_decoder = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, input_seq):
        batch_size, time_steps, channels, height, width = input_seq.shape

        # ViT 输入格式转换 (batch * time_steps, channels, height, width)
        vit_input = input_seq.view(batch_size * time_steps, channels, height, width)
        vit_output = self.vit(vit_input)  # (batch * time_steps, hidden_dim)

        # 重新调整形状 (batch_size, time_steps, hidden_dim, 1, 1)
        #vit_output = vit_output.view(batch_size, time_steps, hidden_dim, 1, 1)
        vit_output = vit_output[:, -1, :, :, :]  # 取最后一个时间步

        output = self.conv_decoder(vit_output)  # 输出单通道

        return output


