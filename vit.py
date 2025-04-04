from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net import TransformerBlock
from net2 import ConvLSTM
from vision_transformer import PatchEmbed, GELU, Block


class VisionTransformer(nn.Module):
    def __init__(
            self, input_shape=(64, 64), patch_size=16, in_chans=64, num_classes=4096, num_features=4096,
            depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans,
                                      num_features=num_features)
        #num_patches = (224 // patch_size) * (224 // patch_size)
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.num_features = num_features
        self.new_feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]
        self.old_feature_shape = [int(224 // patch_size), int(224 // patch_size)]

        # transformer layers
        self.blocks = nn.ModuleList([
            Block(dim=num_features, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                  act_layer=act_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(num_features)

        # classifier head
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        x = self.head(x)
        return x


class ViTConvLSTMForSnow(nn.Module):
    def __init__(self, input_shape=(64, 64), patch_size=16, in_chans=64, num_features=240, time_steps=3):
        super().__init__()

        # ViT模块
        self.vit = VisionTransformer(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans,
                                     num_features=num_features)

        self.new_feature_shape = [64, 64]

        # ConvLSTM模块
        self.conv_lstm = ConvLSTM(input_dim=64,  # 使用ViT的特征维度作为ConvLSTM的输入维度
                                  hidden_dim=[128, 128, 128, 128],
                                  kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3)],
                                  num_layers=4,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)


        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 处理 LSTM 输出
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 输出单通道图像
            nn.Sigmoid()  # 映射到 [0, 1] 区间
        )


    def forward(self, x):
        B, T, C, H, W = x.shape  # [batch_size, time_steps, channels, height, width]

        vit_outputs = []
        for t in range(T):  # 遍历时间步
            feature = self.vit(x[:, t, :, :, :])  # 每个时间步输入ViT，输出特征图
            feature = feature.view(B, self.new_feature_shape[0], self.new_feature_shape[1], -1)
            vit_outputs.append(feature)
        vit_outputs = torch.stack(vit_outputs, dim=1)

        lstm_out, _ = self.conv_lstm(vit_outputs)  # [B, T, hidden_size, H, W]
        lstm_out = lstm_out[-1]  # 获取列表中的最后一层的输出

        # 3. 对每个时间步进行解码，使用解码器
        decoded_all = []
        for t in range(T):  # 遍历时间步
            decoded = self.decoder(lstm_out[:, t, :, :, :])  # 对每个时间步进行解码
            decoded_all.append(decoded)

        # 将所有时间步的解码结果堆叠
        decoded_all = torch.stack(decoded_all, dim=1)  # [B, T, H', W'] 处理后的多个时间步
        decoded_all = decoded_all.permute(0, 1, 4, 3, 2)
        return decoded_all