# import torch
# import torch.nn as nn
#
#
# # 定义残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
#         super(ResidualBlock, self).__init__()
#         stride = 2 if downsample else 1
#
#         # 根据 kernel_size 是否为整数来设置 padding
#         if isinstance(kernel_size, tuple):
#             padding = (kernel_size[0] // 2, kernel_size[1] // 2)
#         else:
#             padding = kernel_size // 2
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                                stride=stride,
#                                padding=padding,  # 根据 kernel_size 类型计算 padding
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.elu = nn.ELU()
#
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
#                                padding=padding,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         # 如果需要降采样，调整尺寸
#         if downsample or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.shortcut = nn.Identity()
#
#     def forward(self, x):
#         # 主路径
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.elu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         # 加上 shortcut 跳跃连接
#         out += self.shortcut(x)
#         out = self.elu(out)
#         return out
#
#
# # 扩展的 EEGNet 模型
# class ResEEGNet(nn.Module):
#     def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=1000, n_electrodes=126, n_classes=50, dropout=0.5):
#         super().__init__()
#
#         # 原始卷积层及其扩展
#         self.zoe1 = nn.ZeroPad2d((K1 // 2, K1 // 2 - 1, 0, 0))
#         self.conv1 = nn.Conv2d(1, F1, (1, K1), bias=False)
#         self.bn1 = nn.BatchNorm2d(F1)
#
#         # 残差块
#         self.res_block1 = ResidualBlock(F1, F1 * D, kernel_size=(n_electrodes, 1))
#         self.res_block2 = ResidualBlock(F1 * D, F1 * D, kernel_size=K2, downsample=True)
#
#         self.act1 = nn.ELU()
#         self.pool1 = nn.AvgPool2d((1, 4))
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.zoe3 = nn.ZeroPad2d((K2 // 2, K2 // 2 - 1, 0, 0))
#         self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, K2), bias=False, groups=F1 * D)
#         self.conv4 = nn.Conv2d(F1 * D, F2, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(F2)
#
#         self.act2 = nn.ELU()
#         self.pool2 = nn.AvgPool2d((1, 8))
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.flatten = nn.Flatten()
#
#         # 全连接层
#         self.fc = nn.Linear(F2 * (n_timesteps // 32), n_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.zoe1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#
#         # 残差块
#         x = self.res_block1(x)
#         x = self.res_block2(x)
#
#         x = self.pool1(x)
#         x = self.dropout1(x)
#         x = self.zoe3(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.bn3(x)
#         x = self.act2(x)
#
#         x = self.pool2(x)
#         x = self.dropout2(x)
#
#         x = self.flatten(x)
#         x = self.fc(x)
#
#         return x


import torch
import torch.nn as nn

import torch.nn.functional as F


class EEGNetResidualBlock(nn.Module):
    def __init__(self, F_in, F_out, K, n_electrodes, dropout=0.5):
        super(EEGNetResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(F_in, F_out, (1, K), padding=(0, K // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F_out)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv2d(F_out, F_out, (n_electrodes, 1), groups=F_out, bias=False)
        self.bn2 = nn.BatchNorm2d(F_out)

        self.shortcut = nn.Conv2d(F_in, F_out, kernel_size=1, bias=False) if F_in != F_out else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)  # This will have shape [8, 16, 1, 501]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Upsample residual if dimensions mismatch
        if residual.shape[2] != x.shape[2]:  # Check if the second dimension is different
            residual = F.interpolate(residual, size=(x.shape[2], residual.shape[3]), mode='bilinear',
                                     align_corners=False)

        x += residual  # Adding the residual connection
        return x


class ResEEGNet(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=128, n_electrodes=64, n_classes=50, dropout=0.5):
        super(ResEEGNet, self).__init__()

        self.initial = nn.Sequential(
            nn.ZeroPad2d((K1 // 2, K1 // 2 - 1, 0, 0)),
            nn.Conv2d(1, F1, (1, K1), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )

        # First series of residual blocks
        self.residual_block1 = EEGNetResidualBlock(F1, F1 * D, K1, n_electrodes, dropout)

        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Second series of residual blocks
        self.residual_block2 = EEGNetResidualBlock(F1 * D, F2, K2, n_electrodes, dropout)

        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(F2 * (n_timesteps // 32), n_classes)

    def forward(self, x):
        x = self.initial(x)

        x = self.residual_block1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.residual_block2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x

