import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.max_norm = kwargs.pop('max_norm', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm:
            self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
        return super().forward(x)

class EEGNet(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=1500, n_electrodes=127, n_classes=50, dropout=0.5):
        super().__init__()

        self.zoe1 = nn.ZeroPad2d((K1 // 2, K1 // 2 - 1, 0, 0))
        self.conv1 = nn.Conv2d(1, F1, (1, K1), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (n_electrodes, 1), bias=False, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.act1 = nn.ELU()

        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.zoe3 = nn.ZeroPad2d((K2 // 2, K2 // 2 - 1, 0, 0))
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, K2), bias=False, groups=F1 * D)
        self.conv4 = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        self.act2 = nn.ELU()

        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(F2 * (n_timesteps // 32), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.zoe1(x)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)

        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.zoe3(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        print("Shape of x after flattening:", x.shape)
        print("Weight shape in self.fc:", self.fc.weight.shape)
        x = self.fc(x)

        return x

class classifier_EEGNet(nn.Module):
    def __init__(self, spatial=127, temporal=500, n_classes=50):
        super(classifier_EEGNet, self).__init__()
        F1 = 8
        F2 = 16
        D = 2
        first_kernel = temporal // 2
        first_padding = first_kernel // 2
        self.network = nn.Sequential(
            nn.ZeroPad2d((first_padding, first_padding - 1, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, first_kernel)),
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                in_channels=F1, out_channels=F1, kernel_size=(spatial, 1), groups=F1
            ),
            nn.Conv2d(in_channels=F1, out_channels=D * F1, kernel_size=1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(),
            nn.ZeroPad2d((8, 7, 0, 0)),
            nn.Conv2d(
                in_channels=D * F1, out_channels=D * F1, kernel_size=(1, 16), groups=F1
            ),
            nn.Conv2d(in_channels=D * F1, out_channels=F2, kernel_size=1),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(),
        )
        # 计算 fc 的输入尺寸
        self.fc = nn.Linear(F2 * (temporal // 32), n_classes)

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x)


class classifier_SyncNet(nn.Module):
    def __init__(self, spatial=127, temporal=512, n_classes=50):  # 修改 spatial 为 127
        super(classifier_SyncNet, self).__init__()
        K = min(10, spatial)
        Nt = min(40, temporal)
        pool_size = Nt
        b = np.random.uniform(low=-0.05, high=0.05, size=(1, spatial, K))
        omega = np.random.uniform(low=0, high=1, size=(1, 1, K))
        zeros = np.zeros(shape=(1, 1, K))
        phi_ini = np.random.normal(loc=0, scale=0.05, size=(1, spatial - 1, K))
        phi = np.concatenate([zeros, phi_ini], axis=1)
        beta = np.random.uniform(low=0, high=0.05, size=(1, 1, K))
        t = np.reshape(range(-Nt // 2, Nt // 2), [Nt, 1, 1])
        tc = np.single(t)
        W_osc = b * np.cos(tc * omega + phi)
        W_decay = np.exp(-np.power(tc, 2) * beta)
        W = W_osc * W_decay
        W = np.transpose(W, (2, 1, 0))
        bias = np.zeros(shape=[K])
        self.net = nn.Sequential(
            nn.ConstantPad1d((Nt // 2, Nt // 2 - 1), 0),
            nn.Conv1d(
                in_channels=spatial, out_channels=K, kernel_size=1, stride=1, bias=True  # 使用 spatial 作为 in_channels
            ),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.ReLU(),
        )
        self.net[1].weight.data = torch.FloatTensor(W)
        self.net[1].bias.data = torch.FloatTensor(bias)
        self.fc = nn.Linear((temporal // pool_size) * K, n_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class classifier_EEGChannelNet(nn.Module):
    def __init__(self, spatial=127, temporal=500, n_classes=50):
        super(classifier_EEGChannelNet, self).__init__()
        
        self.temporal_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 33), stride=(1, 2), padding=(0, 16)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 33), stride=(1, 2), padding=(0, 32)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 33), stride=(1, 2), padding=(0, 64)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 33), stride=(1, 2), padding=(0, 128)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 33), stride=(1, 2), padding=(0, 256)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
            ),
        ])

        self.spatial_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(127, 1), stride=(2, 1), padding=(63, 0)),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(63, 1), stride=(2, 1), padding=(31, 0)),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(31, 1), stride=(2, 1), padding=(15, 0)),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(15, 1), stride=(2, 1), padding=(7, 0)),
                nn.BatchNorm2d(50),
                nn.ReLU(),
            ),
        ])

        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(200),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(200),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(200),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(200),
                nn.ReLU(),
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(200),
            ),
        ])

        self.shortcuts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=200, out_channels=200, kernel_size=1, stride=2),
                nn.BatchNorm2d(200),
            ),
        ])

        self.final_conv = nn.Conv2d(
            in_channels=200,
            out_channels=50,
            kernel_size=(3, 3),
            stride=1,
            dilation=1,
            padding=0,
        )

        self.fc1 = nn.Linear(50 * 2 * 2, 1000)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        y = []
        for i in range(len(self.temporal_layers)):
            if i == 0:
                y.append(self.temporal_layers[i](x))
            else:
                y.append(self.temporal_layers[i](y[-1]))

        x = torch.cat(y, 1)

        y = []
        for i in range(len(self.spatial_layers)):
            y.append(self.spatial_layers[i](x))

        x = torch.cat(y, 1)

        for i in range(len(self.residual_layers)):
            x = F.relu(self.shortcuts[i](x) + self.residual_layers[i](x))

        x = self.final_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class classifier_CNN(nn.Module):
    def __init__(self, in_channel=127, num_points=500, n_classes=50):
        super(classifier_CNN, self).__init__()
        self.channel = in_channel
        conv1_size = 32
        conv1_stride = 1
        self.conv1_out_channels = 8
        self.conv1_out = int(math.floor(((num_points - conv1_size) / conv1_stride + 1)))
        fc1_in = self.channel * self.conv1_out_channels
        fc1_out = 40
        pool1_size = 128
        pool1_stride = 64
        pool1_out = int(math.floor(((self.conv1_out - pool1_size) / pool1_stride + 1)))
        dropout_p = 0.5
        fc2_in = pool1_out * fc1_out
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv1_out_channels,
            kernel_size=conv1_size,
            stride=conv1_stride,
        )
        self.fc1 = nn.Linear(fc1_in, fc1_out)
        self.pool1 = nn.AvgPool1d(kernel_size=pool1_size, stride=pool1_stride)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(fc2_in, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # Swap the dimensions for conv1d
        x = x.contiguous().view(-1, 1, x.size(-1))
        x = self.conv1(x)
        x = self.activation(x)
        x = x.view(batch_size, self.channel, self.conv1_out_channels, -1)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, self.conv1_out, self.channel * self.conv1_out_channels)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.pool1(x)
        x = x.contiguous().view(batch_size, -1)
        x = self.fc2(x)
        return x




if __name__ == '__main__':
    model = EEGNet()
    x = torch.zeros(64,1,127,1500)
    y = model(x)
    print(y.shape)




