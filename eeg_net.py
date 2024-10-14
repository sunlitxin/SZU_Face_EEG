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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) #8X8X1X1
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMLayer(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        #exchange
        x = self.sa(x) * x
        x = self.ca(x) * x

        return x

class CALayer(nn.Module):
    def __init__(self, inplace, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, inplace // reduction)
        self.conv1 = nn.Conv2d(inplace, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(hidden_dim, inplace, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, inplace, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
def get_attn_type(
    attn_name,
    planes,
):
    if attn_name == "se":
        reduction = 16
        return SELayer(planes, reduction)
    elif attn_name == "cbam":
        return CBAMLayer(planes)
    elif attn_name == "ca":
        return CALayer(planes)
    else:
        return nn.Identity()
def get_act_type(act_name):
    if act_name == "elu":
        return nn.ELU(inplace=True)
    elif act_name == "silu":
        return nn.SiLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

class AttenEEGNet(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=1000, n_electrodes=127, n_classes=50, dropout=0.5, attn_name = 'cbam', act_name = 'relu'):
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


        self.attn1 = get_attn_type(attn_name, F1)
        self.attn2 = get_attn_type(attn_name, F1 * D)
        self.attn3 = get_attn_type(attn_name, F2)
        self.act = get_act_type(act_name)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # identity1 = x
        x = self.zoe1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.attn1(x)
        # x += identity1
        x = self.act(x)

        # identity2 = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.attn2(x)
        # x += identity2
        x = self.act1(x)


        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.zoe3(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.attn3(x)
        x = self.act2(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x





class EEGNet(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=1000, n_electrodes=127, n_classes=50, dropout=0.5):
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
        x = self.fc(x)
        return x

class EEGNetTimeWeight(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=1000, n_electrodes=127, n_classes=50, dropout=0.5):
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
        self.time_weights = nn.Parameter(torch.ones(n_timesteps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 1, input_size, seq_length)

        # 1. 对时间维度 (最后一维) 进行加权
        # 将时间权重扩展到与输入形状一致 (batch_size, 1, input_size, seq_length)
        time_weighted_x = x * self.time_weights.unsqueeze(0).unsqueeze(0).unsqueeze(1)

        x = self.zoe1(time_weighted_x)
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
        x = self.fc(x)
        return x


class EEGNet_Double_FC(nn.Module):
    def __init__(self, F1=8, F2=16, D=2, K1=64, K2=16, n_timesteps=1000, n_electrodes=127, n_classes1=50, n_classes2=2, dropout=0.5):
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

        self.fc1 = nn.Linear(F2 * (n_timesteps // 32), n_classes1)
        self.fc2 = nn.Linear(F2 * (n_timesteps // 32), n_classes2)

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
        x1 = self.fc1(x)
        x2 = self.fc2(x)

        return x1, x2

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
    x = torch.zeros(64,1,127,100)
    y = model(x)
    print(y.shape)




