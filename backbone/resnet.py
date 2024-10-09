import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

from backbone.ResEEGNet import ResEEGNet
from backbone.vit import Process_ViT
from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet


def modify_resnet(model, num_classes=50):
    # 修改第一个卷积层以适应单通道输入
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 修改全连接层以输出 50 个类别
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_model(num_classes, model_name,n_timestep):

    if model_name == 'EEGNet':
        model = EEGNet(n_timesteps=n_timestep, n_electrodes=126, n_classes=num_classes, dropout=0.6)
    elif model_name == 'classifier_EEGNet':
        model = classifier_EEGNet(temporal=n_timestep)
    elif model_name == 'classifier_SyncNet':
        model = classifier_SyncNet(temporal=n_timestep)
    elif model_name == 'classifier_CNN':
        model = classifier_CNN(num_points=n_timestep, n_classes=num_classes)  # 适配性别分类
    elif model_name == 'classifier_EEGChannelNet':
        model = classifier_EEGChannelNet(temporal=n_timestep)
    elif model_name == 'DoubletaskNet':
        model = EEGNet(n_timesteps=n_timestep, n_electrodes=126, n_classes=num_classes)  # 适配性别分类
    elif model_name == 'ResEEGNet':
        model = ResEEGNet(n_timesteps=n_timestep, n_electrodes=126, n_classes=num_classes)  # 适配性别分类
    elif model_name == 'r18':
        model = modify_resnet(resnet18(pretrained=False), num_classes=num_classes)
    elif model_name == 'r34':
        model = modify_resnet(resnet34(pretrained=False), num_classes=num_classes)
    elif model_name == 'r50':
        model = modify_resnet(resnet50(pretrained=False), num_classes=num_classes)
    elif model_name == 'vit':
        model = Process_ViT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
