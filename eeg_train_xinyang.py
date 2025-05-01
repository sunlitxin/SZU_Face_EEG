import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet
# 1. 通过 Mixed Precision Training 加速
from torch.cuda.amp import autocast, GradScaler

import random
import numpy as np
import torch

# ================= 固定所有随机种子 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置为确定性计算模式（牺牲一点性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 任意固定数值

# ================= 自定义 Dataset =================
class EEGDataset(Dataset):
    def __init__(self, file_path):
        # 一次性加载整个 npz 文件
        data = np.load(file_path)
        self.eeg_data = data['eeg_data']     # shape: (N, 1, 126, 500)
        self.labels = data['labels']         # shape: (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)  # shape: (1, 126, 500)
        # label = torch.tensor(self.labels[idx], dtype=torch.long)     # scalar
        label = torch.tensor(self.labels[idx] - 1, dtype=torch.long)   #1-40标签CELoss会报错，因此减1
        return eeg, label



# ================ 训练逻辑 =================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0

    for eeg_data, labels in dataloader:
        eeg_data, labels = eeg_data.to(device), labels.to(device)

        optimizer.zero_grad()

        # 使用混合精度
        with autocast():
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)

        # 使用GradScaler进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * eeg_data.size(0)

        # Top-k accuracy
        _, pred_topk = outputs.topk(5, dim=1, largest=True, sorted=True)
        labels = labels.view(-1, 1)

        top1_correct += (pred_topk[:, :1] == labels).sum().item()
        top3_correct += (pred_topk[:, :3] == labels).sum().item()
        top5_correct += (pred_topk == labels).sum().item()

        total += labels.size(0)

    avg_loss = running_loss / total
    return avg_loss, top1_correct / total, top3_correct / total, top5_correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for eeg_data, labels in dataloader:
            eeg_data, labels = eeg_data.to(device), labels.to(device)
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * eeg_data.size(0)

            _, pred_topk = outputs.topk(5, dim=1, largest=True, sorted=True)
            labels = labels.view(-1, 1)

            top1_correct += (pred_topk[:, :1] == labels).sum().item()
            top3_correct += (pred_topk[:, :3] == labels).sum().item()
            top5_correct += (pred_topk == labels).sum().item()

            total += labels.size(0)

    avg_loss = val_loss / total
    return avg_loss, top1_correct / total, top3_correct / total, top5_correct / total


# ================ 主训练流程（5折交叉验证） ================
# 2. 在主训练流程中启用混合精度
def train_with_kfold(eeg_data_dir, model_name='EEGNet', num_epochs=150, k_folds=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EEGDataset(eeg_data_dir)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'\n--- Fold {fold + 1} ---')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)

        # 实例化模型
        model = EEGNet(n_timesteps=500, n_electrodes=126, n_classes=40).to(device)
        if torch.cuda.device_count() > 1:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # 使用多个GPU
            model = nn.DataParallel(model, device_ids=device_ids)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 启用混合精度训练
        scaler = GradScaler()

        # 初始化变量用于记录最佳精度和对应epoch
        best_top1 = 0.0
        best_top3 = 0.0
        best_top5 = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # 记录开始时间
            train_loss, train_top1, train_top3, train_top5 = train_one_epoch(model, train_loader, criterion, optimizer,
                                                                             device, scaler)
            val_loss, val_top1, val_top3, val_top5 = validate(model, val_loader, criterion, device)

            # 更新最佳精度
            if val_top1 > best_top1:
                best_top1 = val_top1
                best_top3 = val_top3
                best_top5 = val_top5
                best_epoch = epoch + 1  # +1因为epoch从0开始
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}  Top1-Train: {train_top1:.4f}| "
                  f"Test Loss: {val_loss:.4f}  Top1-Test: {val_top1:.4f}, Top3-Test: {val_top3:.4f}, Top5-Test: {val_top5:.4f}" 
                  f"Epoch Duration: {epoch_duration:.2f} seconds"
                  )

        print("\nTraining completed!")
        print(f"Best Top1 Accuracy: {best_top1:.4f} at Epoch {best_epoch}")
        print(f"Corresponding Top3 Accuracy: {best_top3:.4f}")
        print(f"Corresponding Top5 Accuracy: {best_top5:.4f}")

# ================ 启动训练 ================

set_seed(42)
# eeg_data_dir = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_eeg/all_face_eeg_0.npz'
eeg_data_dir = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_eeg/all_face_eeg_shuffled.npz'
model_name = 'EEGNet'
num_epochs=150
k_folds=5
batch_size = 64
train_with_kfold(eeg_data_dir, model_name, num_epochs, k_folds, batch_size)  # 根据你的类别数修改 num_classes
