import os
import time
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet
from torch.cuda.amp import autocast, GradScaler
import itertools
from collections import defaultdict
import random

# ================= 设置日志 =================
log_filename = os.path.join(os.getcwd(), 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# ================= 固定所有随机种子 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ================= 自定义 Dataset =================
class EEGDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.eeg_data = data['eeg_data']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx] - 1, dtype=torch.long)
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
        with autocast():
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * eeg_data.size(0)

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

# ================ 主训练流程 =================
def train_with_kfold(eeg_data_dir, model_name='EEGNet', num_epochs=150, k_folds=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EEGDataset(eeg_data_dir)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logging.info(f'\n--- Fold {fold + 1} ---')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)

        model = EEGNet(n_timesteps=500, n_electrodes=126, n_classes=40).to(device)
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=device_ids)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()

        best_top1 = 0.0
        best_top3 = 0.0
        best_top5 = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_loss, train_top1, train_top3, train_top5 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_top1, val_top3, val_top5 = validate(model, val_loader, criterion, device)

            if val_top1 > best_top1:
                best_top1 = val_top1
                best_top3 = val_top3
                best_top5 = val_top5
                best_epoch = epoch + 1

            epoch_end_time = time.time()
            duration = epoch_end_time - epoch_start_time

            logging.info(f"Epoch {epoch + 1}/{num_epochs} | "
                         f"Train Loss: {train_loss:.4f} Train_Acc: {train_top1:.4f} | "
                         f"Test Loss: {val_loss:.4f} Top1: {val_top1:.4f}, Top3: {val_top3:.4f}, Top5: {val_top5:.4f} | "
                         f"Best Acc: {best_top1:.4f}  Best Epoch: {best_epoch} | "
                         f"Epoch Duration: {duration:.2f} sec")

# ================ 启动训练 ================
set_seed(42)
eeg_data_dir = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_eeg/all_face_eeg_shuffled.npz'
model_name = 'EEGNet'
num_epochs = 100
k_folds = 5
batch_size = 128

try:
    train_with_kfold(eeg_data_dir, model_name, num_epochs, k_folds, batch_size)
except Exception as e:
    logging.exception("训练过程中出现异常")
