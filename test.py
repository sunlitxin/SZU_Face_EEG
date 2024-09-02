import pickle
import os
import torch

import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from eeg_net import EEGNet


# 加载PKL文件数据的函数
def load_pkl_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        data, labels = pickle.load(f)
    return data, labels

# 自定义Dataset类
class EEGDataset(Dataset):
    def __init__(self, pkl_file_paths):
        self.data = []
        self.labels = []
        for pkl_file in pkl_file_paths:
            data, labels = load_pkl_data(pkl_file)
            self.data.append(data)
            self.labels.append(labels)

        # 将所有数据和标签合并到一个张量中
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据加载器的函数
def create_dataloader(data, labels, batch_size=32, shuffle=True, num_workers=4):
    dataset = EEGDataset([])
    dataset.data = data
    dataset.labels = labels
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# 设定PKL文件路径列表
base_path = '/data0/xinyang/SZU_Face_EEG/small'
# base_path = '/data0/xinyang/SZU_Face_EEG/small_eeg'
output_pkl_dir = os.path.join(base_path, 'pkl')

pkl_file_paths = [os.path.join(output_pkl_dir, file) for file in os.listdir(output_pkl_dir) if file.endswith('.pkl')]
n_timestep = 100
# 加载所有数据到EEGDataset
dataset = EEGDataset(pkl_file_paths)

# 使用KFold进行交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = EEGNet(n_timesteps=n_timestep, n_electrodes=127, n_classes=50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}")

    # 根据索引分割数据
    train_data = dataset.data[train_index]
    train_labels = dataset.labels[train_index]
    val_data = dataset.data[val_index]
    val_labels = dataset.labels[val_index]

    # 创建训练和验证的DataLoader
    train_loader = create_dataloader(train_data, train_labels, batch_size=32, shuffle=True, num_workers=4)
    val_loader = create_dataloader(val_data, val_labels, batch_size=32, shuffle=False, num_workers=4)

    # 训练模型
    for epoch in range(10):
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        print(f"Validation Loss for Fold {fold + 1}, Epoch {epoch + 1}: {val_loss / len(val_loader)}")
