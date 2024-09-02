import pickle
import os
import torch
def load_pkl_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        data, labels = pickle.load(f)
    return data, labels


from torch.utils.data import Dataset, DataLoader


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


