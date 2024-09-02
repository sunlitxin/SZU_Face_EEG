import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import random
import logging
from sklearn.model_selection import KFold
import argparse

from Criterions import NewCrossEntropy
from Dataset import EEGDataset
from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet
from losses import XYLoss, ArcFace

# 设置随机种子，保证结果的可重复性
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)


def get_pkl_file_paths(directory):
    pkl_file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                full_path = os.path.join(root, file)
                pkl_file_paths.append(full_path)
    return pkl_file_paths


def setup_logging(model_name, loss_name, n_timestep, logname):
    log_dir_name = f'{model_name}_{loss_name}'
    log_dir = os.path.join(log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'{logname}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'Starting training with model {model_name}')
    logging.info(f'Loss: {loss_name}')
    logging.info(f'Datasets: {logname}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)  # 从环境变量中获得 local_rank
    args = parser.parse_args()

    # 分布式训练初始化
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # Base path
    base_path = '/data0/xinyang/SZU_Face_EEG/small/pkl_400X5'

    match = re.search(r'(\d+)X(\d+)', base_path)
    if match:
        n_timestep = int(match.group(1))
        sub_nums = int(match.group(2))
    else:
        n_timestep = 100

    edf_files = get_pkl_file_paths(base_path)
    logname = '_'.join(base_path.split('/')[-2:])
    setup_logging(args.model, 'CELoss', n_timestep, logname)

    dataset = EEGDataset(edf_files)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    num_epochs = 300

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logging.info(f"FOLD {fold + 1}")
        print(f"FOLD {fold + 1}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # 使用 DistributedSampler 确保每个进程加载不同的数据
        train_sampler = DistributedSampler(train_subset, num_replicas=torch.distributed.get_world_size(),
                                           rank=args.local_rank)
        val_sampler = DistributedSampler(val_subset, num_replicas=torch.distributed.get_world_size(),
                                         rank=args.local_rank, shuffle=False)

        # Data loaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=False, sampler=train_sampler)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, sampler=val_sampler)

        if args.model == 'EEGNet':
            model = EEGNet(n_timesteps=n_timestep, n_electrodes=127, n_classes=50)
        elif args.model == 'classifier_EEGNet':
            model = classifier_EEGNet(temporal=500)
        elif args.model == 'classifier_SyncNet':
            model = classifier_SyncNet(temporal=500)
        elif args.model == 'classifier_CNN':
            model = classifier_CNN(num_points=500, n_classes=50)
        elif args.model == 'classifier_EEGChannelNet':
            model = classifier_EEGChannelNet(temporal=500)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # 使用 DistributedDataParallel 包装模型
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_acc = 0.0
        best_epoch = 0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total

            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                test_acc = 100 * correct_test / total_test

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}"
            )
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                  f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}")


if __name__ == '__main__':
    main()
