import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import mne
from sklearn.model_selection import KFold
import logging
from datetime import datetime
import argparse
import random

from Criterions import NewCrossEntropy
from Dataset import EEGDataset

from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet
from losses import XYLoss, ArcFace

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
    log_filename = os.path.join(log_dir, f'{logname}.log')  # Loss-Dataset-
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'Starting training with model {model_name}')
    logging.info(f'Loss: {loss_name}')
    logging.info(f'Datasets: {logname}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model to use: EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet',
                        default='EEGNet')
    parser.add_argument('--prefix', type=str, default=None, help='File prefix to filter EEG data files')
    args = parser.parse_args()
    # loss_name = 'XYLoss'
    loss_name = 'CELoss'
    model_name = args.model

    # Base path
    # base_path = '/data0/xinyang/SZU_Face_EEG/FaceEEG/'
    # base_path = '/data0/xinyang/SZU_Face_EEG/eeg_xy'
    base_path = '/data0/xinyang/SZU_Face_EEG/small/pkl_400X5'
    # base_path = '/data0/xinyang/SZU_Face_EEG/small_eeg'

    # 使用正则表达式提取两个数字
    match = re.search(r'(\d+)X(\d+)', base_path)
    if match:
        n_timestep = int(match.group(1))
        sub_nums = int(match.group(2))
    else:
        n_timestep = 100




    edf_files = get_pkl_file_paths(base_path)
    logname = '_'.join(base_path.split('/')[-2:])
    # Setup logging
    setup_logging(model_name, loss_name, n_timestep, logname)


    invalid_files = []

    dataset = EEGDataset(edf_files)


    # # 将数据移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    num_epochs = 300

    for fold, (train_idx, var_idx) in enumerate(kfold.split(dataset)):
        logging.info(f"FOLD {fold + 1}")
        print(f"FOLD {fold + 1}")

        # Create subsets for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, var_idx)

        # Data loaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        # 实例化模型
        if model_name == 'EEGNet':
            model = EEGNet(n_timesteps=n_timestep, n_electrodes=127, n_classes=50)
        elif model_name == 'classifier_EEGNet':
            model = classifier_EEGNet(temporal=500)
        elif model_name == 'classifier_SyncNet':
            model = classifier_SyncNet(temporal=500)
        elif model_name == 'classifier_CNN':
            model = classifier_CNN(num_points=500, n_classes=50)
        elif model_name == 'classifier_EEGChannelNet':
            model = classifier_EEGChannelNet(temporal=500)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 支持多GPU训练
        if torch.cuda.device_count() > 1:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # 例如使用 2 个 GPU

            # 将模型分配到指定的 GPU
            model = nn.DataParallel(model, device_ids=device_ids)

        # 将模型移动到 GPU 上
        model = model.to(device)
        if loss_name == 'ArcFace':
            margin_loss = ArcFace(
                margin=0.5
            )
        elif loss_name == 'XYLoss':
            # robustface
            margin_loss = XYLoss(
                50,
                embedding_size=50,
                s=64,
                m2=0.9,  # for arcface margin
                m3=-0.1,  # for xyloss
                t=0.2,
                errsum=0.3  # (1-φ)^2 * cos(θ) noise决定了φ的上限
            ).train().to(device)
        elif loss_name == 'CELoss':
            print('')
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)


        best_acc = 0.0
        best_epoch = 0
        best_acc_list = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if loss_name == 'CELoss':
                    loss = criterion(outputs, labels)
                else:
                    output = margin_loss(outputs, labels)
                    loss = criterion(output, labels)
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

            best_acc_list.append(best_acc)


    if invalid_files:
        logging.info("Files skipped due to invalid channel size:")
        for invalid_file in invalid_files:
            logging.info(invalid_file)


if __name__ == '__main__':
    main()
