import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from backbone.resnet import get_model
from backbone.vit import vit_resize
from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet
from losses import ArcFace, XYLoss
from utils.utils_tools import load_and_preprocess_data, find_edf_and_markers_files
from torchvision.models import resnet18, resnet34, resnet50
import torch.nn.functional as F

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

def setup_logging(model_name, loss_name, n_timestep_list, datadirname, norm_type, classification_target, merge_strategys, lr, regularization_type): #[200,800,200] [n_timestep, n_timestep_end, stride ]
    n_timestep, n_timestep_end, stride = n_timestep_list
    log_dir_name = f'{datadirname}'
    log_dir = os.path.join(log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #log_filename = os.path.join(log_dir, f'{datadirname}-{n_timestep}-single_back.log')
    log_filename = os.path.join(log_dir, f'{model_name}-({n_timestep},{n_timestep_end},{stride})-{loss_name}-{norm_type}-{merge_strategys}-{classification_target}-lr{lr}-{regularization_type}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'recode_name:{log_filename}')
    logging.info('train by double_learn.py')
    logging.info(f'Starting training with model {model_name}')
    logging.info(f'Loss: {loss_name}')
    logging.info(f'Datasets: {datadirname}')
    logging.info(f'Merge_strategys: {merge_strategys}')
    logging.info(f'lr: {lr}')

def l1_regularization(model, lambda_l1):
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model to use: EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet',
                        default='EEGNet')
    parser.add_argument('--prefix', type=str, default=None, help='File prefix to filter EEG data files')

    # parser.add_argument('--single_back', type=bool, default=True,  #True: single back   False: sum back
    #                     help='If True, backpropagate loss1 and loss2 separately')
    args = parser.parse_args()
    # loss_name = 'XYLoss'
    loss_name = 'CELoss' ## [CELoss, BCEWLLoss]
    # model_name = args.model
    model_name = 'AttenEEGNet'  #AttenEEGNet   EEGNetTimeWeight    EEGNet
    file_prefix = args.prefix
    n_timestep_list = [100, 100, 100]  #n_timestep, n_timestep_end, stride
    classification_target = 'id' # sex or id
    norm_type = 'Channel-wise Normalization'  ## [Global Normalization: GN, Channel-wise Normalization: CN, Time-step Normalization: TN, Sliding Window Normalization: SWN, L2Norm]
    merge_strategys = None     #'mean'、'max'、'min'、'median'、'sum'、'variance'、'std'、'range'  None
    lr = 0.001

    # 新增正则化类型参数
    regularization_type = 'CL'  # 选择 L1, L2, CL（L1+L2组合）, NL（无L1和L2）
    lambda_l1 = 1e-5  # L1正则化强度
    lambda_l2 = 1e-4  # L2正则化强度


    # Base path
    base_path0 = '/data0/xinyang/SZU_Face_EEG/'
    datadirname = 'xy_std/Face_EEG_HuShuhan'
    base_path = os.path.join(base_path0, datadirname)
    edf_files = find_edf_and_markers_files(base_path, file_prefix)


    # Setup logging
    setup_logging(model_name, loss_name, n_timestep_list, datadirname, norm_type, classification_target, merge_strategys, lr, regularization_type)

    mapping = {
        0: 0, 1: 0, 2: 0, 3: 1, 4: 1,
        5: 1, 6: 0, 7: 0, 8: 1, 9: 1,
        10: 0, 11: 0, 12: 1, 13: 1, 14: 0,
        15: 1, 16: 0, 17: 1, 18: 1, 19: 0,
        20: 1, 21: 0, 22: 1, 23: 0, 24: 1,
        25: 1, 26: 1, 27: 1, 28: 0, 29: 1,
        30: 0, 31: 0, 32: 1, 33: 0, 34: 0,
        35: 1, 36: 1, 37: 0, 38: 1, 39: 0,
        40: 1, 41: 0, 42: 0, 43: 1, 44: 0,
        45: 0, 46: 0, 47: 0, 48: 1, 49: 1
    }

    all_eeg_data = []
    all_labels = []
    invalid_files = []
    n_timestep, n_timestep_end, stride = n_timestep_list

    for base_name, files in edf_files.items():
        edf_file_path = files['edf']
        label_file_path = files['markers']

        if not os.path.exists(label_file_path):
            logging.info(f"Markers file for {edf_file_path} does not exist. Skipping.")
            continue

        eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path, stim_length_list=n_timestep_list, norm_type=norm_type, merge_strategy=merge_strategys)
        print('-------------------')

        if eeg_data is None or labels is None:
            invalid_files.append(edf_file_path)
            continue

        all_eeg_data.append(eeg_data)
        all_labels.append(labels)

    if len(all_eeg_data) == 0:
        logging.info("No valid EEG data found.")
        return

    all_eeg_data = torch.cat(all_eeg_data)
    all_labels = torch.cat(all_labels)

    # 将数据移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_eeg_data = all_eeg_data.to(device)
    all_labels = all_labels.to(device)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # kfold = KFold(n_splits=5, shuffle=False)
    num_epochs = 300

    scaler = GradScaler()
    num_classes = 50 if classification_target == 'id' else 2

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_eeg_data)):
        logging.info(f"FOLD {fold + 1}")
        print(f"FOLD {fold + 1}")

        # 实例化模型
        model2 = get_model(num_classes=num_classes, model_name=model_name, n_timestep=n_timestep)

        # 根据 regularization_type 选择是否加入 L2 正则化
        if regularization_type in ['L2', 'CL']:
            optimizer = optim.Adam(model2.parameters(), lr=lr, weight_decay=lambda_l2)
            print(f'regularization: {regularization_type}')
        else:
            optimizer = optim.Adam(model2.parameters(), lr=lr)  # 无L2正则化
            print(f'no regularization or l1: {regularization_type}')
            # optimizer = torch.optim.SGD(model2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # 支持多GPU训练
        if torch.cuda.device_count() > 1:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # 例如使用 2 个 GPU
            # device_ids = [0, 1, 2, 3]  # 例如使用 2 个 GPU
            # device_ids = [0]  # 例如使用 2 个 GPU
            # model1 = nn.DataParallel(model1, device_ids=device_ids)
            model2 = nn.DataParallel(model2, device_ids=device_ids)


        # 将模型移动到 GPU 上
        # model1 = model1.to(device)
        model2 = model2.to(device)

        if loss_name == 'CELoss':
            criterion = nn.CrossEntropyLoss()
        elif loss_name == 'BCEWLLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_name == 'BCELoss':
            criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")


        train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
        test_dataset = TensorDataset(all_eeg_data[test_idx], all_labels[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        best_acc = 0.0
        best_epoch = 0
        best_acc_list = []
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # 记录开始时间
            model2.train()
            running_loss = 0.0
            test_running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                label_sex = torch.tensor([mapping[label.item()] for label in labels], device=labels.device)
                inputs, labels, label_sex = inputs.to(device), labels.to(device), label_sex.to(device)
                # # 在时间维度切片
                # start_time = 0
                # end_time = start_time + n_timestep
                # sliced_inputs = inputs[:, :, :, start_time:end_time]
                optimizer.zero_grad()

                with autocast():
                    if model_name == 'vit':
                        inputs = vit_resize(inputs)
                    outputs = model2(inputs)
                    if classification_target == 'sex':
                        labels = label_sex
                    loss = criterion(outputs, labels)

                # 根据 regularization_type 添加 L1 或 L1+L2 正则化
                if regularization_type == 'L1':
                    l1_loss = l1_regularization(model2, lambda_l1)
                    loss += l1_loss
                elif regularization_type == 'CL':  # 组合 L1 和 L2 正则化
                    l1_loss = l1_regularization(model2, lambda_l1)
                    loss += l1_loss


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                # predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total

            model2.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    label_sex = torch.tensor([mapping[label.item()] for label in labels], device=labels.device)
                    inputs, labels, label_sex = inputs.to(device), labels.to(device), label_sex.to(device)
                    with autocast():
                        if model_name == 'vit':
                            inputs = vit_resize(inputs)
                        outputs = model2(inputs)
                        if classification_target == 'sex':
                            labels = label_sex
                        loss_test = criterion(outputs, labels)
                        test_running_loss += loss_test.item()
                    _, predicted = torch.max(outputs, 1)
                    # predicted = (outputs > 0.5).float()
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                epoch_test_loss = test_running_loss / len(test_loader)
                test_acc = 100 * correct_test / total_test

                if test_acc >= best_acc:
                    best_acc = test_acc
                    best_epoch = epoch

            epoch_end_time = time.time()  # 记录结束时间
            epoch_duration = epoch_end_time - epoch_start_time  # 计算持续时间

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, TestLoss:{epoch_test_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                f"Epoch Duration: {epoch_duration:.2f} seconds"  # 记录时间
            )
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, TestLoss:{epoch_test_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                  f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                  f"Epoch Duration: {epoch_duration:.2f} seconds")  # 显示时间

            best_acc_list.append(best_acc)


    if invalid_files:
        logging.info("Files skipped due to invalid channel size:")
        for invalid_file in invalid_files:
            logging.info(invalid_file)


if __name__ == '__main__':
    main()