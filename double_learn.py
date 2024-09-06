import argparse
import logging
import os
import time

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from eeg_train1 import find_edf_and_markers_files, load_and_preprocess_data
from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet
from losses import ArcFace, XYLoss



def setup_logging(model_name, loss_name, n_timestep, datadirname):
    log_dir_name = f'{model_name}_{loss_name}'
    log_dir = os.path.join(log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #log_filename = os.path.join(log_dir, f'{datadirname}-{n_timestep}-single_back.log')
    log_filename = os.path.join(log_dir, f'{datadirname}-{n_timestep}-sum_back.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'recode_name:{log_filename}')
    logging.info('train by double_learn.py')
    logging.info(f'Starting training with model {model_name}')
    logging.info(f'Loss: {loss_name}')
    logging.info(f'Datasets: {datadirname}')

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
    loss_name = 'CELoss'
    model_name = args.model

    file_prefix = args.prefix

    single_back = False   #True: single back   False: sum back
    n_timestep = 500
    # Base path
    base_path0 = '/data0/xinyang/SZU_Face_EEG/'
    datadirname = 'New_FaceEEG'
    # base_path = '/data0/xinyang/SZU_Face_EEG/FaceEEG/'
    # base_path = '/data0/xinyang/SZU_Face_EEG/eeg_xy'
    base_path = os.path.join(base_path0, datadirname)
    # base_path = '/data0/xinyang/SZU_Face_EEG/small'#
    # base_path = '/data0/xinyang/SZU_Face_EEG/small_eeg'
    edf_files = find_edf_and_markers_files(base_path, file_prefix)

    # Setup logging
    setup_logging(model_name, loss_name, n_timestep, datadirname)

    # 定义映射关系
    # mapping = {
    #     0: "Male", 1: "Male", 2: "Male", 3: "Female", 4: "Female",
    #     5: "Female", 6: "Male", 7: "Male", 8: "Female", 9: "Female",
    #     10: "Male", 11: "Male", 12: "Female", 13: "Female", 14: "Male",
    #     15: "Female", 16: "Male", 17: "Female", 18: "Female", 19: "Male",
    #     20: "Female", 21: "Male", 22: "Female", 23: "Male", 24: "Female",
    #     25: "Female", 26: "Female", 27: "Female", 28: "Male", 29: "Female",
    #     30: "Male", 31: "Male", 32: "Female", 33: "Male", 34: "Male",
    #     35: "Female", 36: "Female", 37: "Male", 38: "Female", 39: "Male",
    #     40: "Female", 41: "Male", 42: "Male", 43: "Female", 44: "Male",
    #     45: "Male", 46: "Male", 47: "Male", 48: "Female", 49: "Female"
    # }
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

    for base_name, files in edf_files.items():
        edf_file_path = files['edf']
        label_file_path = files['markers']

        if not os.path.exists(label_file_path):
            logging.info(f"Markers file for {edf_file_path} does not exist. Skipping.")
            continue

        eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path, stim_length=n_timestep)
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
    num_epochs = 300

    scaler = GradScaler()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_eeg_data)):
        logging.info(f"FOLD {fold + 1}")
        print(f"FOLD {fold + 1}")

        # 实例化模型
        if model_name == 'EEGNet':
            model1 = EEGNet(n_timesteps=n_timestep, n_electrodes=127, n_classes=50)
            model2 = EEGNet(n_timesteps=n_timestep, n_electrodes=127, n_classes=2)  # 适配性别分类
        elif model_name == 'classifier_EEGNet':
            model1 = classifier_EEGNet(temporal=500)
            model2 = classifier_EEGNet(temporal=500)
        elif model_name == 'classifier_SyncNet':
            model1 = classifier_SyncNet(temporal=500)
            model2 = classifier_SyncNet(temporal=500)
        elif model_name == 'classifier_CNN':
            model1 = classifier_CNN(num_points=500, n_classes=50)
            model2 = classifier_CNN(num_points=500, n_classes=2)  # 适配性别分类
        elif model_name == 'classifier_EEGChannelNet':
            model1 = classifier_EEGChannelNet(temporal=500)
            model2 = classifier_EEGChannelNet(temporal=500)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 支持多GPU训练
        if torch.cuda.device_count() > 1:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # 例如使用 2 个 GPU
            model1 = nn.DataParallel(model1, device_ids=device_ids)
            model2 = nn.DataParallel(model2, device_ids=device_ids)


        # 将模型移动到 GPU 上
        model1 = model1.to(device)
        model2 = model2.to(device)

        criterion = nn.CrossEntropyLoss()

        # 如果 single_back 为 True，使用两个独立的优化器
        if single_back:
            optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
            optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        else:
            # 否则只使用一个优化器
            optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=0.001)

        train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
        test_dataset = TensorDataset(all_eeg_data[test_idx], all_labels[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        best_acc = 0.0
        best_epoch = 0
        best_acc_list = []
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # 记录开始时间

            model1.train()
            model2.train()
            running_loss = 0.0
            correct1 = 0
            correct2 = 0
            total = 0

            for inputs, labels in train_loader:
                # Map each tensor value to its corresponding label
                label_sex = torch.tensor([mapping[label.item()] for label in labels], device=labels.device)
                inputs, labels, label_sex = inputs.to(device), labels.to(device), label_sex.to(device)

                if single_back:
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                else:
                    optimizer.zero_grad()

                with autocast():
                    outputs1 = model1(inputs)
                    outputs2 = model2(inputs)
                    loss1 = criterion(outputs1, labels)
                    loss2 = criterion(outputs2, label_sex)
                    if single_back:
                        # 分开反向传播
                        scaler.scale(loss1).backward(retain_graph=True)  # 保留计算图以便第二次反向传播
                        scaler.scale(loss2).backward()

                        # 分别更新参数
                        scaler.step(optimizer1)
                        scaler.step(optimizer2)
                    else:
                        # 一起反向传播
                        loss = loss1 + loss2
                        scaler.scale(loss).backward()

                        # 更新参数
                        scaler.step(optimizer)

                    scaler.update()

                running_loss += loss1.item() + loss2.item()
                _, predicted1 = torch.max(outputs1, 1)
                _, predicted2 = torch.max(outputs2, 1)
                total += labels.size(0)
                correct1 += (predicted1 == labels).sum().item()
                correct2 += (predicted2 == label_sex).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc1 = 100 * correct1 / total
            epoch_acc2 = 100 * correct2 / total

            model1.eval()
            model2.eval()
            correct_test1 = 0
            correct_test2 = 0
            total_test = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    label_sex = torch.tensor([mapping[label.item()] for label in labels], device=labels.device)
                    inputs, labels, label_sex = inputs.to(device), labels.to(device), label_sex.to(device)

                    with autocast():
                        outputs1 = model1(inputs)
                        outputs2 = model2(inputs)

                    _, predicted_test1 = torch.max(outputs1, 1)
                    _, predicted_test2 = torch.max(outputs2, 1)
                    total_test += labels.size(0)
                    correct_test1 += (predicted_test1 == labels).sum().item()
                    correct_test2 += (predicted_test2 == label_sex).sum().item()

                test_acc1 = 100 * correct_test1 / total_test
                test_acc2 = 100 * correct_test2 / total_test

                if test_acc1 > best_acc:
                    best_acc = test_acc1
                    best_epoch = epoch

                epoch_end_time = time.time()  # 记录结束时间
                epoch_duration = epoch_end_time - epoch_start_time  # 计算持续时间

            # logging.info(
            #     f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
            #     f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
            #     f"Epoch Duration: {epoch_duration:.2f} seconds"  # 记录时间
            # )
            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
            #       f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
            #       f"Epoch Duration: {epoch_duration:.2f} seconds")  # 显示时间

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy (Model1): {epoch_acc1:.2f}%, "
                f"Train Accuracy (Model2): {epoch_acc2:.2f}%, Test Accuracy (Model1): {test_acc1:.2f}%, "
                f"Test Accuracy (Model2): {test_acc2:.2f}%, best_acc (Model1): {best_acc:.2f}%, "
                f"best_epoch: {best_epoch + 1}, Epoch Duration: {epoch_duration:.2f} seconds"
            )
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy (Model1): {epoch_acc1:.2f}%, "
                f"Train Accuracy (Model2): {epoch_acc2:.2f}%, Test Accuracy (Model1): {test_acc1:.2f}%, "
                f"Test Accuracy (Model2): {test_acc2:.2f}%, best_acc (Model1): {best_acc:.2f}%, "
                f"best_epoch: {best_epoch + 1}, Epoch Duration: {epoch_duration:.2f} seconds"
            )

            best_acc_list.append(best_acc)

        # # 在每个折叠结束后，手动释放内存
        # del train_dataset, test_dataset, train_loader, test_loader
        # model1.to('cpu')
        # model2.to('cpu')
        # all_eeg_data = all_eeg_data.to('cpu')
        # all_labels = all_labels.to('cpu')
        # torch.cuda.empty_cache()

    if invalid_files:
        logging.info("Files skipped due to invalid channel size:")
        for invalid_file in invalid_files:
            logging.info(invalid_file)


if __name__ == '__main__':
    main()