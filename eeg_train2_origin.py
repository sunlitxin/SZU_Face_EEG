import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, Subset
import mne
from sklearn.model_selection import KFold
import logging
from datetime import datetime
import argparse
import random
import time  # 添加时间模块
from Criterions import NewCrossEntropy
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

def normalize_samples(x):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x = (x - mean) / (std + 1e-6)

    mean = np.mean(x, axis=2, keepdims=True)
    std = np.std(x, axis=2, keepdims=True)
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized


def sliding_window_augmentation(x, window_length=200, stride=200):
    """
    对脑电数据应用滑动窗口数据增强

    :param x: 输入数据，形状为 (刺激数量, 时间步长, 通道数)
    :param window_length: 滑动窗口的长度
    :param stride: 窗口的步幅
    :return: 使用滑动窗口增强后的数据
    """
    num_trials, num_time_steps, num_channels = x.shape

    # 计算滑动窗口的数量
    num_windows = (num_time_steps - window_length) // stride + 1

    # 初始化增强后的数据列表
    augmented_data = []

    for i in range(num_trials):
        trial_data = x[i]
        for start in range(0, num_time_steps - window_length + 1, stride):
            end = start + window_length
            windowed_data = trial_data[start:end]
            augmented_data.append(windowed_data)

    # 转换为 numpy 数组
    augmented_data = np.array(augmented_data)

    # 新的形状为 (窗口数量, 窗口长度, 通道数)
    return augmented_data
class MNEReader(object):
    def __init__(self, filetype='edf', method='stim', resample=None, length=500, exclude=(), stim_channel='auto',
                 montage=None):
        self.filetype = filetype
        self.file_path = None
        self.resample = resample
        self.length = length
        self.exclude = exclude
        self.stim_channel = stim_channel
        self.montage = montage
        if stim_channel == 'auto':
            assert method == 'manual'

        if method == 'auto':
            self.method = self.read_auto
        elif method == 'stim':
            self.method = self.read_by_stim
        elif method == 'manual':
            self.method = self.read_by_manual
        self.set = None
        self.pos = None

    def get_set(self, file_path, stim_list=None):
        self.file_path = file_path
        self.set = self.method(stim_list)
        return self.set

    def get_pos(self):
        assert self.set is not None
        return self.pos

    def get_item(self, file_path, sample_idx, stim_list=None):
        if self.file_path == file_path:
            return self.set[sample_idx]
        else:
            self.file_path = file_path
            self.set = self.method(stim_list)
            return self.set[sample_idx]

    def read_raw(self):
        if self.filetype == 'bdf':
            raw = mne.io.read_raw_bdf(self.file_path, preload=True, exclude=self.exclude,
                                      stim_channel=self.stim_channel)
            print(raw.info['sfreq'])
        elif self.filetype == 'edf':
            raw = mne.io.read_raw_edf(self.file_path, preload=True, exclude=self.exclude,
                                      stim_channel=self.stim_channel)
        else:
            raise Exception('Unsupported file type!')
        return raw

    def read_by_manual(self, stim_list):
        raw = self.read_raw()
        picks = mne.pick_types(raw.info, eeg=True, stim=False)
        set = []
        for i in stim_list:
            end = i + self.length
            data, times = raw[picks, i:end]
            set.append(data.T)
        return set

    def read_auto(self, *args):
        raw = self.read_raw()
        events = mne.find_events(raw, stim_channel=self.stim_channel, initial_event=True, output='step')
        event_dict = {'stim': 65281, 'end': 0}
        epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True).drop_channels('Status')
        epochs.equalize_event_counts(['stim'])
        stim_epochs = epochs['stim']
        del raw, epochs, events
        return stim_epochs.get_data().transpose(0, 2, 1)


def ziyan_read(file_path):
    with open(file_path) as f:
        stim = []
        target_class = []
        for line in f.readlines():
            if line.strip().startswith('Stimulus'):
                line = line.strip().split(',')
                classes = int(line[1][-2:])
                time = int(line[2].strip())
                stim.append(time)
                target_class.append(classes)
    return stim, target_class


def find_edf_and_markers_files(base_path, file_prefix=None):
    edf_files = {}
    for filename in os.listdir(base_path):
        if filename.endswith('.edf') and (file_prefix is None or filename.startswith(file_prefix)):
            base_name = filename[:-4]
            edf_files[base_name] = {
                'edf': os.path.join(base_path, filename),
                'markers': os.path.join(base_path, base_name + '.Markers')
            }
    return edf_files

def pad_last_array(x, n_timestep):
    # 确保最后一个数组的形状为 (99, 127)
    if x[-1].shape[0] != n_timestep and x[-1].shape[1] == 127:
        # 获取前一个数组，用于补全
        last_row = x[-1][-1:]

        # 用最后一行进行补全
        padding = last_row

        # 将补全后的数组重新赋值给最后一个元素
        x[-1] = np.vstack((x[-1], padding))
    return x
def load_and_preprocess_data(edf_file_path, label_file_path, stim_length):
    stim_length1 = stim_length + 600
    edf_reader = MNEReader(filetype='edf', method='manual', length=stim_length1)
    stim, target_class = ziyan_read(label_file_path)

    # 将标签值减1，以使标签范围从0到49
    target_class = [cls - 1 for cls in target_class]

    xx = edf_reader.get_set(file_path=edf_file_path, stim_list=stim)
    if xx[-1].shape[0] != stim_length1:
        xx = pad_last_array(xx, stim_length)
    xx_np = np.array(xx)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_np.shape= {xx_np.shape}")

    # 如果通道数不是127，跳过
    if xx_np.shape[2] != 127:
        logging.info(f"Skipping file {edf_file_path}, expected 127 channels but got {xx_np.shape[2]}.")
        return None, None

        # 进行滑动窗口数据增强
    xx_np_augmented = sliding_window_augmentation(xx_np, stim_length)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_np_augmented.shape= {xx_np_augmented.shape}")

    # 将归一化函数应用到增强后的数据
    xx_normalized = normalize_samples(xx_np_augmented)

    logging.info(f"{os.path.basename(edf_file_path)} - xx_normalized.shape= {xx_normalized.shape}")

    eeg_data = np.transpose(xx_normalized, (0, 2, 1))
    eeg_data = eeg_data[:, np.newaxis, :, :]
    logging.info(f"{os.path.basename(edf_file_path)} - eeg_data.shape= {eeg_data.shape}")

    eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    # 生成与增强后的数据相对应的标签
    num_augmented_samples = xx_normalized.shape[0]  # 增强后的样本数
    num_windows_per_sample = num_augmented_samples // len(target_class)  # 每个刺激对应的窗口数
    labels_augmented = np.repeat(target_class, num_windows_per_sample)

    # 如果增强后的样本数和生成的标签数不匹配，进行额外处理（边界情况）
    if len(labels_augmented) < num_augmented_samples:
        extra_labels = labels_augmented[-1]  # 如果需要额外标签，可以使用最后一个标签
        labels_augmented = np.concatenate(
            [labels_augmented, [extra_labels] * (num_augmented_samples - len(labels_augmented))])

    labels_tensor = torch.tensor(labels_augmented, dtype=torch.long)

    return eeg_data_tensor, labels_tensor


def setup_logging(model_name, loss_name, n_timestep, datadirname):
    log_dir_name = f'{datadirname}-{model_name}'
    log_dir = os.path.join(log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'{datadirname}-{n_timestep}-origin-4Xwin-time_channel_bn_shuffled.log')  # Loss-Dataset-time_channel_bn
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'recode_name:{log_filename}')
    logging.info('train by eeg_train2_origin.py')
    logging.info(f'Starting training with model {model_name}')
    logging.info(f'Loss: {loss_name}')
    logging.info(f'Datasets: {datadirname}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model to use: EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet',
                        default='EEGNet')
    parser.add_argument('--prefix', type=str, default=None, help='File prefix to filter EEG data files')
    args = parser.parse_args()
    # loss_name = 'XYLoss'
    loss_name = 'CELoss'
    optims = 'Adam'
    # optims = 'AdamW'
    model_name = 'EEGNet'

    file_prefix = args.prefix

    n_timestep = 200
    # Base path
    base_path0 = '/data0/xinyang/SZU_Face_EEG/'
    datadirname = 'small_new'
    # datadirname = 'small_new'
    # base_path = '/data0/xinyang/SZU_Face_EEG/FaceEEG/'
    # base_path = '/data0/xinyang/SZU_Face_EEG/eeg_xy'
    base_path = os.path.join(base_path0, datadirname)
    # base_path = '/data0/xinyang/SZU_Face_EEG/small'#
    # base_path = '/data0/xinyang/SZU_Face_EEG/small_eeg'
    edf_files = find_edf_and_markers_files(base_path, file_prefix)

    # Setup logging
    setup_logging(model_name, loss_name, n_timestep, datadirname)


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

    #kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True)
    num_epochs = 150

    scaler = GradScaler()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_eeg_data)):
        logging.info(f"FOLD {fold + 1}")
        print(f"FOLD {fold + 1}")

        # 实例化模型
        if model_name == 'EEGNet':
            model = EEGNet(n_timesteps=n_timestep, n_electrodes=127, n_classes=50)
        elif model_name == 'classifier_EEGNet':
            model = classifier_EEGNet(temporal=n_timestep)
        elif model_name == 'classifier_SyncNet':
            model = classifier_SyncNet(temporal=n_timestep)
        elif model_name == 'classifier_CNN':
            model = classifier_CNN(num_points=n_timestep, n_classes=50)
        elif model_name == 'classifier_EEGChannelNet':
            model = classifier_EEGChannelNet(temporal=n_timestep)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 支持多GPU训练
        if torch.cuda.device_count() > 1:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]   # 例如使用 2 个 GPU

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
        if optims == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
        elif optims == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=0.0001/2)

        train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
        test_dataset = TensorDataset(all_eeg_data[test_idx], all_labels[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        best_acc = 0.0
        best_epoch = 0
        best_acc_list = []
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # 记录开始时间

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # # 在时间维度切片
                # start_time = 0
                # end_time = start_time + n_timestep
                # sliced_inputs = inputs[:, :, :, start_time:end_time]
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    if loss_name == 'CELoss':
                        loss = criterion(outputs, labels)
                    else:
                        output = margin_loss(outputs, labels)
                        loss = criterion(output, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # # 在时间维度切片
                    # start_time = 0
                    # end_time = start_time + n_timestep
                    # sliced_inputs = inputs[:, :, :, start_time:end_time]
                    with autocast():
                        outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                test_acc = 100 * correct_test / total_test

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch

            epoch_end_time = time.time()  # 记录结束时间
            epoch_duration = epoch_end_time - epoch_start_time  # 计算持续时间

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                f"Epoch Duration: {epoch_duration:.2f} seconds"  # 记录时间
            )
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                  f"Test Accuracy: {test_acc:.2f}%, best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                  f"Epoch Duration: {epoch_duration:.2f} seconds")  # 显示时间

            best_acc_list.append(best_acc)

    if invalid_files:
        logging.info("Files skipped due to invalid channel size:")
        for invalid_file in invalid_files:
            logging.info(invalid_file)


if __name__ == '__main__':
    main()
