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
import xml.etree.ElementTree as ET
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
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized
def normalize_samples_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    对输入张量 x 按照 dim=1（可替换为所需维度）进行归一化。
    x: [N, C, ...]
    """
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    std[std == 0] = 1e-6  # 防止除以 0
    return (x - mean) / std

def normalize_eeg_per_channel(data_tensor: torch.Tensor) -> torch.Tensor:
    """
    对 EEG 数据进行按通道归一化（z-score 标准化），即对每个样本的每个通道，在时间维度上归一化。

    参数:
        data_tensor (torch.Tensor): 输入张量，形状为 [N, 1, C, T]，
                                    其中 N 是样本数，C 是通道数（如127），T 是时间点数。

    返回:
        torch.Tensor: 归一化后的张量，形状不变。
    """
    if data_tensor.ndim != 4:
        raise ValueError(f"输入张量维度应为 [N, 1, C, T]，但得到的是 {data_tensor.shape}")

    mean = data_tensor.mean(dim=-1, keepdim=True)  # 每个样本每个通道的均值
    std = data_tensor.std(dim=-1, keepdim=True)    # 每个样本每个通道的标准差

    std[std == 0] = 1e-6  # 避免除以 0

    normalized_tensor = (data_tensor - mean) / std
    return normalized_tensor

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


# def ziyan_read(file_path):
#     with open(file_path) as f:
#         stim = []
#         target_class = []
#         for line in f.readlines():
#             if line.strip().startswith('Stimulus'):
#                 line = line.strip().split(',')
#                 classes = int(line[1][-2:])
#                 time = int(line[2].strip())
#                 stim.append(time)
#                 target_class.append(classes)
#     return stim, target_class

def ziyan_read(file_path):
    stim = []
    target_class = []

    try:
        # 解析 XML 文件
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 提取命名空间
        ns = {'ns': 'http://www.brainproducts.com/MarkerSet'}

        # 遍历所有 <Marker> 元素
        for marker in root.findall('.//ns:Marker', ns):  # 使用命名空间查找 Marker
            # 获取 <Type>， <Description> 和 <Position>
            marker_type = marker.find('ns:Type', ns).text
            description = marker.find('ns:Description', ns).text
            position = int(marker.find('ns:Position', ns).text)

            # 只处理 Stimulus 类型的 Marker
            if marker_type == 'Stimulus' and description:
                # 提取类别信息，通常在 Description 中以 "S" 开头
                class_str = description.strip().split()[-1]  # 获取类别，去掉前缀 "S"
                if class_str.isdigit():
                    classes = int(class_str)
                    stim.append(position)
                    target_class.append(classes)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    # 打印前几个刺激信息用于调试
    print(f"Stimulus list: {stim[:10]}")
    print(f"Target class list: {target_class[:10]}")

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
#不够长用最后一行补齐
# def pad_last_array(x, n_timestep):
#     # 确保最后一个数组的形状为 (99, 127)
#     if x[-1].shape[0] != n_timestep and x[-1].shape[1] == 127:
#         # 获取前一个数组，用于补全
#         last_row = x[-1][-1:]
#
#         # 用最后一行进行补全
#         padding = last_row
#
#         # 将补全后的数组重新赋值给最后一个元素
#         x[-1] = np.vstack((x[-1], padding))
#     return x

#不够长用0补齐
def pad_last_array(x, n_timestep):
    """
    补齐最后一个样本的时间步长到指定长度 n_timestep。
    使用零填充而非重复最后一行，避免数据偏移。

    Args:
        x (list of np.ndarray): 每个元素形状为 (time, channels)
        n_timestep (int): 目标时间步数

    Returns:
        list of np.ndarray: 补齐后的列表
    """
    if x and x[-1].shape[0] < n_timestep:
        time_len, n_channels = x[-1].shape
        pad_len = n_timestep - time_len
        # 构造零填充部分
        zero_pad = np.zeros((pad_len, n_channels), dtype=x[-1].dtype)
        # 拼接补齐
        x[-1] = np.concatenate([x[-1], zero_pad], axis=0)
    return x


#---------------增加了10个合并在一起------------------
def merge_samples(samples, method='mean'):
    """
    合并一组样本（tensor shape: [merge_size, 1, 127, 500]）为一个样本（[1, 127, 500]）
    method: 'mean', 'max', 'median'
    """
    if method == 'mean':
        merged = samples.mean(dim=0, keepdim=True)
    elif method == 'max':
        merged, _ = samples.max(dim=0, keepdim=True)
    elif method == 'median':
        merged = samples.median(dim=0, keepdim=True).values
    else:
        raise ValueError(f"Unsupported merge method: {method}")
    # merged = normalize_samples_tensor(merged)  #归一化，合并之后进行，有多种归一化方式-------------------------------
    # merged = normalize_eeg_per_channel(merged)  #归一化，合并之后进行，有多种归一化方式-------------------------------
    return merged

import torch


def load_and_preprocess_data(
    edf_file_path,
    label_file_path,
    stim_length,
    do_merge=True,
    merge_method='mean',
    merge_size=10
):
    edf_reader = MNEReader(filetype='edf', method='manual', length=stim_length)
    stim, target_class = ziyan_read(label_file_path)
    target_class = [cls - 1 for cls in target_class]

    xx = edf_reader.get_set(file_path=edf_file_path, stim_list=stim)
    if xx[-1].shape[0] != stim_length:
        xx = pad_last_array(xx, stim_length)

    xx_np = np.array(xx)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_np.shape= {xx_np.shape}")
    xx_normalized = normalize_samples(xx_np)

    eeg_data = np.transpose(xx_normalized, (0, 2, 1))  # [N, 127, 500]
    eeg_data = eeg_data[:, np.newaxis, :, :]           # [N, 1, 127, 500]

    eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    labels_tensor = torch.tensor(target_class, dtype=torch.long)

    if not do_merge:
        logging.info(f"{os.path.basename(edf_file_path)} - No merging applied.")
        return eeg_data_tensor, labels_tensor

    # 合并逻辑
    merged_data, merged_labels = [], []
    i = 0
    while i + merge_size <= len(labels_tensor):
        block_labels = labels_tensor[i:i+merge_size]
        if torch.all(block_labels == block_labels[0]):
            block_data = eeg_data_tensor[i:i+merge_size]  # shape: [merge_size, 1, 127, 500]
            merged_sample = merge_samples(block_data, method=merge_method)
            merged_data.append(merged_sample)
            merged_labels.append(block_labels[0].unsqueeze(0))
            i += merge_size
        else:
            i += 1  # 滑动窗口前进

    if len(merged_data) == 0:
        logging.warning(f"No valid merged samples found in {edf_file_path}")
        return None, None

    merged_data_tensor = torch.cat(merged_data, dim=0)       # [N', 1, 127, 500]
    merged_labels_tensor = torch.cat(merged_labels, dim=0)   # [N']

    logging.info(f"{os.path.basename(edf_file_path)} - merged_data_tensor.shape= {merged_data_tensor.shape}")
    return merged_data_tensor, merged_labels_tensor
#------------------------------------------------------------------------------------------------------------

def setup_logging(model_name, loss_name, n_timestep, datadirname):
    log_dir_name = f'{model_name}_{loss_name}'
    log_dir = os.path.join(log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'{datadirname}-{n_timestep}-Adam.log')  # Loss-Dataset-
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
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
    model_name = args.model

    n_timestep = 500

    file_prefix = args.prefix


    # Base path
    base_path0 = '/data0/xinyang/SZU_Face_EEG/'
    datadirname = 'New_FaceEEG_merge_max_no_normal'
    # base_path = '/data0/xinyang/SZU_Face_EEG/FaceEEG/'

    # base_path = '/data0/xinyang/SZU_Face_EEG/new_eeg_xy'

    base_path = '/data0/xinyang/SZU_Face_EEG/FaceEEG2025_export'

    # base_path = os.path.join(base_path0, datadirname)
    # base_path = '/data0/xinyang/SZU_Face_EEG/small_new'
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
        # 原始代码：
        # eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path, stim_length=n_timestep) #tensor[1200, 1, 127, 500]
        #
         # 不开启融合：
        eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path, stim_length=n_timestep, do_merge=False)

        # #  开启合并（平均融合）：
        # eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path, stim_length=n_timestep, do_merge=True,
        #                                              merge_method='mean')

        # #  开启合并（最大值融合）：
        # eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path, stim_length=n_timestep, do_merge=True,
        #                                             merge_method='max')
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
    num_epochs = 100

    scaler = GradScaler()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_eeg_data)):
        logging.info(f"FOLD {fold + 1}")
        print(f"FOLD {fold + 1}")

        # 实例化模型
        if model_name == 'EEGNet':
            model = EEGNet(n_timesteps=n_timestep, n_electrodes=126, n_classes=40)
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
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]   # 例如使用 2 个 GPU

            # 将模型分配到指定的 GPU
            model = nn.DataParallel(model, device_ids=device_ids)

        # 将模型移动到 GPU 上
        model = model.to(device)
        if loss_name == 'ArcFace':
            margin_loss = ArcFace(
                margin=0.0
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
        optimizer = optim.Adam(model.parameters(), lr=0.00005)
        # optimizer = optim.AdamW(model.parameters(), lr=0.0002)

        train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
        test_dataset = TensorDataset(all_eeg_data[test_idx], all_labels[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
            top1_correct = 0
            top3_correct = 0
            top5_correct = 0
            total_test = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast():
                        outputs = model(inputs)

                    total_test += labels.size(0)

                    # Top-k accuracy
                    _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
                    pred = pred.t()
                    correct = pred.eq(labels.view(1, -1).expand_as(pred))

                    top1_correct += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                    top3_correct += correct[:3].reshape(-1).float().sum(0, keepdim=True).item()
                    top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()

                top1_acc = 100 * top1_correct / total_test
                top3_acc = 100 * top3_correct / total_test
                top5_acc = 100 * top5_correct / total_test

                if top1_acc > best_acc:
                    best_acc = top1_acc
                    best_epoch = epoch

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                f"Top1-Test: {top1_acc:.6f}%, Top3-Test: {top3_acc:.6f}%, Top5-Test: {top5_acc:.6f}%, "
                f"best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                f"Epoch Duration: {epoch_duration:.2f} seconds"
            )

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
                  f"Top1-Test: {top1_acc:.6f}%, Top3-Test: {top3_acc:.6f}%, Top5-Test: {top5_acc:.6f}%, "
                  f"best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                  f"Epoch Duration: {epoch_duration:.2f} seconds")

            best_acc_list.append(best_acc)
        # # 在每个折叠结束后，手动释放内存
        # del train_dataset, test_dataset, train_loader, test_loader
        # model.to('cpu')
        # all_eeg_data = all_eeg_data.to('cpu')
        # all_labels = all_labels.to('cpu')
        # torch.cuda.empty_cache()

    if invalid_files:
        logging.info("Files skipped due to invalid channel size:")
        for invalid_file in invalid_files:
            logging.info(invalid_file)


if __name__ == '__main__':
    main()
