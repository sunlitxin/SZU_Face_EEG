import logging
import os

import mne
import numpy as np
import torch


def normalizes_setup(eeg_data, norm_type):
    if norm_type == 'Global Normalization' or 'GN':
        # 全局标准化
        eeg_data_normalized = global_normalize_eeg(eeg_data)
    elif norm_type == 'Channel-wise Normalization' or 'CN':
        # 通道内标准化
        eeg_data_normalized = channel_wise_normalize_eeg(eeg_data)
    elif norm_type == 'Time-step Normalization' or 'TN':
        # 时间步标准化
        eeg_data_normalized = time_step_normalize_eeg(eeg_data)
    elif norm_type == 'Sliding Window Normalization' or 'SWN':
        # 滑动窗口标准化
        eeg_data_normalized = sliding_window_normalize_eeg(eeg_data, window_length=200, stride=100)
    else:
        eeg_data_normalized = normalize_samples(eeg_data)
    return eeg_data_normalized


def normalize_samples(x):## 通道内标准化
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized

def global_normalize_eeg(eeg_data):
    """
    对脑电数据进行全局标准化，所有数据点都标准化为 0 均值、1 方差。

    :param eeg_data: 输入的脑电数据, 形状为 (样本数, 时间步长, 通道数)
    :return: 全局标准化后的数据
    """
    mean = np.mean(eeg_data)
    std = np.std(eeg_data)

    if std == 0:
        std = 1  # 避免标准差为 0

    normalized_eeg = (eeg_data - mean) / std
    return normalized_eeg


def channel_wise_normalize_eeg(eeg_data):
    """
    对脑电数据进行通道内标准化，每个通道独立标准化。

    :param eeg_data: 输入的脑电数据, 形状为 (样本数, 时间步长, 通道数)
    :return: 通道内标准化后的数据
    """
    mean = np.mean(eeg_data, axis=1, keepdims=True)  # 按时间步长计算每个通道的均值
    std = np.std(eeg_data, axis=1, keepdims=True)  # 按时间步长计算每个通道的标准差

    # 避免标准差为零
    std[std == 0] = 1

    normalized_eeg = (eeg_data - mean) / std
    return normalized_eeg


def time_step_normalize_eeg(eeg_data):
    """
    对脑电数据进行时间步标准化，每个时间步的所有通道数据标准化。

    :param eeg_data: 输入的脑电数据, 形状为 (样本数, 时间步长, 通道数)
    :return: 时间步标准化后的数据
    """
    mean = np.mean(eeg_data, axis=2, keepdims=True)  # 按通道数计算每个时间步的均值
    std = np.std(eeg_data, axis=2, keepdims=True)  # 按通道数计算每个时间步的标准差

    # 避免标准差为零
    std[std == 0] = 1

    normalized_eeg = (eeg_data - mean) / std
    return normalized_eeg


def sliding_window_normalize_eeg(eeg_data, window_length=200, stride=100):
    """
    对脑电数据进行逐段标准化，使用滑动窗口方法进行分段标准化。

    :param eeg_data: 输入的脑电数据, 形状为 (样本数, 时间步长, 通道数)
    :param window_length: 每个滑动窗口的长度
    :param stride: 滑动窗口的步幅
    :return: 逐段标准化后的数据
    """
    num_samples, num_time_steps, num_channels = eeg_data.shape
    num_windows = (num_time_steps - window_length) // stride + 1

    normalized_windows = []

    for sample_idx in range(num_samples):
        sample_data = eeg_data[sample_idx]
        for start in range(0, num_time_steps - window_length + 1, stride):
            end = start + window_length
            window = sample_data[start:end, :]

            # 对窗口内数据进行标准化
            mean = np.mean(window, axis=0, keepdims=True)
            std = np.std(window, axis=0, keepdims=True)
            std[std == 0] = 1

            normalized_window = (window - mean) / std
            normalized_windows.append(normalized_window)

    normalized_windows = np.array(normalized_windows)

    # 返回标准化后的数据，形状为 (窗口数量, 窗口长度, 通道数)
    return normalized_windows


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


def load_and_preprocess_data(edf_file_path, label_file_path, stim_length_list, norm_type):
    stim_length, n_timestep_end, stride = stim_length_list
    edf_reader = MNEReader(filetype='edf', method='manual', length=n_timestep_end)
    stim, target_class = ziyan_read(label_file_path)

    # 将标签值减1，以使标签范围从0到49
    target_class = [cls - 1 for cls in target_class]

    xx = edf_reader.get_set(file_path=edf_file_path, stim_list=stim)
    # if xx[-1].shape[0] != stim_length:
    #     xx = pad_last_array(xx, stim_length)
    xx_np = np.array(xx)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_np.shape= {xx_np.shape}")

    # 如果通道数不是127，跳过
    if xx_np.shape[2] != 126:
        logging.info(f"Skipping file {edf_file_path}, expected 127 channels but got {xx_np.shape[2]}.")
        return None, None

        # 进行滑动窗口数据增强
    xx_np_augmented = sliding_window_augmentation(xx_np, stim_length, stride=200)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_np_augmented.shape= {xx_np_augmented.shape}")
    # xx_np_augmented = xx_np
    # 将归一化函数应用到增强后的数据
    xx_normalized = normalizes_setup(xx_np_augmented, norm_type)

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
