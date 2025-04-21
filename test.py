# import torch
# from vit_pytorch import ViT
#
#
#
# v = ViT(
#     image_size = 224,
#     patch_size = 32,
#     num_classes = 50,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
#
# img = torch.randn(1, 3, 224, 224)
#
# preds = v(img) # (1, 1000)
# print(preds)
#
# #1560X126


import numpy as np

import numpy as np

import numpy as np

def sliding_window_augmentation(x, window_length=200, stride=200, method=None):
    """
    对脑电数据应用滑动窗口数据增强，并可选地对滑动窗口的片段进行叠加处理。

    :param x: 输入数据，形状为 (刺激数量, 时间步长, 通道数)
    :param window_length: 滑动窗口的长度
    :param stride: 窗口的步幅
    :param method: 叠加方式，可选 'mean'、'max'、'min'、'median'、'sum'、'variance'、'std'、'range'，为 None 时不进行叠加
    :return: 使用滑动窗口增强并叠加后的数据，形状为 (刺激数量, 窗口长度, 通道数)
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

    # 如果 method 为 None，则直接返回增强后的数据片段，形状为 (窗口数量, 窗口长度, 通道数)
    if method is None:
        return augmented_data

    # 否则进行叠加处理，将所有片段叠加为一个与 window_length 相同的片段
    # 初始化叠加后的数据数组，大小为 (num_trials, window_length, num_channels)
    combined_data = np.zeros((num_trials, window_length, num_channels))

    # 对每个试验分别进行叠加
    for i in range(num_trials):
        # 取出当前试验的所有窗口片段
        trial_windows = augmented_data[i * num_windows:(i + 1) * num_windows]

        if method == 'mean':
            combined_data[i] = np.mean(trial_windows, axis=0)
        elif method == 'max':
            combined_data[i] = np.max(trial_windows, axis=0)
        elif method == 'min':
            combined_data[i] = np.min(trial_windows, axis=0)
        elif method == 'median':
            combined_data[i] = np.median(trial_windows, axis=0)
        elif method == 'sum':
            combined_data[i] = np.sum(trial_windows, axis=0)
        elif method == 'variance':
            combined_data[i] = np.var(trial_windows, axis=0)
        elif method == 'std':
            combined_data[i] = np.std(trial_windows, axis=0)
        elif method == 'range':
            combined_data[i] = np.ptp(trial_windows, axis=0)  # ptp = peak to peak, 即 max - min

    return combined_data



if __name__ == '__main__':
    # 假设输入数据形状为 (10, 1000, 64)
    x = np.random.randn(10, 1000, 64)

    # 使用滑动窗口增强，不进行叠加
    augmented_data_no_combine = sliding_window_augmentation(x, window_length=200, stride=200, method=None)

    # 使用滑动窗口增强并叠加，取平均值
    augmented_data_mean = sliding_window_augmentation(x, window_length=200, stride=200, method='mean')

    # 使用滑动窗口增强并叠加，取最大值
    augmented_data_max = sliding_window_augmentation(x, window_length=200, stride=200, method='max')
