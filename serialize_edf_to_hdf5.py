import os
import numpy as np
import torch
from joblib import Parallel, delayed
from eeg_train1 import MNEReader, ziyan_read, load_and_preprocess_data, find_edf_and_markers_files, normalize_samples
import einops
import h5py
from tqdm import tqdm

## Usage：此文件是将edf读取为hdf5文件，可以通过调节参数stim_length 和 slice_set_nums分别选择读取edf的 time-length以及对 time-length进行几等分的切分

def trial_average(eeg, axis=0):
    #  [1000, 127]
    ave = np.mean(eeg, axis=axis)
    std = np.std(eeg, axis=axis)
    return (eeg - ave) / std

def save_to_hdf5(data, labels, output_file_path, segment_index):
    with h5py.File(output_file_path, 'a') as f:
        f.create_dataset(f'data_segment_{segment_index}', data=data.numpy(), compression='gzip')
        f.create_dataset(f'labels_segment_{segment_index}', data=labels.numpy(), compression='gzip')
    print(f"Data and labels saved to {output_file_path}")

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

def load_and_preprocess_data(edf_file_path, label_file_path, stim_length, output_hdf5_base_path, slice_set_nums):
    edf_reader = MNEReader(filetype='edf', method='manual', length=stim_length)
    stim, target_class = ziyan_read(label_file_path)

    # 将标签值减1，以使标签范围从0到49
    target_class = [cls - 1 for cls in target_class]

    # 获取数据
    xx = edf_reader.get_set(file_path=edf_file_path, stim_list=stim)
    if xx[-1].shape[0] != stim_length:
        xx = pad_last_array(xx, stim_length)
    xx_np = np.array(xx)
    print(f"{os.path.basename(edf_file_path)} - xx_np.shape= {xx_np.shape}")

    # 如果通道数不是127，跳过
    if xx_np.shape[2] != 127:
        print(f"Skipping file {edf_file_path}, expected 127 channels but got {xx_np.shape[2]}.")
        return None, None

    # 划分为多个subsets
    segment_length = stim_length // slice_set_nums

    for i in range(slice_set_nums):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = xx_np[:, start_idx:end_idx, :]

        # 归一化处理
        segment_normalized = normalize_samples(segment)
        print(f"{os.path.basename(edf_file_path)} - Segment {i+1} normalized.shape= {segment_normalized.shape}")

        # 转换数据为 (samples, channels, time) 并添加新维度
        eeg_data = np.transpose(segment_normalized, (0, 2, 1))
        eeg_data = eeg_data[:, np.newaxis, :, :]
        print(f"{os.path.basename(edf_file_path)} - Segment {i+1} eeg_data.shape= {eeg_data.shape}")

        # 转换为张量
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        labels_tensor = torch.tensor(target_class, dtype=torch.long)
        print(f"eeg_data_tensor Segment {i + 1} shape: {eeg_data_tensor.shape}")
        print(f"labels_tensor Segment {i + 1} to {labels_tensor.shape}")

        base_path_without_ext = os.path.splitext(output_hdf5_base_path)[0]
        # 生成输出文件路径（添加段编号）
        output_hdf5_path = f"{base_path_without_ext}.h5"

        # 保存为HDF5格式
        save_to_hdf5(eeg_data_tensor, labels_tensor, output_hdf5_path, i + 1)
        print(f"Saved Segment {i+1} to {output_hdf5_path}")

    return eeg_data_tensor, labels_tensor

def process_file(edf_file_path, label_file_path, stim_length, output_hdf5_dir, slice_set_nums):
    output_hdf5_path = os.path.join(output_hdf5_dir, os.path.basename(edf_file_path).replace('.edf', '.h5'))
    return load_and_preprocess_data(edf_file_path, label_file_path, stim_length, output_hdf5_path, slice_set_nums)

# 并行处理多个文件
def parallel_process_files(edf_files, label_file_path, stim_length, output_hdf5_dir, slice_set_nums, n_jobs=-1):
    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(edf_file, label_file_path, stim_length, output_hdf5_dir, slice_set_nums)
        for edf_file in edf_files
    )

if __name__ == "__main__":

    # 设置参数
    sub_stim_length = 2000
    slice_set_nums = 1
    stim_length = sub_stim_length * slice_set_nums

    # Base path
    base_path = '/data0/xinyang/SZU_Face_EEG/eeg_nv/11'

    output_hdf5_dir = os.path.join(base_path, f"hdf5_{sub_stim_length}X{slice_set_nums}")

    # 查找EDF和对应的标记文件
    file_prefix = None
    edf_files = find_edf_and_markers_files(base_path, file_prefix)

    # 确保输出目录存在
    os.makedirs(output_hdf5_dir, exist_ok=True)

    # 处理每个EDF文件和标记文件对
    for base_name, files in edf_files.items():
        edf_file_path = files['edf']
        label_file_path = files['markers']

        # 检查标记文件是否存在
        if not os.path.exists(label_file_path):
            print(f"Markers file for {edf_file_path} does not exist. Skipping.")
            continue

        # 并行处理EDF文件
        parallel_process_files([edf_file_path], label_file_path, stim_length, output_hdf5_dir, slice_set_nums, n_jobs=1)
