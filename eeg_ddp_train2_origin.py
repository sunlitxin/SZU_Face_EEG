import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
import mne
from sklearn.model_selection import KFold
import logging
from datetime import datetime
import argparse
import random
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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



def setup_ddp():
    # 初始化分布式训练
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    return device, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def normalize_samples(x):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized

def normalize_samples(x):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized

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
    edf_reader = MNEReader(filetype='edf', method='manual', length=stim_length)
    stim, target_class = ziyan_read(label_file_path)

    # 将标签值减1，以使标签范围从0到49
    target_class = [cls - 1 for cls in target_class]

    xx = edf_reader.get_set(file_path=edf_file_path, stim_list=stim)
    if xx[-1].shape[0] != stim_length:
        xx = pad_last_array(xx, stim_length)
    xx_np = np.array(xx)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_np.shape= {xx_np.shape}")

    # 如果通道数不是127，跳过
    if xx_np.shape[2] != 127:
        logging.info(f"Skipping file {edf_file_path}, expected 127 channels but got {xx_np.shape[2]}.")
        return None, None

    xx_normalized = normalize_samples(xx_np)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_normalized.shape= {xx_normalized.shape}")

    eeg_data = np.transpose(xx_normalized, (0, 2, 1))
    eeg_data = eeg_data[:, np.newaxis, :, :]
    logging.info(f"{os.path.basename(edf_file_path)} - eeg_data.shape= {eeg_data.shape}")

    eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    labels_tensor = torch.tensor(target_class, dtype=torch.long)

    return eeg_data_tensor, labels_tensor

def setup_logging(model_name, loss_name, n_timestep, datadirname):
    log_dir_name = f'{model_name}_{loss_name}'
    log_dir = os.path.join(log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'{datadirname}-{n_timestep}-origin.log')  # Loss-Dataset-
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'recode_name:{log_filename}')
    logging.info('train by eeg_train2_origin.py')
    logging.info(f'Starting training with model {model_name}')
    logging.info(f'Loss: {loss_name}')
    logging.info(f'Datasets: {datadirname}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--prefix', type=str, default=None, help='File prefix to filter EEG data files')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
    args = parser.parse_args()

    device, local_rank = setup_ddp()

    # loss_name = 'XYLoss'
    loss_name = 'CELoss'
    optims = 'Adam'
    # optims = 'AdamW'
    model_name = args.model
    file_prefix = args.prefix
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

    if local_rank == 0:
        print('---------finished load and preprocess data----------')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    num_epochs = 300
    scaler = GradScaler()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_eeg_data)):

        if local_rank == 0:
            logging.info(f"FOLD {fold + 1}")
            print(f"FOLD {fold + 1}")

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

        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif optims == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=0.0001/2)

        train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
        test_dataset = TensorDataset(all_eeg_data[test_idx], all_labels[test_idx])

        # 使用分布式采样器
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

        best_acc = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):

            epoch_start_time = time.time()  # 记录开始时间

            model.train()
            running_loss = torch.tensor(0.0, device=device)
            correct = torch.tensor(0, device=device)
            total = torch.tensor(0, device=device)

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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

                # running_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 汇总并计算平均损失和准确率
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

            epoch_loss = running_loss / total
            epoch_acc = correct.float() / total * 100

            model.eval()
            correct_test = torch.tensor(0, device=device)
            total_test = torch.tensor(0, device=device)

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    with autocast():
                        outputs = model(inputs)

                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            dist.all_reduce(correct_test, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_test, op=dist.ReduceOp.SUM)

            test_acc = correct_test.float() / total_test * 100

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

            epoch_end_time = time.time()  # 记录结束时间
            epoch_duration = epoch_end_time - epoch_start_time  # 计算持续时间

            if local_rank == 0:  # 仅在主进程输出
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                             f"Train Accuracy: {epoch_acc:.2f}%, Test Accuracy: {test_acc:.2f}%, "
                             f"best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                             f"Epoch Duration: {epoch_duration:.2f} seconds")
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                      f"Train Accuracy: {epoch_acc:.2f}%, Test Accuracy: {test_acc:.2f}%, "
                      f"best_acc: {best_acc:.2f}%, best_epoch: {best_epoch + 1}, "
                      f"Epoch Duration: {epoch_duration:.2f} seconds")

    cleanup_ddp()

    if invalid_files:
        if local_rank == 0:
            logging.info("Files skipped due to invalid channel size:")
            for invalid_file in invalid_files:
                logging.info(invalid_file)


if __name__ == '__main__':
    main()
