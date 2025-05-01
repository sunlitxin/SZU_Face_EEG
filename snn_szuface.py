r"""
In this task, we train a spiking convolutional network to learn the
MNIST digit recognition task.
"""
from argparse import ArgumentParser
import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision

from norse.torch.module.encode import ConstantCurrentLIFEncoder

from eeg_train1 import find_edf_and_markers_files, load_and_preprocess_data


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]




class ConvNet4(torch.nn.Module):
    def __init__(
            self,
            num_channels=1,
            input_height=127,
            input_width=500,  # 新增参数区分高宽
            method="super",
            dtype=torch.float
    ):
        super(ConvNet4, self).__init__()
        # 添加高度方向填充层，确保卷积后尺寸可被2整除
        self.pad = torch.nn.ZeroPad2d((0, 0, 1, 0))  # 顶部填充1行，使高度从127→128

        # 动态计算各层输出形状
        def calc_shape(dim, kernel, stride, pad):
            return (dim + 2 * pad - kernel) // stride + 1

        # 计算各层输出尺寸
        h = input_height + 1  # 填充后的高度（128）
        w = input_width  # 填充后的宽度（500）

        # Conv1 + Pool1 后的尺寸
        h = calc_shape(h, kernel=5, stride=1, pad=0)  # 128→124
        w = calc_shape(w, kernel=5, stride=1, pad=0)  # 500→496
        h = h // 2  # 池化后 124→62
        w = w // 2  # 池化后 496→248

        # Conv2 + Pool2 后的尺寸
        h = calc_shape(h, kernel=5, stride=1, pad=0)  # 62→58
        w = calc_shape(w, kernel=5, stride=1, pad=0)  # 248→244
        h = h // 2  # 池化后 58→29
        w = w // 2  # 池化后 244→122

        # 全连接层输入维度
        self.features_flat = h * w * 64

        # 网络层定义
        self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
        self.fc1 = torch.nn.Linear(self.features_flat, 1024)
        self.lif0 = LIFCell(p=LIFParameters(method=method, alpha=100.0, v_th=0.7))
        self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=100.0, v_th=0.7))
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=100.0))
        self.out = LILinearCell(1024, 50)
        self.dtype = dtype

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        s0, s1, s2, so = None, None, None, None
        voltages = torch.zeros(seq_length, batch_size, 50, device=x.device, dtype=self.dtype)

        for ts in range(seq_length):
            # 处理单个时间步的输入
            z = self.pad(x[ts, :])  # 填充高度 [batch, 1, 128, 500]
            z = self.conv1(z)  # → [batch, 32, 124, 496]
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)  # → [batch, 32, 62, 248]
            z = self.conv2(z)  # → [batch, 64, 58, 244]
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)  # → [batch, 64, 29, 122]
            z = z.view(-1, self.features_flat)  # → [batch, 29*122*64]
            z = self.fc1(z)  # → [batch, 1024]
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.relu(z), so)
            voltages[ts, :, :] = v
        return voltages

class LIFConvNet(torch.nn.Module):
    def __init__(self, input_features, seq_length, input_scale, model="super", only_first_spike=False):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.only_first_spike = only_first_spike
        self.input_features = input_features
        self.rsnn = ConvNet4(
            num_channels=1,
            input_height=127,
            input_width=500,  # 明确传入实际尺寸
            method=model
        )
        self.seq_length = seq_length
        self.input_scale = input_scale

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.constant_current_encoder(x.view(-1, self.input_features) * self.input_scale)
        print("Encoder output spike rate: ", x.sum().item() / x.numel())
        if self.only_first_spike:
            # delete all spikes except for first
            zeros = torch.zeros_like(x.cpu()).detach().numpy()
            idxs = x.cpu().nonzero().detach().numpy()
            spike_counter = np.zeros((batch_size, 127 * 500))
            for t, batch, nrn in idxs:
                if spike_counter[batch, nrn] == 0:
                    zeros[t, batch, nrn] = 1
                    spike_counter[batch, nrn] += 1
            x = torch.from_numpy(zeros).to(x.device)

        x = x.reshape(self.seq_length, batch_size, 1, 127, 500)
        voltages = self.rsnn(x)
        m, _ = torch.max(voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y




def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    clip_grad,
    grad_clip_value,
    epochs,
    log_interval,
    do_plot,
    plot_interval,
    seq_length,
    writer,
):
    model.train().cuda()
    losses = []

    batch_len = len(train_loader)
    step = batch_len * epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(f"data shape: {data.shape}")
        # print(f"target shape: {target.shape}")

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad mean: {param.grad.abs().mean().item()}")

        print("Model output sample (first 5):", output[0][:5])
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        step += 1

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    epochs,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        if step % log_interval == 0:
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", accuracy.item(), step)

            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram(tag, value.data.cpu().numpy(), step)
                if value.grad is not None:
                    writer.add_histogram(
                        tag + "/grad", value.grad.data.cpu().numpy(), step
                    )

        if do_plot and batch_idx % plot_interval == 0:
            ts = np.arange(0, seq_length)
            fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
            axs = axs.reshape(-1)  # flatten
            for nrn in range(10):
                one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
                fig.sca(axs[nrn])
                fig.plot(ts, one_trace)
            fig.xlabel("Time [s]")
            fig.ylabel("Membrane Potential")

            writer.add_figure("Voltages/output", fig, step)

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, test_loader, epoch, method, writer):
    model.eval()
    test_loss = 0
    correct = 0
    n_timestep = 500
    datadirname = 'New_FaceEEG'
    base_path = '/data0/xinyang/SZU_Face_EEG/small_new'
    file_prefix = args.prefix  # 根据实际需要设置

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set {method}: Average loss: {test_loss:.4f}, \
            Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
    )
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)

    return test_loss, accuracy


def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "is_best": is_best,
        },
        path,
    )


def load(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    return model, optimizer


def main(args):
    from torch.utils.data import DataLoader, random_split
    from torch.utils.tensorboard import SummaryWriter
    import logging

    class EEGDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    writer = SummaryWriter()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # ====== EEG 数据加载 ======
    n_timestep = 500
    # base_path = '/data0/xinyang/SZU_Face_EEG/small_new'
    base_path = '/data0/xinyang/SZU_Face_EEG/new_eeg_xy'
    file_prefix = ''  # 根据你实际使用的文件前缀来设置
    edf_files = find_edf_and_markers_files(base_path, file_prefix)

    all_data = []
    all_labels = []

    for base_name, files in edf_files.items():
        edf_file_path = files['edf']
        label_file_path = files['markers']

        if not os.path.exists(label_file_path):
            logging.info(f"Markers file for {edf_file_path} does not exist. Skipping.")
            continue

        eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path, stim_length=n_timestep)
        all_data.append(eeg_data)
        all_labels.append(labels)

    eeg_data = torch.cat(all_data, dim=0)  # shape: [N, 1, 127, 500]
    labels = torch.cat(all_labels, dim=0)  # shape: [N]

    dataset = EEGDataset(eeg_data, labels)

    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # ====== 模型定义 ======
    input_features = 127 * 500  # 注意匹配模型的输入层

    num_classes = 50  # EEG任务类别数
    model = LIFConvNet(input_features=127*500, seq_length=10, input_scale=3.0).to(device)
    # model = LIFConvNet(
    #     input_features=input_features,
    #     seq_length=args.seq_length,
    #     input_scale=args.input_scale,
    #     model=args.method,
    #     only_first_spike=args.only_first_spike,
    # ).to(device)

    # ====== 优化器 ======
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.only_output:
        optimizer = torch.optim.Adam(model.out.parameters(), lr=args.learning_rate)

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(args.epochs):
        training_loss, mean_loss = train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            clip_grad=args.clip_grad,
            grad_clip_value=args.grad_clip_value,
            epochs=args.epochs,
            log_interval=args.log_interval,
            do_plot=args.do_plot,
            plot_interval=args.plot_interval,
            seq_length=args.seq_length,
            writer=writer,
        )
        test_loss, accuracy = test(
            model, device, test_loader, epoch, method=args.method, writer=writer
        )

        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        max_accuracy = np.max(np.array(accuracies))

        if (epoch % args.model_save_interval == 0) and args.save_model:
            model_path = f"eeg-{epoch}.pt"
            save(
                model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                is_best=accuracy > max_accuracy,
            )

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("test_losses.npy", np.array(test_losses))
    np.save("accuracies.npy", np.array(accuracies))

    model_path = "eeg-final.pt"
    save(
        model_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        is_best=accuracy > max_accuracy,
    )



if __name__ == "__main__":
    parser = ArgumentParser(
        "MNIST digit recognition with convolutional SNN. Requires Tensorboard, Matplotlib, and Torchvision"
    )
    parser.add_argument(
        "--only-first-spike",
        type=bool,
        default=False,
        help="Only one spike per input (latency coding).",
    )
    parser.add_argument(
        "--save-grads",
        type=bool,
        default=False,
        help="Save gradients of backward pass.",
    )
    parser.add_argument(
        "--grad-save-interval",
        type=int,
        default=10,
        help="Interval for gradient saving of backward pass.",
    )
    parser.add_argument(
        "--refrac", type=bool, default=False, help="Use refractory time."
    )
    parser.add_argument(
        "--plot-interval", type=int, default=10, help="Interval for plotting."
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Scaling factor for input current.",
    )
    parser.add_argument(
        "--find-learning-rate",
        type=bool,
        default=False,
        help="Use learning rate finder to find learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use by pytorch.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training episodes to do."
    )
    parser.add_argument(
        "--seq-length", type=int, default=10, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of examples in one minibatch.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="super",
        choices=["super", "tanh", "circ", "logistic", "circ_dist"],
        help="Method to use for training.",
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix to use for saving the results"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--clip-grad",
        type=bool,
        default=False,
        help="Clip gradient during backpropagation",
    )
    parser.add_argument(
        "--grad-clip-value", type=float, default=1.0, help="Gradient to clip at."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3, help="Learning rate to use."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="In which intervals to display learning progress.",
    )
    parser.add_argument(
        "--model-save-interval",
        type=int,
        default=50,
        help="Save model every so many epochs.",
    )
    parser.add_argument(
        "--save-model", type=bool, default=True, help="Save the model after training."
    )
    parser.add_argument("--big-net", type=bool, default=False, help="Use bigger net...")
    parser.add_argument(
        "--only-output", type=bool, default=False, help="Train only the last layer..."
    )
    parser.add_argument(
        "--do-plot", type=bool, default=False, help="Do intermediate plots"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1234, help="Random seed to use"
    )
    args = parser.parse_args()
    main(args)

from argparse import ArgumentParser
import os
import uuid
import torch

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFCell
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision

from norse.torch.module.encode import ConstantCurrentLIFEncoder


# class ConvNet4(torch.nn.Module):
#     """
#     A convolutional network with LIF dynamics
#
#     Arguments:
#         num_channels (int): Number of input channels
#         feature_size (int): Number of input features
#         method (str): Threshold method
#     """
#
#     def __init__(
#         self, num_channels=1, feature_size=28, method="super", dtype=torch.float
#     ):
#         super(ConvNet4, self).__init__()
#         self.features = int(((feature_size - 4) / 2 - 4) / 2)
#
#         self.conv1 = torch.nn.Conv2d(num_channels, 32, 5, 1)
#         self.conv2 = torch.nn.Conv2d(32, 64, 5, 1)
#         self.fc1 = torch.nn.Linear(self.features * self.features * 64, 1024)
#         self.lif0 = LIFCell(
#             p=LIFParameters(
#                 method=method, alpha=torch.tensor(100.0), v_th=torch.as_tensor(0.7)
#             ),
#         )
#         self.lif1 = LIFCell(
#             p=LIFParameters(
#                 method=method, alpha=torch.tensor(100.0), v_th=torch.as_tensor(0.7)
#             ),
#         )
#         self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=torch.tensor(100.0)))
#         self.out = LILinearCell(1024, 10)
#         self.dtype = dtype
#
#     def forward(self, x):
#         seq_length = x.shape[0]
#         batch_size = x.shape[1]
#
#         # specify the initial states
#         s0, s1, s2, so = None, None, None, None
#
#         voltages = torch.zeros(
#             seq_length, batch_size, 10, device=x.device, dtype=self.dtype
#         )
#
#         for ts in range(seq_length):
#             z = self.conv1(x[ts, :])
#             z, s0 = self.lif0(z, s0)
#             z = torch.nn.functional.max_pool2d(z, 2, 2)
#             z = 10 * self.conv2(z)
#             z, s1 = self.lif1(z, s1)
#             z = torch.nn.functional.max_pool2d(z, 2, 2)
#             z = z.view(-1, self.features**2 * 64)
#             z = self.fc1(z)
#             z, s2 = self.lif2(z, s2)
#             v, so = self.out(torch.nn.functional.relu(z), so)
#             voltages[ts, :, :] = v
#         return voltages
#
#
#
# class LIFConvNet(torch.nn.Module):
#     def __init__(
#         self,
#         input_features,
#         seq_length,
#         input_scale,
#         model="super",
#         only_first_spike=False,
#     ):
#         super(LIFConvNet, self).__init__()
#         self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
#         self.only_first_spike = only_first_spike
#         self.input_features = input_features
#         self.rsnn = ConvNet4(method=model)
#         self.seq_length = seq_length
#         self.input_scale = input_scale
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = self.constant_current_encoder(
#             x.view(-1, self.input_features) * self.input_scale
#         )
#         if self.only_first_spike:
#             # delete all spikes except for first
#             zeros = torch.zeros_like(x.cpu()).detach().numpy()
#             idxs = x.cpu().nonzero().detach().numpy()
#             spike_counter = np.zeros((batch_size, 127 * 500))
#             for t, batch, nrn in idxs:
#                 if spike_counter[batch, nrn] == 0:
#                     zeros[t, batch, nrn] = 1
#                     spike_counter[batch, nrn] += 1
#             x = torch.from_numpy(zeros).to(x.device)
#
#         x = x.reshape(self.seq_length, batch_size, 1, 127, 500)
#         voltages = self.rsnn(x)
#         m, _ = torch.max(voltages, 0)
#         log_p_y = torch.nn.functional.log_softmax(m, dim=1)
#         return log_p_y



# if __name__ == "__main__":
#     model = LIFConvNet(input_features=127*500, seq_length=10, input_scale=1.0)
#     x = torch.rand(32, 1, 127, 500)  # batch_size = 4
#     print("------------------")
#     output = model(x)
#     print("Output shape:", output.shape)  # should be [4, 50]
