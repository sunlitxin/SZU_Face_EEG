
"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange


# 固定所有随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设定种子值
SEED = 2023
set_seed(SEED)

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = '/data0/xinyang/mapping/NICE_EEG_running/results/'
model_idx = 'test0'

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='80', type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=128):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 64, (1, 25), stride=(1, 2)),   # [N,64,126,238]
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, (5, 1), stride=(2, 1)),   # [N,64,61,238]
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((2, 5)),                      # [N,64,30,46]
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_size, kernel_size=1),
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x):
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
# # 模块定义
# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x

class FlattenHead(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=128):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Module):
    def __init__(self, input_dim=None, proj_dim=512, drop_proj=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.drop_proj = drop_proj

        # 占位，稍后在 forward 中动态创建
        self.net = None

    def forward(self, x):
        if self.net is None:
            # 第一次调用时初始化网络
            self.input_dim = x.shape[1]
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.proj_dim),
                ResidualAdd(nn.Sequential(
                    nn.GELU(),
                    nn.Linear(self.proj_dim, self.proj_dim),
                    nn.Dropout(self.drop_proj),
                )),
                nn.LayerNorm(self.proj_dim)
            ).to(x.device)
        return self.net(x)



class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=512, proj_dim=512, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return super().forward(x)


class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 400
        self.batch_size = 256
        self.batch_size_test = 400
        self.batch_size_img = 500
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/data0/xinyang/RichEEGData/PreprocessedEEGData'
        self.img_data_path = './dnn_feature/'
        self.test_center_path = './dnn_feature/'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load('/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_eeg/all_face_eeg.npz')
        train_data = train_data['eeg_data']
        # train_data = np.mean(train_data, axis=1)
        # train_data = np.expand_dims(train_data, axis=1)

        # test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy',
        #                     allow_pickle=True).item()
        # test_data = test_data['preprocessed_eeg_data']
        # test_data = np.mean(test_data, axis=1)
        # test_data = np.expand_dims(test_data, axis=1)

        return train_data, test_label

    # def get_image_data(self):
    #
    #     train_img_feature = np.load('/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_img_future/all_face_img_0.npz')
    #     train_img_feature = train_img_feature['features']
    #     test_img_feature = []
    #     # test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)
    #
    #     train_img_feature = np.squeeze(train_img_feature)
    #     # test_img_feature = np.squeeze(test_img_feature)
    #
    #     return train_img_feature, test_img_feature

    def get_image_data(self):
        train_img_feature_path = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_img_future/clip_features_faces_new.pt'
        test_img_feature = []

        # 判断文件扩展名
        if train_img_feature_path.endswith('.pt'):
            data = torch.load(train_img_feature_path)
            train_img_feature = data['features']
            train_img_feature = train_img_feature.numpy() if isinstance(train_img_feature,
                                                                        torch.Tensor) else train_img_feature
        elif train_img_feature_path.endswith('.npz'):
            data = np.load(train_img_feature_path)
            train_img_feature = data['features']
        else:
            raise ValueError(f"不支持的文件格式: {train_img_feature_path}")

        # 去除多余维度（如果有）
        train_img_feature = np.squeeze(train_img_feature)

        return train_img_feature, test_img_feature
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):

        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        # train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_eeg, test_label = self.get_eeg_data()
        train_img_feature, _ = self.get_image_data()
        # test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        # train_shuffle = np.random.permutation(len(train_eeg))
        # train_eeg = train_eeg[train_shuffle]
        # train_img_feature = train_img_feature[train_shuffle]
        seed_value = 2025
        np.random.seed(seed_value)
        indices = np.arange(len(train_eeg))
        np.random.shuffle(indices)
        train_eeg = train_eeg[indices]
        train_img_feature = train_img_feature[indices]

        #切分数据集为train, val, test
        test_eeg = torch.from_numpy(train_eeg[:200])
        val_eeg = torch.from_numpy(train_eeg[-740:])
        train_eeg = torch.from_numpy(train_eeg[200:-740])

        test_image = torch.from_numpy(train_img_feature[:200])
        val_image = torch.from_numpy(train_img_feature[-740:])
        train_image = torch.from_numpy(train_img_feature[200:-740])

        # val_eeg = torch.from_numpy(train_eeg[:740])
        # val_image = torch.from_numpy(train_img_feature[:740])
        #
        # train_eeg = torch.from_numpy(train_eeg[740:])
        # train_image = torch.from_numpy(train_img_feature[740:])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size,
                                                          shuffle=False)
        if isinstance(test_eeg, np.ndarray):
            test_eeg = torch.from_numpy(test_eeg)
        if not isinstance(test_image, torch.Tensor):
            test_image = torch.from_numpy(test_image)
        test_center = test_image

        if not isinstance(test_label, torch.Tensor):
            test_label = torch.from_numpy(test_label)

        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test,
                                                           shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()),
            lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):
                eeg = Variable(eeg.cuda().type(self.Tensor))
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)
                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                img_features = self.Proj_img(img_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            os.makedirs('./model', exist_ok=True)
                            torch.save(self.Enc_eeg.module.state_dict(), './model/' + model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_eeg.module.state_dict(), './model/' + model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), './model/' + model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n' % (
                e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))

        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model/' + model_idx + 'Proj_img_cls.pth'), strict=False)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)

        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))

        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():
    args = parser.parse_args()

    num_sub = args.num_sub
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    for i in range(num_sub):
        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i + 1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num + 1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + 'result.csv')


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))