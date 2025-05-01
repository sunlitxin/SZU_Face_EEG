# import os
# import numpy as np
# import torch
# #
# #存放 .npy 文件的路径
# eeg_data_dir = '/data0/xinyang/train_arcface/processed_data/eeg_datas'
# img_future_dir = '/data0/xinyang/train_arcface/processed_data/image_features_new'
#
# # 遍历所有 .npy 文件
# for filename in sorted(os.listdir(eeg_data_dir)):
#     if not filename.endswith('.npy'):
#         continue
#
#     file_path = os.path.join(eeg_data_dir, filename)
#     print(f"\n📂 正在读取文件：{filename}")
#
#     # 加载数据
#     data = np.load(file_path, allow_pickle=True).item()  # 必须加 .item()
#     eeg_data = data['eeg_data']  # shape: (N, 1, 126, 500)
#     labels = data['labels']      # shape: (N,)
#
#     # 转为 Tensor（可选）
#     eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.long)
#
#     # 打印维度和前几个值
#     print(f"EEG 数据形状: {eeg_data.shape}")  # 例如: torch.Size([40, 1, 126, 500])
#     print(f"标签形状: {labels.shape}")        # 例如: torch.Size([40])，每个样本对应一个标签
#     print(f"前30个标签: {labels[:30].tolist()}")
#     print(f"第1个样本的EEG片段 shape: {eeg_data[0].shape}")  # torch.Size([1, 126, 500])
#
# for filename in sorted(os.listdir(img_future_dir)):
#     if not filename.endswith('.npz'):
#         continue
#
#     file_path = os.path.join(img_future_dir, filename)
#     print(f"\n📂 正在读取文件：{filename}")
#
#     # 加载数据
#     img_data = np.load(file_path) # 必须加 .item()
#     features = img_data['features']  # 特征数据
#     img_labels = img_data['labels']  # 标签数据
#     # 转为 Tensor（可选）
#     features = torch.tensor(features, dtype=torch.float32)
#     img_labels = torch.tensor(img_labels, dtype=torch.long)
#
#     # 打印维度和前几个值
#     print(f"EEG 数据形状: {features.shape}")  # 例如: torch.Size([40, 1, 126, 500])
#     print(f"标签形状: {img_labels.shape}")        # 例如: torch.Size([40])，每个样本对应一个标签
#     print(f"前30个标签: {img_labels[:30].tolist()}")
#     print(f"第1个样本的EEG片段 shape: {features[0].shape}")  # torch.Size([1, 126, 500])
# #
# # 加载 npz 文件
# data = np.load(r"/data0/xinyang/train_arcface/processed_data/image_data_2_new.npz")
#
# # 取出里面的内容
# images = data['images']  # 图片数据，shape = (N,112,112,3)
# labels = data['labels']  # 标签数据，shape = (N,)
# names = data['names']    # 图片名字，shape = (N,)
#
# print(images.shape)  # 比如 (3200,112,112,3)
# print(labels.shape)  # 比如 (3200,)
# print(names.shape)   # 比如 (3200,)
# print(f"前30个标签: {labels[:30].tolist()}")

import os
import numpy as np
import torch
import re
import pandas as pd

# 文件路径
eeg_data_dir = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_eeg'
img_feature_dir = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_img_future'
output_excel_path = 'all_label_comparison.xlsx'

# 获取 EEG 和图像特征文件的编号对应映射
eeg_files = {re.search(r'(\d+)', f).group(1): f for f in os.listdir(eeg_data_dir) if f.endswith('.npz')}
img_files = {re.search(r'(\d+)', f).group(1): f for f in os.listdir(img_feature_dir) if f.endswith('.npz')}

# 获取交集编号
common_ids = sorted(set(eeg_files.keys()) & set(img_files.keys()), key=lambda x: int(x))

# 创建一个ExcelWriter对象（使用xlsxwriter）
with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    workbook = writer.book
    red_format = workbook.add_format({'bg_color': '#FF0000'})  # 红色高亮格式

    for idx in common_ids:
        eeg_file = eeg_files[idx]
        img_file = img_files[idx]

        eeg_path = os.path.join(eeg_data_dir, eeg_file)
        eeg_data_dict = np.load(eeg_path)
        labels = torch.tensor(eeg_data_dict['labels'], dtype=torch.long)

        img_path = os.path.join(img_feature_dir, img_file)
        img_data = np.load(img_path)
        img_labels = torch.tensor(img_data['labels'], dtype=torch.long)

        # 转成列表方便处理
        eeg_list = labels.tolist()
        img_list = img_labels.tolist()

        sheet_name = f'id_{idx}'
        worksheet = writer.book.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        # 写表头
        worksheet.write(0, 0, 'eeg_label')
        worksheet.write(0, 1, 'img_label')

        # 写入标签数据并对比，标红不一致项
        for row_num, (eeg_val, img_val) in enumerate(zip(eeg_list, img_list), start=1):  # 从Excel第2行开始（0是表头）
            if eeg_val != img_val:
                worksheet.write(row_num, 0, eeg_val, red_format)
                worksheet.write(row_num, 1, img_val, red_format)
            else:
                worksheet.write(row_num, 0, eeg_val)
                worksheet.write(row_num, 1, img_val)

print(f"\n✅ 标签对比写入并完成高亮：{output_excel_path}")



