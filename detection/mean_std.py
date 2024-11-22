import numpy as np
import os
import cv2

# 假设 `datasets` 是一个生成器或列表，包含所有数据
# 模拟数据生成：例如，data = (np.random.random((14, 1024, 1024)) for _ in range(2000))
num_channels = 14
# num_channels = 14

def compute_stats(datasets):
    # mean = np.zeros((num_channels, 1024, 1024))
    # M2 = np.zeros((num_channels, 1024, 1024))
    mean = np.zeros((14, 1024, 1024))
    M2 = np.zeros((14, 1024, 1024))
    n = 0

    for data in datasets:
        n += 1
        delta = data - mean
        mean += delta / n
        delta2 = data - mean
        M2 += delta * delta2

    # 最终的均值和标准差
    variance = M2 / (n - 1) if n > 1 else np.zeros_like(M2)
    std_dev = np.sqrt(variance)
    
    # 压缩到通道级别的均值和标准差
    channel_means = np.mean(mean, axis=(1, 2))
    channel_stds = np.mean(std_dev, axis=(1, 2))

    return channel_means, channel_stds

# 使用模拟的数据集
# ids = [x.split('.')[0] for x in os.listdir('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/val_2021/') if x.split('.')[-1]=='npy' and  x[9:13]!='0000']
# print(len(ids))

# for id in ids:
#     data = np.load('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/GOES16_Data/val/'+id)
#     if np.sum(np.isnan(data))>0:
#         print(id)
#         ids.remove(id)
# print(len(ids))
        
# datasets = (np.load('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/FY4A_Data/train/'+id).transpose(2,0,1) for id in ids)
# datasets = (np.load('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/motion_tc/results/'+id+'/V_'+id+'.npy') for id in ids)
# datasets = (np.load('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/H8_Data/motion_nc/dist_1024/V_'+id+'.npy') for id in ids)
# channel_means, channel_stds = compute_stats(datasets)


# Alltc YB H8+FY4
# ids = [x for x in os.listdir('/nas2/data/users/xmq/cloud_tpami/AllNC_1025/train/') if x.split('.')[-1]=='jpg']
# # ids = [x for x in os.listdir('/nas2/data/users/xmq/cloud_tpami/AllNC_1025/train/') if x.split('.')[-1]=='jpg' and 'FY4A' in x]
# print(len(ids),ids[0])
# datasets = (np.transpose(cv2.imread('/nas2/data/users/xmq/cloud_tpami/AllNC_1025/train/'+id),[2,0,1]) for id in ids)
# channel_means, channel_stds = compute_stats(datasets)

# # 输出结果
# for i, (mean, std) in enumerate(zip(channel_means, channel_stds)):
#     print(f"Channel {i}: Mean = {mean:.5f}, Std = {std:.5f}")


# Alltc YB FY4 npy
# ids = [x for x in os.listdir('/nas2/data/users/xmq/cloud_tpami/AllNC_1025/train/') if x.split('.')[-1]=='npy']
ids = [x for x in os.listdir('/nas2/data/users/xmq/cloud_tpami/FY4A_npy_1025/train/') if x.split('.')[-1]=='npy' and 'FY4A' in x]
print(len(ids),ids[0])
datasets = (np.load('/nas2/data/users/xmq/cloud_tpami/FY4A_npy_1025/train/'+id) for id in ids)
channel_means, channel_stds = compute_stats(datasets)

# 输出结果
for i, (mean, std) in enumerate(zip(channel_means, channel_stds)):
    print(f"Channel {i}: Mean = {mean:.5f}, Std = {std:.5f}")

### sst读取max和min
# datasets = [np.load('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/SST_Data/sst_H8_52/'+id)['data'][(np.load('/groups/lmm2024/home/share/Sat_Pretrain_xmq/Data/SST_Data/sst_H8_52/'+id)['mask']==False)] for id in ids]
# print(np.array(datasets).max(),np.array(datasets).min())
'''
H8:
Channel 0: Mean = 123.89498, Std = 32.43055
Channel 1: Mean = 116.85378, Std = 35.77256
Channel 2: Mean = 104.82203, Std = 44.36134
Channel 3: Mean = 111.66169, Std = 46.73310
Channel 4: Mean = 91.35532, Std = 44.10594
Channel 5: Mean = 75.79264, Std = 45.24835
Channel 6: Mean = 87.90095, Std = 13.45280
Channel 7: Mean = 35.65307, Std = 6.09609
Channel 8: Mean = 44.84987, Std = 7.99963
Channel 9: Mean = 52.52286, Std = 9.45515
Channel 10: Mean = 73.86671, Std = 16.41026
Channel 11: Mean = 50.67246, Std = 9.00882
Channel 12: Mean = 75.92148, Std = 17.11562
Channel 13: Mean = 75.38618, Std = 17.57728
Channel 14: Mean = 73.20148, Std = 16.98814
Channel 15: Mean = 60.67884, Std = 12.56128

FY4A:
Channel 0: Mean = 0.22759, Std = 0.13692
Channel 1: Mean = 0.19084, Std = 0.14933
Channel 2: Mean = 0.22735, Std = 0.17427
Channel 3: Mean = 0.04194, Std = 0.02514
Channel 4: Mean = 0.16690, Std = 0.10447
Channel 5: Mean = 0.10811, Std = 0.07835
Channel 6: Mean = 291.49146, Std = 13.11718
Channel 7: Mean = 291.17612, Std = 13.42006
Channel 8: Mean = 237.19173, Std = 7.53162
Channel 9: Mean = 249.74316, Std = 9.64154
Channel 10: Mean = 274.50387, Std = 16.81833
Channel 11: Mean = 276.33183, Std = 17.86769
Channel 12: Mean = 274.82485, Std = 17.91964
Channel 13: Mean = 249.75440, Std = 10.33481

GOES16:
Channel 0: Mean = 22.34028, Std = 12.09479
Channel 1: Mean = 15.96896, Std = 12.10941
Channel 2: Mean = 19.19256, Std = 13.97513
Channel 3: Mean = 2.45230, Std = 4.30587
Channel 4: Mean = 12.45656, Std = 9.70764
Channel 5: Mean = 10.43512, Std = 7.96441
Channel 6: Mean = 280.94988, Std = 15.78521
Channel 7: Mean = 233.35981, Std = 8.08015
Channel 8: Mean = 241.85406, Std = 10.33527
Channel 9: Mean = 248.95759, Std = 12.58787
Channel 10: Mean = 267.05292, Std = 20.35541
Channel 11: Mean = 244.75735, Std = 15.01789
Channel 12: Mean = 269.12980, Std = 21.15982
Channel 13: Mean = 268.76019, Std = 21.63044
Channel 14: Mean = 267.07629, Std = 21.21018
Channel 15: Mean = 254.86040, Std = 16.18952

tc motion:
U: Channel 0: Mean = -0.04018, Std = 0.46250
V: Channel 0: Mean = 0.01095, Std = 0.53001
'''