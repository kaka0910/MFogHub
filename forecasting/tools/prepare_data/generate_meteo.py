import numpy as np
import cv2
import os

def find_files_with_substring(directory, substring):
    matched_files = []
    # 使用os.walk遍历目录及其所有子目录
    for dirpath, dirnames, filenames in os.walk(directory):
        # 检查每个文件的文件名是否包含特定子串
        for filename in filenames:
            if substring in filename:
                # 如果包含，添加完整路径到结果列表中
                matched_files.append(os.path.join(dirpath, filename))
    
    return matched_files

# 最终目标是要搞到一个（B x T x C x H x W）的数组
png_path = '/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/meteo/meteo_png/'
# train 2020+2021
path_trains = [x[:-8] for x in os.listdir(png_path) if 'FS' in x and (x.split('-')[2][:4]=='2020' or x.split('-')[2][:4]=='2021')]
# path_trains = [x[:-8] for x in os.listdir(png_path) if ('-E-' in x or 'DW' in x or 'DE' in x or 'B' in x)]  # [B E DE DW]
# test 2022
# path_trains = [x[:-8] for x in os.listdir(png_path) if '-E-' in x and (x.split('-')[2][:4]=='2022')]
print(len(path_trains))  # train-4278, test-1734
path_trains = list(set(path_trains))
print(len(path_trains))  # train-252, test-102

train_all = []

for path_train in path_trains:
    # 指定要搜索的目录，调用函数并打印结果
    results = find_files_with_substring(png_path, path_train)
    results = sorted(results)
    if len(results)<17:
        print(path_train)
    # print(results)
    # 4 predict 4
    for i in range(8):
        frames = []
        for result in results[i:i+8]:
            img = cv2.imread(result)
            img = np.array(img).transpose(2,0,1)
            # print(img.shape)  # shape (3,256,256)
            frames.append(img)  
        frames_npy = np.stack(frames) # length=8
        # print(frames_npy.shape)  # shape (8,3,256,256)
    
        train_all.append(frames_npy)
        # print(len(train_all))

all_npy = np.stack(train_all) # shape (B,8,3,256,256)
print(all_npy.shape)
np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/meteo/supply/train_FS_20-21.npy', all_npy)

train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/meteo/supply/train_FS_20-21.npy')
train_data_test = train_data[0,:,:,:,:].transpose(0,2,3,1)
long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/meteo/supply/png/longimage_8_train_FS_20-21.png', long_image)