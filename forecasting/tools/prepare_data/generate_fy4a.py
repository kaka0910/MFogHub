import numpy as np
import cv2
import os
from datetime import datetime, timedelta

# def find_files_with_substring(directory, substring):
#     matched_files = []
#     # 使用os.walk遍历目录及其所有子目录
#     for dirpath, dirnames, filenames in os.walk(directory):
#         # 检查每个文件的文件名是否包含特定子串
#         for filename in filenames:
#             if substring in filename:
#                 # 如果包含，添加完整路径到结果列表中
#                 matched_files.append(os.path.join(dirpath, filename))
    
#     return matched_files

# def generate_time_strings(start_time_str, n):
#     # 将初始时间字符串转换为 datetime 对象
#     start_time = datetime.strptime(start_time_str, "%H%M")
    
#     # 创建一个列表来存储结果
#     time_strings = []
    
#     # 生成 n 个半小时增加的时间字符串
#     for i in range(n):
#         # 计算新的时间
#         new_time = start_time + timedelta(minutes=30 * i)
#         # 将新的时间转换回字符串并添加到列表中
#         time_strings.append(new_time.strftime("%H%M"))
    
#     return time_strings

# # 最终目标是要搞到一个（B x T x C x H x W）的数组
# png_path = '/nas2/data/private/Atmosphere/FY4A_seafog/png/generate_png/'
# # train 2020+19+18
# path_trains = [x[:-8] for x in os.listdir(png_path) if int(x.split('_')[1][:4])<2022]
# # test 2022+23+24+21 (1053, 8, 3, 256, 256)
# # path_trains = [x[:-8] for x in os.listdir(png_path) if int(x.split('_')[1][:4])>=2021]
# print(len(path_trains))  # train-4278, test-1734
# path_trains = list(set(path_trains))
# print(len(path_trains))  

# train_all = []

# for path_train in path_trains:
#     if 'HB' in path_train:
#         # print(path_train)
#         # 指定要搜索的目录，调用函数并打印结果
#         results = find_files_with_substring(png_path, path_train)
#         results = sorted(results)

#         if len(results)<9:
#             print(path_train)
#         # print(results)
#         # 4 predict 4
#         for result in results:
#             frames = []
            
#             date_begin = result.split('.')[-2][-4:]
#             time_strings = generate_time_strings(date_begin, 8)
            
#             for string in time_strings:
#                 img_path = result.split('.')[-2][:-5]+'_'+string+'.png'
#                 if os.path.exists(img_path):
#                     img = cv2.imread(img_path)
#                     img = np.array(img).transpose(2,0,1)
#                     # print(img.shape)  # shape (3,256,256)
#                     frames.append(img)  
#                     frames_npy = np.stack(frames) # length=8
#                     # print(frames_npy.shape)  # shape (8,3,256,256)
            
#             if frames_npy.shape[0]==8:
#                 train_all.append(frames_npy)
#                 # print(len(train_all))

# all_npy = np.stack(train_all) # shape (B,8,3,256,256)
# print(all_npy.shape)
# np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/fy4a/train.npy', all_npy)
# train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/fy4a/train.npy')
# train_data_test = train_data[1,:,:,:,:].transpose(0,2,3,1)
# long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
# cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/examples/work_dirs/longimage_8train_fy4a.png', long_image)

# all_npy = np.stack(train_all) # shape (B,8,3,256,256)
# print(all_npy.shape)
# np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/fy4a_new/train_hb.npy', all_npy)

train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/fy4a_new/train.npy')
indexes = [0,50,100,150,200,300,500]
for index in indexes:
    if index>train_data.shape[0]:
        pass
    
    train_data_test = train_data[index,:,:,:,:].transpose(0,2,3,1)
    long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
    cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/fy4a_new/longimage_8train_fy4a_'+str(index)+'.png', long_image)