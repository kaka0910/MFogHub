import numpy as np
import cv2
import os

def find_files_with_substring(directory, substring):
    matched_files = []
    # 使用os.walk遍历目录及其所有子目录
    for dirpath, dirnames, filenames in os.walk(directory):
        # 检查每个文件的文件名是否包含特定子串
        for filename in filenames:
            # if substring in filename:
            if filename.startswith(substring):
                # 如果包含，添加完整路径到结果列表中
                matched_files.append(os.path.join(dirpath, filename))
    
    return matched_files

# 最终目标是要搞到一个（B x T x C x H x W）的数组
png_path = '/nas2/data/users/xmq/H8_256/'
# train <2021
# import pdb;
# pdb.set_trace()
path_trains = ['ES_'+x.split('_')[1] for x in os.listdir(png_path) if (x[:2]=='ES' and x.split('_')[1][:4]=='2021')]
# path_trains = [x[:-9] for x in os.listdir(png_path) if '2021' not in x]
# test 2021
# path_trains = [x.split('_')[0] for x in os.listdir(png_path) if (x[:2]!='ES' and x[:4]=='2021')]
# path_trains = [x[:-9] for x in os.listdir(png_path) if '2021' in x]
print(len(path_trains))  
path_trains = list(set(path_trains))
print(len(path_trains))  
# train area1 4137--》256--》1840
# test area1 1428-->84-->672
# train all 8076--》501--》3600
# test all 2856-->168-->1344


train_all = []

for path_train in path_trains:
    # 指定要搜索的目录，调用函数并打印结果
    results = find_files_with_substring(png_path, path_train)
    # import pdb;
    # pdb.set_trace()
    results = sorted(results)
    # print(results, len(results))
    # 4 predict 4
    if len(results)>=16:
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
    else:
        print(path_train)

all_npy = np.stack(train_all) # shape (B,8,3,256,256)
print(all_npy.shape)
np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_1w/test_area_es.npy', all_npy)

# train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_1w/test_area_es.npy')
# train_data_test = train_data[1000,:,:,:,:].transpose(0,2,3,1)
# long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
# cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/examples/work_dirs/longimage_h8_8test.png', long_image)