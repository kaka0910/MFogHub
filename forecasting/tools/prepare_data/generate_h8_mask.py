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
# H8_17x的地址在'/nas2/data/users/xmq/seafog_H8_17x'
npy_path = '/nas2/data/users/xmq/seafog_H8_17x/'

# train file: train_all
# test file: val_2021
path_trains = [x.split('_')[0] for x in os.listdir(npy_path+'train_all/') if (x.split('.')[-1]=='npy')]
# path_trains = [x.split('_')[0] for x in os.listdir(npy_path+'val_2021/') if (x.split('.')[-1]=='npy')]
print('Original path number:', len(path_trains))
path_trains = list(set(path_trains))
print('Processed path number:', len(path_trains))  
# train 1088--》4+4 --》640
# test 680--> 4+4 --> 400

# train_tc, train_nc
train_tc = []
train_nc = []

for path_train in path_trains:
    # 指定要搜索的目录，调用函数并打印结果
    results = find_files_with_substring(npy_path, path_train)
    results = sorted(results)
    # print(results, len(results))
    # 4 predict 4
    if len(results)>=16:
        for i in range(10):
            frames_tc = []
            frames_nc = []
            # img 4+4
            for result in results[i:i+8]:
                npy = np.load(result)
                # tc:
                img_tc = npy[:,:,[0,1,2]]
                img_tc = cv2.resize(img_tc,(256,256))
                img_tc = np.array(img_tc).transpose(2,0,1)
                frames_tc.append(img_tc) 
                # nc:
                img_nc = npy[:,:,[2,3,13]]
                img_nc[:,:,2] = (img_nc[:,:,2]-img_nc[:,:,2].min())/(img_nc[:,:,2].max()-img_nc[:,:,2].min())*255
                img_nc = cv2.resize(img_nc,(256,256))
                img_nc = np.array(img_nc).transpose(2,0,1)
                frames_nc.append(img_nc)
            
            # label +4   
            for result in results[i:i+8]:          
                label_result = result[:-17]+'conn_'+result[-17:-4]+'_300_sp.png'
                label = cv2.imread(label_result)
                label = cv2.resize(label,(256,256))
                label = np.array(label).transpose(2,0,1)
                frames_tc.append(label) 
                frames_nc.append(label) 
            
            frames_tc_npy = np.stack(frames_tc) # length=12
            frames_nc_npy = np.stack(frames_nc) # length=12
            
            train_tc.append(frames_tc_npy)
            train_nc.append(frames_nc_npy)
    else:
        print(path_train)

# tc
tc_npy = np.stack(train_tc) # shape (B,8,3,256,256)
print(tc_npy.shape)
np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/train_tc_640.npy', tc_npy)

train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/train_tc_640.npy')
train_data_test = train_data[1,:,:,:,:].transpose(0,2,3,1)
long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/longimage_h8_train_tc_1.png', long_image)

# nc
nc_npy = np.stack(train_nc) # shape (B,8,3,256,256)
print(nc_npy.shape)
np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/train_nc_640.npy', nc_npy)

train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/train_nc_640.npy')
train_data_test = train_data[1,:,:,:,:].transpose(0,2,3,1)
long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/longimage_h8_train_nc_1.png', long_image)

# # tc
# tc_npy = np.stack(train_tc) # shape (B,8,3,256,256)
# print(tc_npy.shape)
# np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/val_tc_400.npy', tc_npy)

# train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/val_tc_400.npy')
# train_data_test = train_data[1,:,:,:,:].transpose(0,2,3,1)
# long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
# cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/longimage_h8_val_tc_1.png', long_image)

# # nc
# nc_npy = np.stack(train_nc) # shape (B,8,3,256,256)
# print(nc_npy.shape)
# np.save('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/val_nc_400.npy', nc_npy)

# train_data = np.load('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/val_nc_400.npy')
# train_data_test = train_data[1,:,:,:,:].transpose(0,2,3,1)
# long_image = cv2.hconcat([train_data_test[i] for i in range(train_data_test.shape[0])])
# cv2.imwrite('/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/H8_17x/longimage_h8_val_nc_1.png', long_image)