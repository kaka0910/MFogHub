import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from torchvision import transforms
import cv2
import numpy as np
import os
from PIL import Image
import random

def dropout(img, d=0.1, p=0.5):
    if random.random() > p:
        h, w, c = img.shape
        num_drop = int(h*d)
        y = np.random.randint(0, h, num_drop)
        x = np.random.randint(0, w, num_drop)
        for y_, x_ in zip(y, x):
            drop_c = np.random.randint(0, c, 1)
            img[y_, x_, drop_c] = 0
    return img

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        if len(image.shape)!=4:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() > u:
        times=int(np.random.random()*4)
        for i in range(times):
            image=np.rot90(image)
            #print("rotimg", image.shape)
            # print("typtofmask",type(mask))

            mask=np.rot90(mask)
            #print("rotmask", mask.shape)
    return image, mask

def rot_90(img, mask, p=0.5):
    if random.random() > p: 
        img = np.rot90(img, k=-1)
        mask = np.rot90(mask, k=-1)
    return img.copy(), mask.copy()

def horizontal_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return img.copy(), mask.copy()

def vertical_flip(img, mask, p=0.5):
    if random.random() > p: 
        img = img[::-1, :, :]
        mask = mask[::-1, :]
    return img.copy(), mask.copy()

def heb(image,id):
    # date=int(id.split('_')[-1])
    # month=int(id[4:6])
    # if month==12 or month==1 or month==2:
    #     if (date>=730 and date <= 930) or date==2330 or date<=100:
    #         pass
    #     else:
    #         return image
    # else:
    #     return image

    # channels=[0,1,2,3,4,5]
    channels = range(16)
    for i in channels:
        tmp=image[:,:,i]
        heb_image=cv2.equalizeHist(tmp)
        image[:,:,i]=heb_image
        # tmp=image[:,:,i]
        # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        # heb_image = clahe.apply(tmp)
        # image[:,:,i]=heb_image
    return image

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def default_loader(id, root, split):

    npy = np.load(os.path.join(root,'{}.npy').format(id)) # goes是float
    mask = cv2.imread(os.path.join(root,'{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    
    mask = np.array(mask, np.float32)/255.0
    mask[mask>=0.5] = 1
    mask[mask<0.5] = 0

    return npy, mask
    
class GOES16Set(data.Dataset):
    NUM_CLASS = 2  # NUM_CLASS = 2
    def __init__(self, root,split,get_name=False):
        self.get_name=get_name
        self.split=split
        root=os.path.join(root,split)
        
        # 2.Get data list (P.S.Data and Labels are in the same folder)
        # imagelist = filter(lambda x: (x.find('png')==-1) and (x.split('_')[1]=='J'), os.listdir(root))
        imagelist = filter(lambda x: (x.find('png')==-1), os.listdir(root))
        trainlist = map(lambda x: x[:-4], imagelist)
        self.ids = list(trainlist)
        # print(self.ids)
        self.loader = default_loader
        self.root = root
        self.tran = transforms.ToTensor()
        
        # self.norm = torch.FloatTensor([211, 211, 211, 211, 208, 211, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]).unsqueeze(1).unsqueeze(1)
        self.mean = torch.FloatTensor([22.34, 15.97, 19.19, 2.45, 12.46, 10.44, 280.95, 233.36,
                                       241.85, 248.96, 267.05, 244.76, 269.13, 268.76, 267.07, 254.86]).view(-1, 1, 1)
        self.std = torch.FloatTensor([12.09, 12.11, 13.98, 4.31, 9.71, 7.96, 15.79, 8.08,
                                      10.34, 12.59, 20.36, 15.02, 21.16, 21.63, 21.21, 16.19]).view(-1, 1, 1)


    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root, self.split)
        # import pdb;
        # pdb.set_trace()
        img = torch.FloatTensor(img)
        img = (img-self.mean)/self.std
        
        mask = torch.FloatTensor(mask)
        
        if self.get_name:
            return img,mask,id
        
        return img, mask
    
    def __len__(self):
        return len(self.ids)

# from collections import  Counter
if __name__ == "__main__":
    a = GOES16Set("/home/ssdk/mmengine/seafog_year_0412/","train")
    pass
'''
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
'''