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

def default_loader(id, root):
    
    # npy = np.load(os.path.join(root,'{}.npy').format(id))
    img = cv2.imread(os.path.join(root,'{}.jpg').format(id))
    img = np.transpose(img,[2,0,1]) # 通道在前

    mask = cv2.imread(os.path.join(root,'{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask, np.float32)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    
    return img, mask
    
class M4Fog_NC(data.Dataset):
    NUM_CLASS = 2  # NUM_CLASS = 2
    def __init__(self, root, split, get_name=False):
        self.get_name = get_name
        self.split = split
        root = os.path.join(root,split)
        self.root = root
        
        # 2.Get data list (P.S.Data and Labels are in the same folder)
        imagelist = filter(lambda x: x.find('png') == -1, os.listdir(root))
        trainlist = map(lambda x: x[:-4], imagelist)
        
        # 一定要改mean和std！！！！
        # trainlist_selected = [x for x in trainlist if 'H8_YB' in x]
        # trainlist_selected = [x for x in trainlist if 'FY4A_YB' in x]
        trainlist_selected = trainlist 
        self.ids = list(trainlist_selected)

        self.loader = default_loader
        self.tran = transforms.ToTensor()
        
        # 缩放到0-255，所有通道
        # H8 mean and std
        # self.mean = torch.FloatTensor([104.82, 111.69, 172.59]).view(-1, 1, 1)
        # self.std  = torch.FloatTensor([44.05, 46.40, 49.43]).view(-1, 1, 1)
        # FY4A mean and std
        # self.mean = torch.FloatTensor([60.15, 72.67, 173.90]).view(-1, 1, 1)
        # self.std  = torch.FloatTensor([52.50, 51.22, 50.94]).view(-1, 1, 1)
        # H8+FY4A mean and std
        # self.mean = torch.FloatTensor([82.84, 92.49, 173.24]).view(-1, 1, 1)
        # self.std  = torch.FloatTensor([53.79, 53.79, 50.18]).view(-1, 1, 1)
        
        # 1，2通道缩放，第3通道-200
        # H8 mean and std
        # self.mean = torch.FloatTensor([104.83, 111.68, 75.35]).view(-1, 1, 1)
        # self.std  = torch.FloatTensor([44.05, 46.42, 18.01]).view(-1, 1, 1)
        # FY4A mean and std
        # self.mean = torch.FloatTensor([60.17, 72.67, 74.84]).view(-1, 1, 1)
        # self.std  = torch.FloatTensor([52.50, 51.23, 17.39]).view(-1, 1, 1)
        # H8+FY4A mean and std
        self.mean = torch.FloatTensor([82.85, 92.49, 75.10]).view(-1, 1, 1)
        self.std  = torch.FloatTensor([53.79, 53.80, 17.72]).view(-1, 1, 1)

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root)
        
        # 处理img 归一化
        img = torch.Tensor(img)
        img = (img-self.mean)/self.std
        
        # 处理mask 0-1
        mask = torch.FloatTensor(mask)
        
        if self.get_name:
            return img, mask, id
        
        return img, mask
    
    def __len__(self):
        return len(self.ids)

# from collections import  Counter
if __name__ == "__main__":
    a = M4Fog("/nas2/data/users/xmq/cloud_tpami/AllTC_1009/", "train")
    pass
