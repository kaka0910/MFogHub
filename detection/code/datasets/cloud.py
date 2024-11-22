import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch._C import dtype


class CloudSet(data.Dataset):
    NUM_CLASS = 10
    def __init__(self, root, split, mode=None, transform=None, fft=False,
                 target_transform=None, base_size=256, crop_size=256):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.fft=fft
        self.base_size = base_size
        self.crop_size = crop_size
        self.norm = torch.FloatTensor([211, 211, 211, 211, 208, 211, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]).view(1, 1, -1)
        # .npy list
        if self.split == 'train':
            self.file_list = glob(os.path.join(self.root, 'train', '*.npy'))
        elif self.split == 'val':
            self.file_list = glob(os.path.join(self.root, 'val', '*.npy'))
        else:
            raise RuntimeError("Error mode! Please use 'train' or 'val'.")
        if len(self.file_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def make_pred(self, x):
        return x

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        pair_dir = self.file_list[index]
        if self.mode == 'test':
            pair = np.load(pair_dir)
            img = pair[..., :16]
            mask = pair[..., -1]
            mask[mask == 255] = 10
            img = torch.from_numpy(np.array(img, dtype=np.float32)).float()
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).unsqueeze(0).long()
            img = img / self.norm
            if self.transform is not None:
                img = self.transform(img.permute(2, 0, 1))
            return (img, mask), os.path.basename(pair_dir)
        pair = np.load(pair_dir)
        img = pair[..., :16]
        mask = pair[..., -1]
        mask[mask == 255] = 10
        if self.mode == 'train':
            # Data Augmentationaw
            img, mask = rot_90(img=img, mask=mask)
            img, mask = horizontal_flip(img=img, mask=mask)
            img, mask = vertical_flip(img=img, mask=mask)
            # img = dropout(img=img)
        # Normalize
        img = torch.from_numpy(np.array(img, dtype=np.float32)).float()
        img = img / self.norm
        if self.transform is not None:
            img = self.transform(img.permute(2, 0, 1))
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.fft:
            img=torch.fft.fft2(img,norm="ortho").real.float()
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(mask).long()

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

def rot_90(img, mask, p=0.5):
    if random.random() > p: 
        img = np.rot90(img, k=-1)
        mask = np.rot90(mask, k=-1)
    return img.copy(), mask.copy()
