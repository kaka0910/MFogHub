import os
import cv2
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torchvision
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from openstl.datasets.utils import create_loader


class Cloudtc(Dataset):
    """ Himawari-8 Cloud Dataset (True Color)

    Args:
        data_root (str): Path to the dataset.
        split (list): The years of the dataset.
        image_size (int: The target resolution of Human3.6M images.
        pre_seq_length (int): The input sequence length.
        aft_seq_length (int): The output sequence length for prediction.
    """

    def __init__(self, data_root, split, image_size=256,
                 pre_seq_length=4, aft_seq_length=8, data_name='cloud_tc'):
        super(Cloudtc, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.input_shape = (self.seq_length, self.image_size, self.image_size, 3)

        self.time_list = pd.timedelta_range(start='00:00:00', end='08:00:00', freq='30min')
        self.time_list = [str(time).split(' ')[2].replace(':', '')[:4] for time in self.time_list]

        self.date_list = open(os.path.join(self.data_root, self.split + '.txt')).read().split()
        self.num_seq_per_day = len(self.time_list) - self.seq_length + 1
        self.num_seq = len(self.date_list) * self.num_seq_per_day

        self.mean = None
        self.std = None
        self.data_name = data_name

    def _to_tensor(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = torch.from_numpy(imgs[i].copy()).float()
        return imgs

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        date_idx = idx // self.num_seq_per_day
        time_idx = idx % self.num_seq_per_day

        img_seq = []
        for i in range(self.seq_length):
            img_path = os.path.join(self.data_root, 'images', self.date_list[date_idx], self.time_list[time_idx + i] + '.png')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img_seq.append(img)

        img_seq = self._to_tensor(img_seq)

        # transform
        img_seq = torch.stack(img_seq, 0).permute(0, 3, 1, 2) / 255  # min-max to [0, 1]
        data = img_seq[:self.pre_seq_length, ...]
        labels = img_seq[self.pre_seq_length:, ...]

        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=1, data_name='cloud_tc',
              pre_seq_length=4, aft_seq_length=4, in_shape=[4, 3, 256, 256],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    if data_name == 'cloud_tc':
        data_root = os.path.join(data_root, 'H8_tc')
    else:
        print(f"Data name {data_name} is not supported")
    image_size = in_shape[-1] if in_shape is not None else 256

    train_set = Cloudtc(data_root, 'train', image_size,
                        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    val_set = Cloudtc(data_root, 'val', image_size,
                        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    test_set = Cloudtc(data_root, 'test', image_size,
                        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_val = create_loader(val_set,
                                   batch_size=val_batch_size,
                                   shuffle=False, is_training=False,
                                   pin_memory=True, drop_last=drop_last,
                                   num_workers=num_workers,
                                   distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_val, dataloader_test


if __name__ == '__main__':
    dataset = Cloudtc("/data/mfb/dataset/H8_tc", 'test')
    
    import cv2
    print(dataset[0][0].shape, dataset[0][1].shape)

