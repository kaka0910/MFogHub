import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class MeteoDataset(Dataset):

    def __init__(self, X, Y, use_augment=False, data_name='meteo'):
        super(MeteoDataset, self).__init__()
        self.X = X # channel is 2
        self.Y = Y
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()/255.0
        labels = torch.tensor(self.Y[index]).float()/255.0
        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=1,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    traindata = np.load(os.path.join(data_root, 'meteo/train.npy'))
    X_train = traindata[:,:pre_seq_length,:,:,:]
    Y_train = traindata[:,pre_seq_length:,:,:,:]

    testdata = np.load(os.path.join(data_root, 'meteo/test.npy'))
    X_test = testdata[:,:pre_seq_length,:,:,:]
    Y_test = testdata[:,pre_seq_length:,:,:,:]

    print('XMQ zhe ci yi ding neng pao dui:', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    assert X_train.shape[1] == pre_seq_length and Y_train.shape[1] == aft_seq_length
    
    train_set = MeteoDataset(X=X_train, Y=Y_train, use_augment=False)
    test_set = MeteoDataset(X=X_test, Y=Y_test, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
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

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='/nas2/data/users/xmq/Prediction/OpenSTL-OpenSTL-Lightning/data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
