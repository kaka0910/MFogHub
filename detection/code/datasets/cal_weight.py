import os

import numpy as np
import torch

# from mapillary import Mapillary
# from pascal_voc import VOCSegmentation
from pascal_aug import VOCAugSegmentation


def calc_median_frequency(classes, present_num):
    """
    Class balancing by median frequency balancing method.
    Reference: https://arxiv.org/pdf/1411.4734.pdf
       'a = median_freq / freq(c) where freq(c) is the number of pixels
        of class c divided by the total number of pixels in images where
        c is present, and median_freq is the median of these frequencies.'
    """
    class_freq = classes / present_num
    median_freq = np.median(class_freq)
    return median_freq / class_freq

def calc_log_frequency(classes, value=1.02):
    """Class balancing by ERFNet method.
       prob = each_sum_pixel / each_sum_pixel.max()
       a = 1 / (log(1.02 + prob)).
    """
    class_freq = classes / classes.sum()  # ERFNet is max, but ERFNet is sum
    return 1 / np.log(value + class_freq)

if __name__ == '__main__':
    from tqdm import tqdm
    dst = VOCAugSegmentation(root='./dataset')
    nclass = 21
    classes, present_num = ([0 for i in range(nclass)] for i in range(2))
    for label in tqdm(dst):
        for nc in range(nclass):
            num_pixel = (label == nc).sum()
            if num_pixel:
                classes[nc] += num_pixel
                present_num[nc] += 1
        # if (idx+1) % 1000 == 0:
        #     print(classes)
    if 0 in classes:
        raise Exception("Some classes are not found")

    classes = np.array(classes, dtype="f")
    presetn_num = np.array(classes, dtype="f")

    class_freq = torch.from_numpy(classes).float()
    weights = 1 / torch.log1p(class_freq)
    weights = 21 * weights / torch.sum(weights)
    print(weights)

    # median_freq = calc_median_frequency(classes, present_num)
    # print(median_freq)
    # median_freq = 65 * median_freq / torch.sum(median_freq)
    # print(median_freq)

