import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
class focal_loss(nn.Module):
    def __init__(self):
        super(focal_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
    def focal_loss(self, loss):
        gamma = 2
        alpha = 0.25

        pt = 1 - torch.exp(-1 * loss)
        ptgamma = pt ** gamma
        result = ptgamma * pt *alpha
        return result
    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        c = self.focal_loss(a)
        # d = self.mse_loss(y_true,y_pred)
        return c

class bce_loss(nn.Module):
    def __init__(self):
        super(bce_loss, self).__init__()
        # self.bce_loss = nn.BCELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred.float(), y_true.float())
        return a

class ce_loss(nn.Module):
    def __init__(self):
        super(ce_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    def __call__(self, y_pred, y_true):
        # y_true = torch.unsqueeze(y_true,1)
        # print('y_pred: ', y_pred.shape)
        # print('y_true: ', y_true.shape)
        a = self.ce_loss(y_pred, y_true.long())
        return a