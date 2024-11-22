import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F


def get_temporal_unet(dataset='cloud', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = tem_unet(n_classes=2, n_channels=16, **kwargs)
    return model

class tem_unet(nn.Module):

    def __init__(self, n_classes, n_channels,bilinear=True,criterion_seg=None):
        super(tem_unet, self).__init__()

        self.in_channels = n_channels
        self.nclass = n_classes
        self.bilinear = bilinear

        # All layers which have weights are created and initialized in init.
        # parameterless modules are used in functional style F. in forward
        # (object version of parameterless modules can be created with nn.init too )
        # https://pytorch.org/docs/master/nn.html#conv2d
        # in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        self.conv1_0 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)

        # https://pytorch.org/docs/master/nn.html#batchnorm2d
        # num_features/channels, eps, momentum, affine, track_running_stats
        self.conv2_0 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # https://pytorch.org/docs/master/nn.html#maxpool2d
        # kernel_size, stride, padding, dilation, return_indices, ceil_mode
        self.maxPool1_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_0 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.conv4_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.maxPool2_0 = nn.MaxPool2d(2, stride=2)

        self.conv5_0 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.conv6_0 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.maxPool3_0 = nn.MaxPool2d(2, stride=2)

        self.conv7_0 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.conv8_0 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.maxPool4_0 = nn.MaxPool2d(2, stride=2)

        # https://pytorch.org/docs/master/nn.html#conv2d
        # in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        self.conv1_1 = nn.Conv2d(n_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        # https://pytorch.org/docs/master/nn.html#batchnorm2d
        # num_features/channels, eps, momentum, affine, track_running_stats
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # https://pytorch.org/docs/master/nn.html#maxpool2d
        # kernel_size, stride, padding, dilation, return_indices, ceil_mode
        self.maxPool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.maxPool2_1 = nn.MaxPool2d(2, stride=2)

        self.conv5_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.maxPool3_1 = nn.MaxPool2d(2, stride=2)

        self.conv7_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.maxPool4_1 = nn.MaxPool2d(2, stride=2)

        self.conv3D_01 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)


        # https://pytorch.org/docs/master/nn.html#convtranspose2d
        # in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation
        self.upsampconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv13 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv15 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv17 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.conv19 = nn.Conv2d(64, n_classes, 1, stride=1)
        self.conv19 = nn.Conv2d(64, n_classes, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.criterion_seg = criterion_seg
        

        # # weights can be initialized here:
        # for idx, m in enumerate(self.modules()):
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         # force float division, therefore use 2.0
        #         # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        #         # https://arxiv.org/abs/1502.01852
        #         # a rectifying linear unit is zero for half of its input,
        #         # so you need to double the size of weight variance to keep the signals variance constant.
        #         # xavier would be: scalefactor * sqrt(2/ (inchannels + outchannels )
        #         std = math.sqrt(2.0 / (m.kernel_size[0] * m.kernel_size[0] * m.in_channels))
        #         nn.init.normal_(m.weight, std=std)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, z, gt=None, padding=False):

        # The output image of the net decreases in size because of the multiple 3x3 convolutions
        # 512 input is reduced to 324 output
        # if padding mode is True, the row and column border pixels are mirrored on the side
        # therefore input image size = output image size

        padmode = 'reflect'
        if padding:
            pad = (1, 1, 1, 1)
        else:
            pad = (0, 0, 0, 0)

        batchsize = z.shape[0]
        x = z[:,:16,:,:].clone()
        y = z[:,16:,:,:].clone()

        # https://pytorch.org/docs/master/nn.html#id26 F.relu
        # input, inplace
        # https://pytorch.org/docs/master/nn.html#torch.nn.functional.pad
        # input, pad , mode
        # part 1
        x = F.relu(self.conv1_0(F.pad(x, pad, padmode)))
        x = F.relu(self.conv2_0(F.pad(x, pad, padmode)))
        x = self.maxPool1_0(x)
        # save result for combination in later layer

        x = F.relu(self.conv3_0(F.pad(x, pad, padmode)))
        x = F.relu(self.conv4_0(F.pad(x, pad, padmode)))
        x = self.maxPool2_0(x)

        x = F.relu(self.conv5_0(F.pad(x, pad, padmode)))
        x = F.relu(self.conv6_0(F.pad(x, pad, padmode)))
        x = self.maxPool3_0(x)

        x = F.relu(self.conv7_0(F.pad(x, pad, padmode)))
        x = F.relu(self.conv8_0(F.pad(x, pad, padmode)))
        x = F.dropout(x, 0.5)
        x = self.maxPool4_0(x)

        # input, probability of an element to be zero-ed
        # https://pytorch.org/docs/master/nn.html#dropout


        # part 2

        y = F.relu(self.conv1_1(F.pad(y, pad, padmode)))
        y = F.relu(self.conv2_1(F.pad(y, pad, padmode)))
        y_copy1_2 = y
        y = self.maxPool1_1(y)

        y = F.relu(self.conv3_1(F.pad(y, pad, padmode)))
        y = F.relu(self.conv4_1(F.pad(y, pad, padmode)))
        y_copy3_4 = y
        y = self.maxPool2_1(y)

        y = F.relu(self.conv5_1(F.pad(y, pad, padmode)))
        y = F.relu(self.conv6_1(F.pad(y, pad, padmode)))
        y_copy5_6 = y
        y = self.maxPool3_1(y)

        y = F.relu(self.conv7_1(F.pad(y, pad, padmode)))
        y = F.relu(self.conv8_1(F.pad(y, pad, padmode)))
        y_copy7_8 = y
        y = F.dropout(y, 0.5)
        y = self.maxPool4_1(y)

        # import pdb;
        # pdb.set_trace()
        mix = torch.cat((x, y), 1)
        mix = F.relu(self.conv3D_01(F.pad(mix, pad, padmode)))
        

        mix = F.relu(self.conv10(F.pad(mix, pad, padmode)))

        x_out = F.dropout(mix, 0.5)
        
        x_out = F.relu(self.upsampconv1(x_out))
        x_out = self.crop_and_concat(x_out, y_copy7_8)
        
        x_out = F.relu(self.conv11(F.pad(x_out, pad, padmode)))
        x_out = F.relu(self.conv12(F.pad(x_out, pad, padmode)))

        x_out = F.relu(self.upsampconv2(x_out))
        x_out = self.crop_and_concat(x_out, y_copy5_6)

        x_out = F.relu(self.conv13(F.pad(x_out, pad, padmode)))
        x_out = F.relu(self.conv14(F.pad(x_out, pad, padmode)))

        x_out = F.relu(self.upsampconv3(x_out))
        x_out = self.crop_and_concat(x_out, y_copy3_4)

        x_out = F.relu(self.conv15(F.pad(x_out, pad, padmode)))
        x_out = F.relu(self.conv16(F.pad(x_out, pad, padmode)))

        x_out = F.relu(self.upsampconv4(x_out))
        x_out = self.crop_and_concat(x_out, y_copy1_2)

        x_out = F.relu(self.conv17(F.pad(x_out, pad, padmode)))
        x_out = F.relu(self.conv18(F.pad(x_out, pad, padmode)))
        
        # print('225: ',x_out.shape)
        
        # x_out = F.relu(self.conv17(x_out))
        # x_out = F.relu(self.conv18(x_out))
        x_out = self.conv19(x_out)
        # print('228: ',x_out.shape)

        # x_out = self.sigmoid(x_out)
        if self.training:
            loss = self.criterion_seg(x_out, gt.long())
            return x_out.max(1)[1].detach(), loss
        else:
            return x_out.detach()

        # return x_out

    # when no padding is used, the upsampled image gets smaller
    # to copy a bigger image to the corresponding layer, it needs to be cropped
    def crop_and_concat(self, upsampled, bypass):
        # Python 2 / Integer division ( if int intputs ), // integer division
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        d = c
        # checks if bypass.size() is odd
        # if input image is 512, at   x = self.crop_and_concat(x, x_copy5_6)
        # x_copy5_6 is 121*121
        # therefore cut one more row and column
        if (bypass.size()[2] & 1) == 1:
            d = c + 1
            # padleft padright padtop padbottom
        bypass = F.pad(bypass, (-c, -d, -c, -d))
        return torch.cat((bypass, upsampled), 1)
    
    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union
