from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as V
from torchvision import transforms

class sobel(nn.Module):
    def __init__(self, image, n_channels,cal_motion = False, cal_sobel=False):
        super(sobel, self).__init__()
        self.img = image
        self.cal_motion = cal_motion
        self.cal_sobel = cal_sobel
        if cal_motion == True:
            self.n_channels = n_channels + 1
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)
        self.weight_hori = self.weight_const_hori
        self.gamma = nn.Parameter(torch.zeros(1))

        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)
        self.weight_vertical = self.weight_const_vertical

        # 边缘图的5个卷积
        self.conv2d_1_attention = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2d_2_attention = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2d_3_attention = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2d_4_attention = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv2d_5_attention = nn.Conv2d(8, 1, kernel_size=3, padding=1)

        # Attention Generation
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.conv2d_1_1 =  nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.conv2d_1_2 =  nn.Conv2d(self.n_channels, 1, kernel_size=3, padding=1)
        # self.AdaptiveAverPool_5 = nn.AdaptiveAvgPool2d((60,60))
        # self.AdaptiveAverPool_10 = nn.AdaptiveAvgPool2d((30, 30))
        # self.AdaptiveAverPool_15 = nn.AdaptiveAvgPool2d((20, 20))

        # Motion 

    def gray_pic(self):
        og_im = self.img
        unloader = transforms.ToPILImage()
        newimg = og_im[[0,1,5],:,:]
        newnp = newimg.detach().cpu().numpy()
        # print('newnp: ', type(newnp), newnp.shape)
        image = newimg.squeeze(0)  # remove the fake batch dimension
        # print(image.shape)
        image = unloader(image)
        image = image.convert('L')
        # print(type(image))
        # image.save('example0_gray.png')
        gray_np = np.array(image)
        # 灰度图tensor
        gray_tensor = V(torch.Tensor(gray_np).unsqueeze(0).unsqueeze(0).cuda(), volatile=False)
        # print('gray_tensor: ', gray_tensor.shape)
        return gray_tensor

    def get_edge(self):
        gray_tensor = self.gray_pic()
        


        weight_hori = V(self.weight_hori)
        # print('weight_hori: ', weight_hori.shape)
        weight_vertical = V(self.weight_vertical)
        
        x_hori = F.conv2d(gray_tensor, weight_hori, padding=1 )
        x_vertical = F.conv2d(gray_tensor, weight_vertical, padding=1)
       
        edge_detect = (torch.add(x_hori.pow(2),x_vertical.pow(2))).pow(0.5)
        # print('edge_detect: ', type(edge_detect))
        return edge_detect
    
    def get_motion(self, motion_np):
        motion_np = motion_np.squeeze(0)
        motion_layer = motion_np[0] # 第一个通道存的是motion motion_np.shape = (7,1024,1024)
        motion_layer[motion_layer<1] = 0
        motion_tensor = V(motion_layer.cuda())
        
        motion_tensor = ((motion_tensor - (motion_tensor.min())) / ((motion_tensor.max()) - (motion_tensor.min())))
        return motion_tensor

    def forward(self):
        og_im = self.img
        og_im = og_im.unsqueeze(0)

        if self.cal_sobel == True:
            edge_detect = self.get_edge()
            #convolution of edge image

            edge_detect_conved = self.conv2d_1_attention(edge_detect)
            edge_detect_conved = self.conv2d_2_attention(edge_detect_conved)
            edge_detect_conved = self.conv2d_3_attention(edge_detect_conved)
            edge_detect_conved = self.conv2d_4_attention(edge_detect_conved)
            edge_detect_conved = self.conv2d_5_attention(edge_detect_conved)
            
            #normalization of edge image  
            edge_detect = ((edge_detect - (edge_detect.min())) / ((edge_detect.max()) - (edge_detect.min())))
            # print('edge_detect: ', type(edge_detect), edge_detect.shape)

            # Attention Generation
            rgb_red = torch.cat((og_im, edge_detect * 255), 1)
            # print('Concat_Edge rgb_red', type(rgb_red), rgb_red.shape)

            # 拼接edge输出 和 原图  
            rgb_red_conved = torch.cat((rgb_red, edge_detect_conved), 1)

        #拼接上motion conv
        if self.cal_motion == True:
            # motion attention
            motion_np = og_im
            # print('Motion motion_np', type(motion_np), motion_np.shape)
            motion_tensor = self.get_motion(motion_np).unsqueeze(0).unsqueeze(0)
            print('Motion  motion_tensor: ', type(motion_tensor), motion_tensor.shape)
            motion_conv = self.conv2d_1_attention(motion_tensor)
            motion_conv = self.conv2d_2_attention(motion_conv)
            motion_conv = self.conv2d_3_attention(motion_conv)
            motion_conv = self.conv2d_4_attention(motion_conv)
            print('Motion  motion_tensor line123: ', type(motion_tensor), motion_conv.shape)
            motion_conv = self.conv2d_5_attention(motion_conv)
            motion_tensor = ((motion_tensor - (motion_tensor.min())) / ((motion_tensor.max()) - (motion_tensor.min())))
            if self.cal_sobel == True:
                rgb_red = torch.cat((rgb_red, motion_tensor*255), 1)  # 特征如果不明显可以成比例扩大
                rgb_red_conved = torch.cat((rgb_red_conved, motion_conv), 1)
            elif self.cal_sobel == False:
                rgb_red = torch.cat((og_im, motion_tensor * 255), 1)
                rgb_red_conved = torch.cat((rgb_red, motion_conv), 1)

        rgb_red_conved = self.conv2d_1_2(rgb_red_conved)
        # softmax_output = self.softmax(edge_detect)
        # print('softmax_output:',softmax_output)

        sigmoid_output = self.sigmoid(rgb_red_conved)
        rgb_red = self.gamma * (sigmoid_output * rgb_red) + (1 - self.gamma)*rgb_red
        return rgb_red


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class Aspp_att_unet(nn.Module):
    def __init__(self, n_classes, n_channels,deepsupervision=False, bilinear=True):
        super().__init__()

        print('att_map line198: ',n_channels)
        self.n_channels = n_channels + 1
        self.n_classes = n_classes
        self.deepsupervision = deepsupervision
        self.bilinear = bilinear

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.aspp = ASPP(nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(
            nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(
            nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(
            nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(
                nb_filter[0], self.n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)

    def forward(self, input):
        
        att_map = sobel(input[0],n_channels=self.n_channels,cal_motion=True,cal_sobel=False).cuda()
        newinput = att_map.forward().cuda() 
        
        x0_0 = self.conv0_0(newinput)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))

        # x4_0 = self.aspp(x4_0)## ASPP

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output