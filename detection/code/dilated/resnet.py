# """Dilated ResNet"""
# import math
# import torch
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo




# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'BasicBlock', 'Bottleneck']

# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
# }

# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# class BasicBlock(nn.Module):
#     """ResNet BasicBlock
#     """
#     expansion = 1
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
#                  norm_layer=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
#                                padding=dilation, dilation=dilation, bias=False)
#         self.bn1 = norm_layer(planes)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
#                                padding=previous_dilation, dilation=previous_dilation, bias=False)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     """ResNet Bottleneck
#     """
#     # pylint: disable=unused-argument
#     expansion = 4
#     def __init__(self, inplanes, planes, stride=1, dilation=1,
#                  downsample=None, previous_dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = norm_layer(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=stride,
#             padding=dilation, dilation=dilation, bias=False)
#         self.bn2 = norm_layer(planes)
#         self.conv3 = nn.Conv2d(
#             planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = norm_layer(planes * 4)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.downsample = downsample
#         self.dilation = dilation
#         self.stride = stride

#     def _sum_each(self, x, y):
#         assert(len(x) == len(y))
#         z = []
#         for i in range(len(x)):
#             z.append(x[i]+y[i])
#         return z

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):
#     """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

#     Parameters
#     ----------
#     block : Block
#         Class for the residual block. Options are BasicBlockV1, BottleneckV1.
#     layers : list of int
#         Numbers of layers in each block
#     classes : int, default 1000
#         Number of classification classes.
#     dilated : bool, default False
#         Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
#         typically used in Semantic Segmentation.
#     norm_layer : object
#         Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
#         for Synchronized Cross-GPU BachNormalization).

#     Reference:

#         - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

#         - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
#     """
#     # pylint: disable=unused-variable
#     def __init__(self, block, layers, num_classes=1000, dilated=True,
#                  deep_base=True, norm_layer=nn.BatchNorm2d, output_size=8):
#         self.inplanes = 128 if deep_base else 64
#         super(ResNet, self).__init__()
#         if deep_base:
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
#                 norm_layer(64),
#                 # nn.ReLU(inplace=True),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                 norm_layer(64),
#                 # nn.ReLU(inplace=True),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             )
#         else:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                    bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         # self.relu = nn.ReLU(inplace=True)
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)

#         dilation_rate = 2
#         if dilated and output_size <= 8:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
#                                            dilation=dilation_rate, norm_layer=norm_layer)
#             dilation_rate *= 2
#         else:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                            norm_layer=norm_layer)

#         if dilated and output_size <= 16:
#             self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
#                                            dilation=dilation_rate, norm_layer=norm_layer)
#         else:
#             self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                            norm_layer=norm_layer)

#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, norm_layer):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         if dilation == 1 or dilation == 2:
#             layers.append(block(self.inplanes, planes, stride, dilation=1,
#                                 downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
#         elif dilation == 4:
#             layers.append(block(self.inplanes, planes, stride, dilation=2,
#                                 downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
#         else:
#             raise RuntimeError("=> unknown dilation size: {}".format(dilation))

#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
#                                 norm_layer=norm_layer))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x


# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return model


# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model

# def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
#     """Constructs a ResNest-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         from ..models.model_store import get_model_file
#         model.load_state_dict(torch.load(
#             get_model_file('resnet50', root=root)), strict=False)
#     return model

# def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
#     """Constructs a ResNet-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
#     return model


# def resnet101(pretrained=False, root='~/.encoding/models', **kwargs):
#     """Constructs a ResNet-101 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         from ..models.model_store import get_model_file
#         model.load_state_dict(torch.load(
#             get_model_file('resnet101', root=root)), strict=False)
#     return model


# def resnet152(pretrained=False, root='~/.encoding/models', **kwargs):
#     """Constructs a ResNet-152 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         from ..models.model_store import get_model_file
#         model.load_state_dict(torch.load(
#             get_model_file('resnet152', root=root)), strict=False)
#     return model

'''
Function:
    Implementation of ResNet
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from ..nn import BatchNorm2d

'''model urls'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet18stem': 'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    'resnet50stem': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101stem': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}


'''BasicBlock'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    '''forward'''
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


'''Bottleneck'''
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    '''forward'''
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


'''ResNet'''
class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(self, in_channels=3, base_channels=64, stem_channels=64, depth=101, outstride=8, 
                 contract_dilation=True, use_stem=True, out_indices=(0, 1, 2, 3), use_avg_for_downsample=False, 
                 norm_cfg=None, act_cfg=None, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = stem_channels
        # set out_indices
        self.out_indices = out_indices
        # parse depth settings
        assert depth in self.arch_settings, 'unsupport depth %s...' % depth
        block, num_blocks_list = self.arch_settings[depth]
        # parse outstride
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s...' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]
        # whether replace the 7x7 conv in the input stem with three 3x3 convs
        self.use_stem = use_stem
        if use_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
                BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels // 2, stem_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # make layers
        self.layer1 = self.makelayer(
            block=block, 
            inplanes=stem_channels, 
            planes=base_channels, 
            num_blocks=num_blocks_list[0], 
            stride=stride_list[0], 
            dilation=dilation_list[0], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.layer2 = self.makelayer(
            block=block, 
            inplanes=base_channels * 4 if depth >= 50 else base_channels, 
            planes=base_channels * 2, 
            num_blocks=num_blocks_list[1], 
            stride=stride_list[1], 
            dilation=dilation_list[1], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.layer3 = self.makelayer(
            block=block, 
            inplanes=base_channels * 8 if depth >= 50 else base_channels * 2, 
            planes=base_channels * 4, 
            num_blocks=num_blocks_list[2], 
            stride=stride_list[2], 
            dilation=dilation_list[2], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.layer4 = self.makelayer(
            block=block, 
            inplanes=base_channels * 16 if depth >= 50 else base_channels * 4,  
            planes=base_channels * 8, 
            num_blocks=num_blocks_list[3], 
            stride=stride_list[3], 
            dilation=dilation_list[3], 
            contract_dilation=contract_dilation, 
            use_avg_for_downsample=use_avg_for_downsample,
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
    '''make res layer'''
    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, contract_dilation=True, use_avg_for_downsample=False, norm_cfg=None, act_cfg=None, **kwargs):
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1: dilations[0] = dilation // 2
        if stride != 1 or inplanes != planes * block.expansion:
            if use_avg_for_downsample:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                    BatchNorm2d(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                    BatchNorm2d(planes * block.expansion)
                )
        layers = []
        layers.append(block(inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks): layers.append(block(planes * block.expansion, planes, stride=1, dilation=dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs))
        return nn.Sequential(*layers)
    '''forward'''
    def forward(self, x):
        if self.use_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        outs = []
        for i, feats in enumerate([x1, x2, x3, x4]):
            if i in self.out_indices: outs.append(feats)
        return tuple(outs)


'''build resnet'''
def BuildResNet(resnet_type, **kwargs):
    # assert whether support
    supported_resnets = {
        'resnet18': {'depth': 18},
        'resnet34': {'depth': 34},
        'resnet50': {'depth': 50},
        'resnet101': {'depth': 101},
        'resnet152': {'depth': 152},
    }
    assert resnet_type in supported_resnets, 'unsupport the resnet_type %s...' % resnet_type
    # parse args
    default_args = {
        'outstride': 8,
        'use_stem': False,
        'norm_cfg': None,
        'in_channels': 3,
        'pretrained': True,
        'base_channels': 64,
        'stem_channels': 64,
        'contract_dilation': True,
        'out_indices': (0, 1, 2, 3),
        'pretrained_model_path': '',
        'use_avg_for_downsample': False,
        'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    }
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain args for instanced resnet
    resnet_args = supported_resnets[resnet_type]
    resnet_args.update(default_args)
    # obtain the instanced resnet
    model = ResNet(**resnet_args)
    # load weights of pretrained model
    if default_args['use_stem']: resnet_type = resnet_type + 'stem'
    if default_args['pretrained'] and os.path.exists(default_args['pretrained_model_path']):
        checkpoint = torch.load(default_args['pretrained_model_path'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif default_args['pretrained']:
        checkpoint = model_zoo.load_url(model_urls[resnet_type])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model

def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return BuildResNet('resnet50', pretrained=pretrained)