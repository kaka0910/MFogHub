from .base import *
from .coco import COCOSegmentation
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CitySegmentation
from .lip import LIP
from .cloud import CloudSet
from .fog import FogSet
from .h8 import H8Set
from .h8_sst import H8Set_SST
from .h8_all import H8Set_ALL
from .h8_temporal import H8Set_Temporal
from .h8_2output import H8Set_2output
from .h8_2out1 import H8Set_2out1
from .h8_motion import H8Set_Motion
from .h8_motion_v2 import H8Set_Motion_v2
from .fy4a import FY4ASet
from .fy4a_sst import FY4ASet_SST
from .fy4a_2out1 import FY4ASet_2out1
from .fy4a_2output import FY4ASet_2output
from .fy4a_motion import FY4ASet_motion
from .goes16 import GOES16Set
from .h8_3 import H8Set_3
from .h8sig import H8Set_sig
from .m4fog import M4Fog
from .m4fog_nc import M4Fog_NC
datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'citys': CitySegmentation,
    'lip' : LIP,
    'cloud': CloudSet,
    'fog': FogSet,
    'h8': H8Set,
    'h8_sst':H8Set_SST,
    'h8_all':H8Set_ALL,
    'h8_temporal':H8Set_Temporal,
    'h8_2output':H8Set_2output,
    'h8_2out1':H8Set_2out1,
    'h8_motion':H8Set_Motion,
    'h8_motion_v2':H8Set_Motion_v2,
    'fy4a': FY4ASet,
    'fy4a_sst': FY4ASet_SST,
    'fy4a_2out1': FY4ASet_2out1,
    'fy4a_2output': FY4ASet_2output,
    'fy4a_motion': FY4ASet_motion,
    'goes16': GOES16Set,
    'h8_3': H8Set_3,
    'h8_sig': H8Set_sig,
    'm4fog': M4Fog,
    'm4fog_nc': M4Fog_NC
}

def get_segmentation_dataset(name,**kwargs):
    return datasets[name.lower()](**kwargs)
