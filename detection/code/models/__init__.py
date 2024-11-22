from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
# from .fcn import *
from .psp import *
#from .encnet import *
from .deeplabv3 import *
from .vec_v1 import * 
from .vec_v2 import * 
from .vec_v3 import *
from .vec_v4 import * 
from .s2unet import *
from .unet import *
from .unet_motion import *
from .unet_motion_v1 import *
from .unet_motion_v2 import *
from .unet_motion_v3 import *
from .unet_temporal import *
from .unet_2output import *
from .unet_2output_v1 import *
from .unet_2out1 import *
from .unet_2branchout1 import *
from .temporal_unet import *
from .s2unet_v2 import *
from .s2unet_v3 import *
from .s2unet_v4 import *
from .s2unet_v5 import *
from .s2unet_v6 import *
from .pointshift import *
from .pointshift_a import *
from .pointshift_b import *
from .pointshift_c import *
from .fcn8s import *
from .new_fcn import *
from .Dlink_vit import *
from .Dlink_vit_v1 import *
from .Dlink_vit_v2 import *
from .models_mae import get_Mae
from .models_mae_ori import get_Mae_ori
from .fy4b.R2U_Net import *
from .fy4b.AttU_Net import *
from .fy4b.U_Net_Sig import *
from .fy4b.Aspp_unet import *
from .fy4b.Dlinkvit_h8 import *
from .fy4b.Dlinkvit_h8_2branchout1 import *
from .fy4b.Dlinkvit_h8_2output import *
from .fy4b.Dlinkvit_h8_2output_v0 import *
from .fy4b.Dlinkvit_h8_2output_v1 import *
from .fy4b.Dlinkvit_h8_motion import *
from .fy4b.vit_h8 import *
from .fy4b.BANet import *
from .fy4b.deeplabv3p import *
from .fy4b.scselink import *
from .fy4b.UNetFormer import *
from .fy4b.DCSwin import *
from .fy4b.ABCNet import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        # 'fcn': get_fcn,
        'fcn8s': get_fcn8s,
        'newfcn': get_fcn, 
        'psp': get_psp,
        # 'encnet': get_encnet,
        'deeplab': get_deeplab,
        'vec_v1': get_vec,
        'vec_v2': get_vec_v2,
        'vec_v3': get_vec_v3,
        'vec_v4': get_vec_v4,
        's2unet': get_s2unet,
        's2unet_v2': get_s2unet_v2,
        's2unet_v3': get_s2unet_v3,
        's2unet_v4': get_s2unet_v4,  # 4x down
        's2unet_v5': get_s2unet_v5,  # os 1 2 4 8 16
        's2unet_v6': get_s2unet_v6,  # point embed
        'pointshift': get_s2unet_v7,  # hierachical
        'pointshift_a': get_s2unet_v7a,  # without spatial shift
        'pointshift_b': get_s2unet_v7b,  # without spatial shift & split attention
        'pointshift_c': get_s2unet_v7c,  # without split attention
        'unet': get_unet,
        'unet_motion': get_unet_motion,
        'unet_motion_v1': get_unet_motion_v1,
        'unet_motion_v2': get_unet_motion_v2,
        'unet_motion_v3': get_unet_motion_v3,
        'unet_temporal': get_unet_temporal,
        'unet_2output': get_unet_2output,
        'unet_2out1': get_unet_2out1,
        'unet_2branchout1': get_unet_2branchout1,
        'unet_2output_v1': get_unet_2output_v1,
        'temporal_unet': get_temporal_unet,
        'dlink_vit': get_Dlink_vit,
        'dlink_vit_v1': get_Dlink_vit_v1,
        'mae':get_Mae,
        'mae_ori':get_Mae_ori,
        'dlink_vit_v2':get_Dlink_vit_v2,
        'unet_sig': get_unet_sigmoid,
        'r2u_net': get_R2U_Net,
        'attu_net': get_AttU_Net,
        'aspp_unet': get_Aspp_UNet,
        'dlinkvit_h8': get_dlinkvit_h8,
        'dlinkvit_h8_2branchout1': get_dlinkvit_h8_2branchout1,
        'dlinkvit_h8_2output': get_dlinkvit_h8_2output,
        'dlinkvit_h8_2output_v0': get_dlinkvit_h8_2output_v0,
        'dlinkvit_h8_2output_v1': get_dlinkvit_h8_2output_v1,
        'dlinkvit_h8_motion': get_dlinkvit_h8_motion,
        'vit_h8': get_vit_h8,
        'banet': get_BANet,
        'deeplabv3p': get_deeplabv3p,
        'scselink': get_scselink,
        'unetformer':get_UNetFormer,
        'dcswin':get_DCSwin,
        'abcnet': get_ABCNet

    }
    print(kwargs)
    return models[name.lower()](**kwargs)
