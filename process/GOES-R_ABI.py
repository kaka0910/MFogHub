import numpy as np
from glob import glob
from satpy import Scene
from pyresample import create_area_def
from dask.diagnostics import ProgressBar
import cv2

def mk_npy(file_list_dir,save_dir,npyname):
    # Step 1: 选择要打开的文件列表
    file_list = glob(file_list_dir + '/*.nc')
    scn = Scene(reader='abi_l1b', filenames=file_list)
    print(1)
    # Step 2: 选择要打开的通道并打开文件
    my_channel = ['C01','C02', 'C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16']
    # 可以选择需要的通道，由于GOES数据是每个通道单独储存的，所以文件夹里有几个通道的数据，这里才能写那几个通道。全通道是16通道，也就是说，每次成功完成一个全员盘数据，至少需要16个文件。
    scn.load(my_channel)
    # Step 3: 定义采样区域经纬度
    # 墨西哥湾
    my_area = create_area_def('my_area', {'proj': 'merc',  'lon_0': -63.4},
                            width=1024, height=1024,
                            area_extent=[-69.8, 42, -57, 54.8], units='degrees')
    # Step 4: 将数据转为np格式
    data1 = np.asarray(scn.resample(my_area)['C01'])
    data2 = np.asarray(scn.resample(my_area)['C02'])
    data3 = np.asarray(scn.resample(my_area)['C03'])
    data4 = np.asarray(scn.resample(my_area)['C04'])
    data5 = np.asarray(scn.resample(my_area)['C05'])
    data6 = np.asarray(scn.resample(my_area)['C06'])
    data7 = np.asarray(scn.resample(my_area)['C07'])
    data8 = np.asarray(scn.resample(my_area)['C08'])
    data9 = np.asarray(scn.resample(my_area)['C09'])
    data10 = np.asarray(scn.resample(my_area)['C10'])
    data11 = np.asarray(scn.resample(my_area)['C11'])
    data12 = np.asarray(scn.resample(my_area)['C12'])
    data13 = np.asarray(scn.resample(my_area)['C13'])
    data14 = np.asarray(scn.resample(my_area)['C14'])
    data15 = np.asarray(scn.resample(my_area)['C15'])
    data16 = np.asarray(scn.resample(my_area)['C16'])

    npy = np.stack([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16])
    print(npy.shape)
    
    # Step 5: 保存结果
    npy_name = npyname
    save_dir = save_dir
    
    np.save(save_dir + '/'+ npy_name,npy)
    
    return npy

def make_TC(np_out):
    """
    0/1/2三个通道的合成图
    """
    b = np_out[0, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = np_out[1, :, :]
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = np_out[2, :, :]
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 255
    FY4_tc = cv2.merge([b,g,r])
    return FY4_tc

def make_NC(np_out):
    """
    0/1/2三个通道的合成图
    """
    b = np_out[1, :, :]
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255
    g = np_out[2, :, :]
    g = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    r = np_out[12, :, :]
    r = (r - np.min(r)) / (np.max(r) - np.min(r)) * 255
    FY4_nc = cv2.merge([b,g,r])
    return FY4_nc

def main():
    npy = mk_npy('./202100317','./result','test.npy')
    
    tc_pic = make_TC(npy)
    nc_pic = make_NC(npy)
    cv2.imwrite('./result'+'/'+'test_rgb.png', tc_pic)
    cv2.imwrite('./result'+'/'+'test_nc.png', nc_pic)
    
    
if __name__ == "__main__":
    main()