import os
import h5py
import numpy as np
from numpy import deg2rad, rad2deg, arctan, arcsin, tan, sqrt, cos, sin
from numpy import *
import cv2
from datetime import *
import time
import math
from netCDF4 import Dataset
from skimage import exposure

ea = 6378.137  # 地球的半长轴[km]
eb = 6356.7523  # 地球的短半轴[km]
h = 42164  # 地心到卫星质心的距离[km]
λD = deg2rad(133.0)  # 卫星星下点所在经度

# 列偏移
COFF = {"0500M": 10991.5,
        "1000M": 5495.5,
        "2000M": 2747.5,
        "4000M": 1373.5}
# 列比例因子
CFAC = {"0500M": 81865099,
        "1000M": 40932549,
        "2000M": 20466274,
        "4000M": 10233137}
        
LOFF = COFF  # 行偏移
LFAC = CFAC  # 行比例因子
def latlon2linecolumn(lat, lon, resolution='4000M'):
    """
    (lat, lon) → (line, column)
    resolution：文件名中的分辨率{'0500M', '1000M', '2000M', '4000M'}
    line, column不是整数
    """
    # Step1.检查地理经纬度
    # Step2.将地理经纬度的角度表示转化为弧度表示
    lat = deg2rad(lat)
    lon = deg2rad(lon)
    # Step3.将地理经纬度转化成地心经纬度
    eb2_ea2 = eb**2 / ea**2
    λe = lon
    φe = arctan(eb2_ea2 * tan(lat))
    # Step4.求Re
    cosφe = cos(φe)
    re = eb / sqrt(1 - (1 - eb2_ea2) * cosφe**2)
    # Step5.求r1,r2,r3
    λe_λD = λe - λD
    r1 = h - re * cosφe * cos(λe_λD)
    r2 = -re * cosφe * sin(λe_λD)
    r3 = re * sin(φe)
    # Step6.求rn,x,y
    rn = sqrt(r1**2 + r2**2 + r3**2)
    x = rad2deg(arctan(-r2 / r1))
    y = rad2deg(arcsin(-r3 / rn))
    # Step7.求c,l
    column = COFF[resolution] + x * 2**-16 * CFAC[resolution]
    line = LOFF[resolution] + y * 2**-16 * LFAC[resolution]
    return line, column

# 黄渤海经纬度
geo_range = [28.7, 41.5, 116.2, 129.01, 0.0125]

lat_S, lat_N, lon_W, lon_E, step = [int(x*10000) for x in geo_range] 
lat = np.arange(lat_N-step, lat_S-step, -step)/10000
lon = np.arange(lon_W, lon_E, step)/10000
lon_mesh, lat_mesh = np.meshgrid(lon, lat)
line_org, column_org = latlon2linecolumn(lat_mesh, lon_mesh)


def hdf2np(hdf_path):
    """
    input:hdf file path
    output:numpy array 14*1000*1200,dtype=float32  || None

    """
    global line_org, column_org
    res = []
    try:
        with h5py.File(hdf_path, 'r') as f:
            line = np.rint(line_org)-f.attrs['Begin Line Number'][()][0]
            line[line<=0]=0
            line = line.astype(np.uint16)
            column = np.rint(column_org).astype(np.uint16)

            for k in range(1,16):
                nom_name = 'NOMChannel'+'0'*(2-len(str(k)))+str(k)
                # print(nom_name)
                cal_name = 'CALChannel'+'0'*(2-len(str(k)))+str(k) 
                # print(cal_name)
                try:
                    nom = f["Data"][nom_name][()]
                    cal = f["Calibration"][cal_name][()]

                except:

                    with open('./error.log','a') as f:
                        f.write(hdf_path+' Channel:'+str(k)+'\n')
                    return 

                channel = nom[line, column]
                CALChannel = cal.astype(np.float32)


                CALChannel = np.append(CALChannel, 0)
                channel[channel >= 65534] = 4096

                res.append(CALChannel[channel])

            # save
            return np.array(res,dtype=np.float32)


    except:
        with open('./error.log','a') as f:
            f.write('Can not open:'+hdf_path+'\n')

        return None
    
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


if __name__=='__main__':
    INPUT_DIR = "./"  # 请修改存储FY4B-HDF的文件地址
    out_dir = './'    # 请修改存储图片的文件地址
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    paths = [x for x in os.listdir(INPUT_DIR) if x.split('.')[-1]=='hdf' or x.split('.')[-1]=='HDF']
    paths = [x for x in paths if x[:4]=='FY4B']
    for input_file in paths:
        input_name = os.path.basename(input_file)
        time = input_name.split('_')[-4][:-2]
        npy_14 = hdf2np(input_file)
        tc_img = make_TC(npy_14)
        nc_img = make_NC(npy_14)
        npy_savepath = out_dir + 'FY4B_'
        tc_savepath = out_dir + 'FY4B_TC_'
        nc_savepath = out_dir + 'FY4B_NC_'
        np.save(npy_savepath + time + '.npy', npy_14)
        cv2.imwrite(tc_savepath + time + ".png", tc_img)
        cv2.imwrite(nc_savepath + time + ".png", nc_img)
 