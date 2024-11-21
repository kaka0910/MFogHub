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

# 一些默认参数
ea = 6378.137  # 地球的半长轴[km]
eb = 6356.7523  # 地球的短半轴[km]
h = 42164  # 地心到卫星质心的距离[km]
λD = deg2rad(104.7)  # 卫星星下点所在经度

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

# 转化成地球经纬度
def latlon2linecolumn(lat, lon, resolution='4000M'):
    """
    (lat, lon) → (line, column)
    resolution: 文件名中的分辨率{'0500M', '1000M', '2000M', '4000M'}
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

            for k in range(1,15):
                nom_name = 'NOMChannel'+'0'*(2-len(str(k)))+str(k)
                cal_name = 'CALChannel'+'0'*(2-len(str(k)))+str(k) 

                try:
                    nom = f[nom_name][()]
                    cal = f[cal_name][()]

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


def rec_datetime(path):
    mode = path.split('_')[3]
    
    if mode == 'DISK':
        datetime_1 = path.split('_')[-3]
        min = datetime_1[10:12]
        if int(min)>=0 and int(min)<30:
            min = '00'
        else:
            min = '30'
        datetime = datetime_1[:8]+'_'+datetime_1[8:10]+min
    
    else:
        datetime_1 = path.split('_')[-4]
        min = datetime_1[10:12]
        if int(min)==30:
            datetime = datetime_1[:8]+'_'+datetime_1[8:10]+'30'
        else:
            datetime = None
    
    return datetime

if __name__=='__main__':

    geo_range = [ 28.70, 41.50, 116.20, 129.01, 0.0125]  # 黄渤海经纬度

    hdfpath = '/nas2/data/users/xmq/M4Fog/Data_process/Satellites/' # 请修改存储FY4A-HDF的文件地址
    paths = [x for x in os.listdir(hdfpath) if x.split('.')[-1]=='hdf' or x.split('.')[-1]=='HDF']
    paths = [x for x in paths if x[:4]=='FY4A']

    for path in paths:
        pathdir = hdfpath + path
        datetime = rec_datetime(path)
        print(datetime)
        if datetime != None:
            lat_S, lat_N, lon_W, lon_E, step = [x for x in geo_range]
            lat = np.linspace(lat_N, lat_S, int((lat_N-lat_S)/step))
            lon = np.linspace(lon_W, lon_E, int((lon_E-lon_W)/step))

            # 生成经纬度矩阵
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)
            line_org, column_org = latlon2linecolumn(lat_mesh, lon_mesh)
        
            # 存储npy文件, 0/1/2三个通道合成图像
            npy_14 = hdf2np(pathdir)
            tc_img = make_TC(npy_14)
            nc_img = make_NC(npy_14)
            
            npy_savepath = hdfpath+'FY4A_'
            tc_savepath = hdfpath+'FY4A_TC_'
            nc_savepath = hdfpath+'FY4A_NC_'
            np.save(npy_savepath+datetime+'.npy', npy_14)
            cv2.imwrite(tc_savepath+datetime+".png", tc_img)
            cv2.imwrite(nc_savepath+datetime+".png", nc_img)
