import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from satpy.scene import Scene
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action="ignore", category=RuntimeWarning)
from pyresample import create_area_def
import os

'''
The files in the `folder_path` are all unzipped MSG data downloaded from the EUMETSAT official website. The naming format and file structure of each file are as follows:

MSG4-SEVI-MSG15-0100-NA-20210301121243.394000000Z-NA
 - EOPMetadata.xml
 - manifest.xml
 - MSG4-SEVI-MSG15-0100-NA-20210301121243.394000000Z-NA.nat

`png_path` and `npy_path` are the paths where you want to store the target file after processing. The folder of `Year` is required.
'''


folder_path = '/nas/downloads/meteosat_20_21_22_Mar_Apr_12am/60_17/'
png_path = './pure_png/'
npy_path = './npy/'


def draw_png(region, file, png_path):
    save_name = png_path + file[-32:-28] + '/' + 'M11-' + region + '-' + file[-32:-20] + '.png'
    if os.path.exists(save_name):
        print(f"File '{save_name}' already exists. Skipping...")

    else:
        if region == 'E':
            loc = [-7.8, 47.2, 5, 60.0]
        elif region == 'DW':
            loc = [0, 33, 12.8, 45.8]
        elif region == 'DE':
            loc = [15.0, 30, 27.8, 42.8]
        elif region == 'B':
            loc = [27, 37.2, 39.8, 50]
        elif region == 'FN':
            loc = [4, -26.8, 16.8, -14]
        elif region == 'FS':
            loc = [8, -38, 20.8, -25.2]
        else:
            print("Region input ERROR!!!")
            
        file_name = glob.glob(file)
        scn = Scene(reader="seviri_l1b_native", filenames=file_name)
        composite_id = ['natural_color']
        scn.load(composite_id, upper_right_corner="NE")

        area_id = 'msg_seviri_fes_3km'
        proj_dict = {'lon_0': '0', 'proj': 'eqc'}
        my_area = create_area_def(area_id, proj_dict, width=1024, height=1024, area_extent=loc, units='degrees')

        scn_resample_nc = scn.resample(my_area)
        
        image = np.asarray(scn_resample_nc["natural_color"]).transpose(1, 2, 0)
        image = np.interp(image, (np.percentile(image, 1), np.percentile(image, 99)), (0, 1))
        
        print('saving:', save_name)
        plt.imsave(save_name, image)


def draw_npy(region, file, npy_path):
    save_name = npy_path + file[-32:-28] + '/' + 'M11-' + region + '-' + file[-32:-20] + '.npy'
    if os.path.exists(save_name):
        print(f"File '{save_name}' already exists. Skipping...")

    else:
        if region == 'E':
            loc = [-7.8, 47.2, 5, 60.0]
        elif region == 'DW':
            loc = [0, 33, 12.8, 45.8]
        elif region == 'DE':
            loc = [15.0, 30, 27.8, 42.8]
        elif region == 'B':
            loc = [27, 37.2, 39.8, 50]
        elif region == 'FN':
            loc = [4, -26.8, 16.8, -14]
        elif region == 'FS':
            loc = [8, -38, 20.8, -25.2]
        else:
            print("Region input ERROR!!!")
            
        file_name = glob.glob(file)
        scn = Scene(reader="seviri_l1b_native", filenames=file_name)

        area_id = 'msg_seviri_fes_3km'
        proj_dict = {'lon_0': '0', 'proj': 'eqc'}
        my_area = create_area_def(area_id, proj_dict, width=1024, height=1024, area_extent=loc, units='degrees')

        scn.load(['HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'])
        scn_resample_nc = scn.resample(my_area)
        
        image = np.asarray([scn_resample_nc['HRV'], scn_resample_nc['IR_016'], scn_resample_nc['IR_039'], scn_resample_nc['IR_087'], scn_resample_nc['IR_097'], scn_resample_nc['IR_108'], scn_resample_nc['IR_120'], scn_resample_nc['IR_134'], scn_resample_nc['VIS006'], scn_resample_nc['VIS008'], scn_resample_nc['WV_062'], scn_resample_nc['WV_073'] ])
        
        print('saving:', save_name)
        np.save(save_name, image)
    


def main(folder_path, png_path, npy_path):  
    for item in os.listdir(folder_path):
        file_path = os.path.join(folder_path, item, item)
        file = file_path + '.nat'
        print("====================")
        print(file)
        draw_png('E', file, png_path)
        draw_png('DW', file, png_path)
        draw_png('DE', file, png_path)
        draw_png('B', file, png_path)
        draw_png('FN', file, png_path)
        draw_png('FS', file, png_path)

        draw_npy('E', file, npy_path)
        draw_npy('DW', file, npy_path)
        draw_npy('DE', file, npy_path)
        draw_npy('B', file, npy_path)
        draw_npy('FN', file, npy_path)
        draw_npy('FS', file, npy_path)


if __name__ == '__main__':
    main(folder_path, png_path, npy_path)