"""
Subject: BD FORET enhancement project using REMOTE SENSING
preprocess -> base img and ndvi builder

Requirements: my_function.py script file

Authors: M.Tarby - G.Ruiz - L.Sauger - T.Mervant || SIGMA 2024
"""

#lib import
import sys
import time

from osgeo import gdal
import numpy as np 

sys.path.append('C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/scripts_')
import my_function as preprocess

##preprocess main hub
#convert format
start = time.time()
converted = preprocess.convert_to_uint16(
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/data_ori/S2A_15cc_FRE/S2A_L2A_FRE_ALL/24-02-2021/SENTINEL2A_20210224-105858-431_L2A_T31TCJ_C_V2-2_FRE_B12.tif',
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/convert-1/converted.tif'
    )
end = time.time()
print('processed in:', end - start)

#reproj - up/down sample - clip
start = time.time()
warped = preprocess.warp_tif(
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/convert-1/converted.tif',   
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/warp-2/warped.tif',
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/data_ori/emprise_etude.shp',
    'EPSG:2154',
    (10, 10)
)
end = time.time()
print('processed in:', end - start)


#mask || same clipping method should be used
#ensure same extent (condition)
input_fpath = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/warp-2/warped.tif'
mask_fpath = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/masque_foret.tif'
output_fpath = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/mask-3/masked.tif'

img_ext = preprocess.get_raster_extent(input_fpath)
mask_ext = preprocess.get_raster_extent(mask_fpath)

if img_ext != mask_ext:
    print('input files are not of same extent')

#apply mask 
start = time.time()
masked = preprocess.apply_mask(
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/warp-2/warped.tif',   
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/masque_foret.tif',
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/mask-3/masked.tif'
)
end = time.time()
print('processed in:', end - start)



