"""
Subject: BD FORET enhancement project using REMOTE SENSING
preprocess -> base img and ndvi builder

Requirements: my_function.py script file

Authors: M.Tarby - G.Ruiz - L.Sauger - T.Mervant || SIGMA 2024
"""

#lib import
import sys
import os
import glob
import time

from osgeo import gdal


sys.path.append('C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/scripts_')
import my_function as preprocess

folder = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class'
##preprocess main hub
####################################
#convert format
filepaths = glob.glob(os.path.join(folder,'data_ori/S2A_15cc_FRE/S2A_L2A_FRE_ALL/24-02-2021/*.tif'))

output_dir = os.path.join(folder, 'res/convert-1')
#os_makedirs(output_dir, exist_ok=True) - if output_dir does not exist

start = time.time()
for band_file in filepaths:

    #generate unique output_fpath
    file_name, file_extension = os.path.splitext(os.path.basename(band_file))
    output_file = os.path.join(output_dir, f'{file_name}_c{file_extension}')

    #apply convert process
    converted = preprocess.convert_to_uint16(
        band_file,
        output_file
    )

end = time.time()
print('processed in:', end - start)

####################################
#reproj - up/down sample - clip
filepaths = glob.glob(os.path.join(folder,'res/convert-1/*.tif'))

output_dir = os.path.join(folder, 'res/warp-2')
#os_makedirs(output_dir, exist_ok=True) - if output_dir does not exist

start = time.time()
for band_file in filepaths:

    #generate unique output_fpath
    file_name, file_extension = os.path.splitext(os.path.basename(band_file))
    output_file = os.path.join(output_dir, f'{file_name}_w{file_extension}')

    #submit to warp
    warped = preprocess.warp_tif(
        band_file,
        output_file,
        'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/data_ori/emprise_etude.shp',
        'EPSG:2154',
        (10, 10)
    )

end = time.time()
print('processed in:', end - start)

####################################
#mask || same clipping method should be used
#ensure same extent (condition)
filepaths = glob.glob(os.path.join(folder,'res/warp-2/*.tif'))
mask_fpath = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/masque_foret.tif'
output_dir = os.path.join(folder, 'res/mask-3')
#os_makedirs(output_dir, exist_ok=True) - if output_dir does not exist

start = time.time()
for band_file in filepaths:

    img_ext = preprocess.get_raster_extent(band_file)
    mask_ext = preprocess.get_raster_extent(mask_fpath)

    if img_ext != mask_ext:
        print('input files are not of same extent')
        break
    
    #generate unique output_fpath
    file_name, file_extension = os.path.splitext(os.path.basename(band_file))
    output_file = os.path.join(output_dir, f'{file_name}_m{file_extension}')

    #mask values out of mask
    masked = preprocess.apply_mask(
        band_file,
        mask_fpath,
        output_file
    )

end = time.time()
print('processed in:', end - start)