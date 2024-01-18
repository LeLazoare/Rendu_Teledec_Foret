"""
Subject: BD FORET enhancement project using REMOTE SENSING
preprocess -> base img and ndvi builder

Requirements: my_function.py script file | librairies as follows (sys, os, glob)

Authors: M.Tarby - G.Ruiz - L.Sauger - T.Mervant || SIGMA 2024
"""

#lib import
import sys
import os, shutil
import glob
import time

sys.path.append('C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/scripts_')
import my_function as preprocess

folder = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class'
##preprocess main hub
####################################
#convert format
filepaths = glob.glob(os.path.join(folder,'data_ori/S2A_15cc_FRE/S2A_L2A_FRE_ALL/16-12-2021/*.tif'))#FILL IN DATE

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

#WARNING: wait for process to end before removing files
#remove files from previous process -- convert
for f in filepaths:
    os.remove(f)

####################################
#mask || same clipping method should be used
#ensure same extent (condition)
filepaths = glob.glob(os.path.join(folder,'res/warp-2/*.tif'))
mask_fpath = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/masque_foret.tif'
output_dir = os.path.join(folder, 'res/mask-3')
#os_makedirs(output_dir, exist_ok=True) - if output_dir does not exist

start = time.time()
for band_file in filepaths:

    #check extent
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

#WARNING: wait for process to end before removing files
#remove files from previous process -- warp
for f in filepaths:
    os.remove(f)


####################################
#build multispectral img - single date only
filepaths = glob.glob(os.path.join(folder,'res/mask-3/*.tif'))

start = time.time()

ms = preprocess.build_ms(
    filepaths,
    False,
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/multispectral/2021-12-16.tif'#FILL IN DATE
)

end = time.time()
print('processed in:', end - start)

#WARNING: wait for process to end before removing files
#remove files from previous process -- mask
for f in filepaths:
    os.remove(f)

####################################
#build multispectral img - multiple dates

filepaths = glob.glob(os.path.join(folder,'res/multispectral/*.tif'))
start = time.time()

ms = preprocess.build_ms(
    filepaths,
    True,
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/multispectral/maj_ms/maj_ms.tif'
)

end = time.time()
print('processed in:', end - start)

####################################
#compute NDVI for each date
filepaths = glob.glob(os.path.join(folder,'res/multispectral/*.tif'))

output_dir = os.path.join(folder, 'res/ndvi')
#os_makedirs(output_dir, exist_ok=True) - if output_dir does not exist

start = time.time()
for ms_file in filepaths:

    #generate unique output_fpath
    file_name, file_extension = os.path.splitext(os.path.basename(ms_file))
    output_file = os.path.join(output_dir, f'{file_name}_ndvi{file_extension}')

    #compute NDVI
    ndvi = preprocess.compute_ndvi(
        ms_file,
        3,
        7,
        output_file
    )

end = time.time()
print('processed in:', end - start)

####################################
#merge NDVI into a single raster
filepaths = glob.glob(os.path.join(folder,'res/ndvi/*.tif'))
start = time.time()

maj_ndvi = preprocess.build_ms(
    filepaths,
    False,
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/ndvi/maj_ndvi/maj_ndvi.tif'
)

end = time.time()
print('processed in:', end - start)