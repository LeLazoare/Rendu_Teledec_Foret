"""
Subject: BD FORET enhancement project using REMOTE SENSING
preprocess -> base img and ndvi builder

Requirements: my_function.py script file | librairies as follows (sys, os, glob)

Authors: M.Tarby - T.Mervant || SIGMA 2024
"""

#lib import
import sys
import os
import glob
import time

sys.path.append('C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/scripts_')
import my_function as preprocess

#specify existing folder
folder = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/tst'

#required base tree structure
inputs_folder = os.path.join(folder, 'inputs')
if not os.path.exists(inputs_folder):
        os.makedirs(inputs_folder)
        print(f"Created folder: {inputs_folder}")        
else:
        print(f"Folder already exists: {inputs_folder}")

res_folder = os.path.join(folder, 'res')
if not os.path.exists(res_folder):
        os.makedirs(res_folder)
        print(f"Created folder: {res_folder}")
else:
        print(f"Folder already exists: {res_folder}")

input_subfolders = ['sat_img']
for subfolder in input_subfolders:
        subfolder_path = os.path.join(inputs_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created folder: {subfolder_path}")
        else:
            print(f"Folder already exists: {subfolder_path}")

res_subfolders = ['convert-1', 'warp-2', 'mask-3', 'multispectral', 'ndvi']
for subfolder in res_subfolders:
        subfolder_path = os.path.join(res_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created folder: {subfolder_path}")
        else:
            print(f"Folder already exists: {subfolder_path}")


##preprocess main hub
####################################
#convert format
filepaths = glob.glob(os.path.join(folder,'inputs/sat_img/19-07-2021/*.tif'))#FILL IN DATE

output_dir = os.path.join(folder, 'res/convert-1')

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
roi_path = os.path.join(folder, 'inputs/emprise_etude.shp')

output_dir = os.path.join(folder, 'res/warp-2')

start = time.time()
for band_file in filepaths:

    #generate unique output_fpath
    file_name, file_extension = os.path.splitext(os.path.basename(band_file))
    output_file = os.path.join(output_dir, f'{file_name}_w{file_extension}')

    #submit to warp
    warped = preprocess.warp_tif(
        band_file,
        output_file,
        roi_path,
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
mask_fpath = os.path.join(folder, 'inputs/masque_foret.tif')
output_dir = os.path.join(folder, 'res/mask-3')


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
msdate_path = os.path.join(folder, 'res/multispectral/2021-07-19.tif')#FILL IN DATE ACCORDINGLY
start = time.time()

ms = preprocess.build_ms(
    filepaths,
    False,
    msdate_path
)

end = time.time()
print('processed in:', end - start)

#WARNING: wait for process to end before removing files
#remove files from previous process -- mask
for f in filepaths:
    os.remove(f)

####################################
#build multispectral img - multiple dates
output_dir = os.path.join(folder, 'res/multispectral/maj_ms/Serie_temp_S2_allbands.tif')
#extract the directory path from the file path
output_dir_path = os.path.dirname(output_dir)
#check if the directory exists, and create it if not
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

filepaths = glob.glob(os.path.join(folder,'res/multispectral/*.tif'))
start = time.time()

ms = preprocess.build_ms(
    filepaths,
    True,
    output_dir
)

end = time.time()
print('processed in:', end - start)

####################################
#compute NDVI for each date
filepaths = glob.glob(os.path.join(folder,'res/multispectral/*.tif'))

output_dir = os.path.join(folder, 'res/ndvi')
#extract the directory path from the file path
output_dir_path = os.path.dirname(output_dir)
#check if the directory exists, and create it if not
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

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

output_dir = os.path.join(folder, 'res/ndvi/maj_ndvi/Serie_temp_S2_ndvi.tif')
#extract the directory path from the file path
output_dir_path = os.path.dirname(output_dir)
#check if the directory exists, and create it if not
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

maj_ndvi = preprocess.build_ms(
    filepaths,
    False,
    output_dir
)

end = time.time()
print('processed in:', end - start)