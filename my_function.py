"""
Subject: BD FORET enhancement project using REMOTE SENSING
preprocess -> base functions

Requirements: librairies as follows (osgdal, numpy) 

Authors: M.Tarby - G.Ruiz - L.Sauger - T.Mervant || SIGMA 2024
"""

#lib import
from osgeo import gdal #main img handling lib
import numpy as np #overall calculations


def warp_tif(
    input_fpath,
    output_fpath,
    shapefile_fpath,
    crs,
    target_res, 
    ):

    """
    Using GDAL WARP TOOL : reprojection, upsampling/downsampling, clipping to shp for a specified tif img

    REQUIREMENTS
    __________
    from osgeo import gdal
    
    PARAMETERS
    __________
    input_fpath | string
    filepath to file related to tif img to process 

    output_fpath | string
    filepath to file where processed img is to be written

    shapefile_fpath | string
    filepath to file related to shp mask to clip tif img with

    crs | string
    chosen espg to reproject, as follows : 'EPSG:2154'

    target_res | list of int
    xRes and yRes for resampling, as follows : (10, 10)
    """
      
    ##open as gdal dataset obj
    input_dataset = gdal.Open(input_fpath)
    
    ##warp properties
    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS = crs,
        xRes=target_res[0],
        yRes=target_res[1],
        resampleAlg=gdal.GRA_Bilinear,
        cutlineDSName=shapefile_fpath,
        cropToCutline=True,
        dstNodata=0, 
    )
    
    ##process warp
    output_dataset = gdal.Warp(
                        output_fpath, 
                        input_dataset, 
                        options = warp_options
                             )

    #close datasets to save memory space
    input_dataset = None
    output_dataset = None

def convert_to_uint16(input_fpath, output_fpath):

    '''
    Using GDAL : convert input tif img format to UInt16

    REQUIREMENTS
    __________
    from osgeo import gdal
    
    PARAMETERS
    __________
    input_fpath | string
    filepath to file related to tif img to process 

    output_fpath | string
    filepath to file where processed img is to be written
    '''
    
    ##open as gdal dataset obj
    input_dataset = gdal.Open(input_fpath)

    #check data type
    band_val = input_dataset.GetRasterBand(1)
    gdal_data_type = gdal.GetDataTypeName(band_val.DataType)
    if gdal_data_type == 'UInt16':
        print('input_dataset already coded in UInt16')
        return

    #create empty dataset using input_dataset main properties with chosen format
    output_dataset = gdal.GetDriverByName('GTiff').Create(
        output_fpath,
        input_dataset.RasterXSize,
        input_dataset.RasterYSize,
        input_dataset.RasterCount,
        gdal.GDT_UInt16 
    )

    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    output_dataset.SetProjection(input_dataset.GetProjection())

    #retrieve and write bands with chosen format
    for band_num in range(1, input_dataset.RasterCount + 1):
        input_band = input_dataset.GetRasterBand(band_num)
        output_band = output_dataset.GetRasterBand(band_num)

        data = input_band.ReadAsArray().astype('uint16')
        output_band.WriteArray(data)
        output_band.FlushCache()

    #close datasets to save memory space
    input_dataset = None
    output_dataset = None

def convert_to_uint8(input_fpath, output_fpath):

    '''
    Using GDAL : convert input tif img format to Byte #UInt8

    REQUIREMENTS
    __________
    from osgeo import gdal
    
    PARAMETERS
    __________
    input_fpath | string
    filepath to file related to tif img to process 

    output_fpath | string
    filepath to file where processed img is to be written
    '''
    
    ##open as gdal dataset obj
    input_dataset = gdal.Open(input_fpath)

    #check data type
    band_val = input_dataset.GetRasterBand(1)
    gdal_data_type = gdal.GetDataTypeName(band_val.DataType)
    if gdal_data_type == 'Byte':
        print('input_dataset already coded in uint8')
        return

    #create empty dataset using input_dataset main properties with chosen format
    output_dataset = gdal.GetDriverByName('GTiff').Create(
        output_fpath,
        input_dataset.RasterXSize,
        input_dataset.RasterYSize,
        input_dataset.RasterCount,
        gdal.GDT_Byte
    )

    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    output_dataset.SetProjection(input_dataset.GetProjection())

    #retrieve and write bands with chosen format
    for band_num in range(1, input_dataset.RasterCount + 1):
        input_band = input_dataset.GetRasterBand(band_num)
        output_band = output_dataset.GetRasterBand(band_num)

        data = input_band.ReadAsArray().astype('uint8')
        output_band.WriteArray(data)
        output_band.FlushCache()

    #close datasets to save memory space
    input_dataset = None
    output_dataset = None

def get_raster_extent(input_fpath):
    '''
    Using GDAL : get raster extent

    REQUIREMENTS
    __________
    from osgeo import gdal
    
    PARAMETERS
    __________
    input_fpath | string
    filepath to file related to tif img to process 
    '''

    ##open as gdal dataset obj
    input_dataset = gdal.Open(input_fpath)

    ##retrieve pixel size and original coordinates
    geotransform = input_dataset.GetGeoTransform()

    ##get raster extent
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + geotransform[1] * input_dataset.RasterXSize
    ymin = ymax + geotransform[5] * input_dataset.RasterYSize

    #close dataset to save memory space
    input_dataset = None
    
    return(xmin, ymin, xmax, ymax)

def apply_mask(input_fpath, mask_fpath, output_fpath):
    '''
    Using GDAL : apply raster file as mask to another raster of SAME EXTENT, 
    so that values strictly OUTSIDE the mask are set to 0

    REQUIREMENTS
    __________
    from osgeo import gdal
    
    PARAMETERS
    __________
    input_fpath | string
    filepath to file related to tif img to process 

    mask_fpath | string
    filepath to file related to tif img to use as mask

    output_fpath | string
    filepath to file where processed img is to be written
    '''

    ##open as gdal dataset obj
    input_dataset = gdal.Open(input_fpath)
    mask = gdal.Open(mask_fpath)

    #load as np.ndarray
    input_dataset_ar = input_dataset.ReadAsArray()
    mask_ar = mask.ReadAsArray()

    mask_ar = mask_ar.astype('bool')  # convert to boolean format
    mask_ar = np.squeeze(mask_ar)  # supress the third dimension so
                                        # that it can be use img_masked

    # do some processing with numpy: mask the image
    data_masked = input_dataset_ar.copy()
    data_masked[~mask_ar] = 0  #bitwise operator -- equivalent to NOT such as in !
    data_masked = data_masked

    #create empty dataset using input_dataset main properties
    output_dataset = gdal.GetDriverByName('GTiff').Create(
        output_fpath,
        input_dataset.RasterXSize,
        input_dataset.RasterYSize,
        input_dataset.RasterCount,
        input_dataset.GetRasterBand(1).DataType
        )

    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    output_dataset.SetProjection(input_dataset.GetProjection())

    #retrieve and write bands
    for band_num in range(1, input_dataset.RasterCount + 1):
        output_band = output_dataset.GetRasterBand(band_num)
        #output_band.WriteArray(data_masked[band_num - 1, :, :])#error: mask array cannot be 3D
        output_band.WriteArray(data_masked)

    # Close datasets
    input_dataset = None
    mask = None
    output_dataset = None

def build_ms(input_fpaths, output_fpath):
    '''
    Using GDAL : build multispectral img from bands in distinct raster files

    REQUIREMENTS
    __________
    from osgeo import gdal
    
    PARAMETERS
    __________
    input_fpaths | string
    filepaths to files related to tif imgs to process 

    output_fpath | string
    filepath to file where processed img is to be written
    '''

    #open as gdal dataset obj
    input_datasets = [gdal.Open(file) for file in input_fpaths]

    #template img
    template_dataset = input_datasets[0]
    width = template_dataset.RasterXSize
    height = template_dataset.RasterYSize
    projection = template_dataset.GetProjection()

    #create empty dataset using template_dataset main properties
    driver = gdal.GetDriverByName(
        template_dataset.GetDriver().ShortName
        )

    output_dataset = driver.Create(
        output_fpath, 
        template_dataset.RasterXSize, 
        template_dataset.RasterYSize, 
        len(input_datasets), 
        template_dataset.GetRasterBand(1).DataType
        )

    #write down bands
    for i, input_dataset in enumerate(input_datasets):
        for band_num in range(1, input_dataset.RasterCount + 1):
            output_band = output_dataset.GetRasterBand(i + 1)
            input_band = input_dataset.GetRasterBand(band_num)
            output_band.WriteArray(input_band.ReadAsArray())

    #set geotransform and projection
    output_dataset.SetGeoTransform(template_dataset.GetGeoTransform())
    output_dataset.SetProjection(template_dataset.GetProjection())

    #close datasets to save memory space
    for dataset in input_datasets:
        dataset = None

    output_dataset = None

