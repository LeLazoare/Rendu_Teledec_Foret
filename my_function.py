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
    filepath to file where process img is to be written

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
    filepath to file where process img is to be written
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
    filepath to file where process img is to be written
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
