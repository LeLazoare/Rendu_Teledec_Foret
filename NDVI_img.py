# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:35:12 2024

@author: lazar
"""
import numpy as np
from osgeo import gdal
import os
data_type_match = {'uint8': gdal.GDT_Byte,
                   'uint16': gdal.GDT_UInt16,
                   'uint32': gdal.GDT_UInt32,
                   'int16': gdal.GDT_Int16,
                   'int32': gdal.GDT_Int32,
                   'float32': gdal.GDT_Float32,
                   'float64': gdal.GDT_Float64}

# define parameters
dirname = 'E:/M2/903_Algo_Avancee/teledetection_avancee/td_M2_read_and_write'
out_dirname = 'E:/M2/903_Algo_Avancee/teledetection_avancee/outputs'
filename = os.path.join(dirname, 'imagette_fabas.tif')
out_ndvi_filename = os.path.join(out_dirname, 'imagette_fabas_ndvi.tif')
def ndvi_out():
    def open_image(filename):
      
      data_set = gdal.Open(filename, gdal.GA_ReadOnly)
    
      if data_set is None:
          print('Impossible to open {}'.format(filename))
      else:
          print('{} is open'.format(filename))
    
      return data_set
    
    def get_image_dimension(data_set):
    
        nb_col = data_set.RasterXSize
        nb_lignes = data_set.RasterYSize
        nb_band = data_set.RasterCount
        print('Number of columns :', nb_col)
        print('Number of lines :', nb_lignes)
        print('Number of bands :', nb_band)
    
        return nb_lignes, nb_col, nb_band
    
    def convert_data_type_from_gdal_to_numpy(gdal_data_type):
        
        if gdal_data_type == 'Byte':
            numpy_data_type = 'uint8'
        else:
            numpy_data_type = gdal_data_type.lower()
        return numpy_data_type
    
    def load_img_as_array(filename):
        
    
        # Get size of output array
        data_set = open_image(filename)
        nb_lignes, nb_col, nb_band = get_image_dimension(data_set)
    
        # Get data type
        band = data_set.GetRasterBand(1)
        gdal_data_type = gdal.GetDataTypeName(band.DataType)
        numpy_data_type = convert_data_type_from_gdal_to_numpy(gdal_data_type)
    
        # Initialize an empty array
        array = np.empty((nb_lignes, nb_col, nb_band), dtype=numpy_data_type)
    
        # Fill the array
        for idx_band in range(nb_band):
            idx_band_gdal = idx_band + 1
            array[:, :, idx_band] = data_set.GetRasterBand(idx_band_gdal).ReadAsArray()
    
        # close data_set
        data_set = None
        band = None
    
        return array
    
    def write_image(out_filename, array, data_set=None, gdal_dtype=None,
                    transform=None, projection=None, driver_name=None,
                    nb_col=None, nb_ligne=None, nb_band=None):
        
        # Get information from array if the parameter is missing
        nb_col = nb_col if nb_col is not None else array.shape[1]
        nb_ligne = nb_ligne if nb_ligne is not None else array.shape[0]
        array = np.atleast_3d(array)  # not asked in the instructions.
                                      # but it deals with the case a 2d
                                      # dimension array is passed.
        nb_band = nb_band if nb_band is not None else array.shape[2]
    
    
        # Get information from data_set if provided
        transform = transform if transform is not None else data_set.GetGeoTransform()
        projection = projection if projection is not None else data_set.GetProjection()
        gdal_dtype = gdal_dtype if gdal_dtype is not None \
            else data_set.GetRasterBand(1).DataType
        driver_name = driver_name if driver_name is not None \
            else data_set.GetDriver().ShortName
    
        # Create DataSet
        driver = gdal.GetDriverByName(driver_name)
        output_data_set = driver.Create(out_filename, nb_col, nb_ligne, nb_band,
                                        gdal_dtype)
        output_data_set.SetGeoTransform(transform)
        output_data_set.SetProjection(projection)
    
        # Fill it and write image
        for idx_band in range(nb_band):
            output_band = output_data_set.GetRasterBand(idx_band + 1)
            output_band.WriteArray(array[:, :, idx_band])  # not working with a 2d array.
                                                           # this is what np.atleast_3d(array)
                                                           # was for
            output_band.FlushCache()
    
        del output_band
        output_data_set = None
    # load data
    data_set = open_image(filename)
    img = load_img_as_array(filename)
    
    # do some processingwith numpy: compute ndvi
    ir = img[:,:, 3].astype('float32')
    r = img[:,:, 0].astype('float32')
    ndvi = (ir - r) / (ir + r)
    
    # write it
    write_image(out_ndvi_filename, ndvi, data_set=data_set,
                gdal_dtype=data_type_match['float32'])
    return 

ndvi_out()
