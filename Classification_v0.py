# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:06:01 2024

@author: Michel Tarby
"""
# Import libraries
import sys
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from osgeo import gdal
from osgeo import ogr

# personal librairies
sys.path.append('C:\tmp\projet_TLD_SIGMA\GitHub_projet\Rendu_Teledec_Foret')
import my_function as function
import function_test as function2

# 1 --- inputs
my_folder = 'C:/tmp/projet_TLD_SIGMA/Results'
sample_filename = os.path.join(my_folder, 'Sample_BD_foret_T31TCJ.shp')
image_filename = os.path.join(my_folder, 'D:/3A/TLD/rm_bdforet_res/rm_bdforet_res/multispectral/maj_ms/maj_ms.tif') #Image à 60 bandes

# 2 --- Rasterize each level from polygons
xmin, ymin, xmax, ymax, sptial_resolution = function.get_raster_extent(image_filename)
fields = ["Code_lvl1", "Code_lvl2", "Code_lvl3"]
encoding = 'Byte'
# Définition du pattern de la commande avec les paramètres
# Le raster de sortie sera au format GTiff et encodé en Byte
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot {encoding} -of GTiff "
               "{in_vector} {out_image}")

# Rasterization
samples_raster = []
for field_name in fields:
    out_image = os.path.join(my_folder,'Sample_BD_foret_{}.tif'.format(field_name))
    samples_raster.append(out_image)
    # Utilisation des paramètres dans la commande
    cmd = cmd_pattern.format(in_vector=sample_filename, xmin=xmin, ymin=ymin, 
                             xmax=xmax, ymax=ymax, encoding = encoding, 
                             out_image=out_image, 
                             field_name=field_name,
                             sptial_resolution=sptial_resolution)
    
    # Execution du de la commande dans le terminal
    os.system(cmd)
    
# 3 --- Create a raster with groups of each polygon
gdf = gpd.read_file(sample_filename)  #Load the data in a gdf
gdf['poly_id'] = range(1, len(gdf) + 1) #Attribute a unique code for each polygon

# Export in shp the gdf
in_vector = os.path.join(my_folder,'code_group.shp')
gdf.to_file(in_vector)

# Rasterize the vector file
out_name = os.path.join(my_folder,'code_group.tif')

# There is more than 250 polygons, Int16 is used instead of Byte
encoding = 'Int16'
cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, 
                             xmax=xmax, ymax=ymax, encoding = encoding,
                             out_image=out_name, 
                             field_name="poly_id",
                             sptial_resolution=sptial_resolution)
os.system(cmd)

# Remove shp file from directory
driver = ogr.GetDriverByName("ESRI Shapefile")
if os.path.exists(in_vector):
    driver.DeleteDataSource(in_vector)


# 4 --- Perform classification and validation

# Get pixels values of the image that will be classified
X_img, _, t_img = function2.get_samples_from_roi(image_filename, image_filename)
# Get groups from samples
_, groups, _ = function2.get_samples_from_roi(image_filename, out_name)

# Iterate through each level of classification
mean_report, std_report = [], []
for samp in range(0, len(samples_raster)):
    out_classif = os.path.join(my_folder, 'carte_essences{}'.format(
        samples_raster[samp][-9:]))
    print("Nom out_classif: {}".format(out_classif))
    
    # Perform classification and get X, Y and t for validation
    X, Y, t = function2.classif_final(image_filename, samples_raster[samp], 
                                      out_classif, X_img, t_img)
    print("X, Y et t récupérés, carte créée")
    name = 'Niveau {}'.format(samp+1)
    
    # Perform validation
    mean_df_report, std_df_report = function2.classif_Kfolds(groups, X, Y, t, 
                                                             name)
    # Store report for each level
    mean_report.append(mean_df_report)
    std_report.append(std_df_report)
    print("Classification et validation effectuée pour {}".format(name))

