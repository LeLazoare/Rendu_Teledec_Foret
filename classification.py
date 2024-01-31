# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:06:01 2024

@author: Michel Tarby, Tom Mervant, Gabriel Ruiz, Lazare Sauger
"""
# Import libraries
import geopandas as gpd
import os
from osgeo import ogr
from osgeo import gdal
# personal librairies
import my_function as function

# 1 --- inputs
my_folder = 'C:/your_path_to_directory'
sample_filename = os.path.join(my_folder, 'Sample_BD_foret_T31TCJ.shp')
image_filename = os.path.join(my_folder, 'Serie_temp_S2_allbands.tif') 

# 2 --- Rasterize each level from polygons
xmin, ymin, xmax, ymax, sptial_resolution = function.get_raster_extent(
    image_filename)
fields = ["Code_lvl1", "Code_lvl2", "Code_lvl3"]
encoding = 'Byte'
# Definition of the command pattern with parameters
# The output raster will be in GTiff format and encoded in Byte
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot {encoding} -of GTiff "
               "{in_vector} {out_image}")

# Rasterization
samples_raster = []
for field_name in fields:
    out_image = os.path.join(my_folder,'Sample_BD_foret_{}.tif'.format(
        field_name))
    samples_raster.append(out_image)
    # Utilisation des paramètres dans la commande
    cmd = cmd_pattern.format(in_vector=sample_filename, xmin=xmin, ymin=ymin, 
                             xmax=xmax, ymax=ymax, encoding = encoding, 
                             out_image=out_image, 
                             field_name=field_name,
                             sptial_resolution=sptial_resolution)
    
    # Execute command in a terminal
    os.system(cmd)
    
# 3 --- Create a raster with groups of each polygon for each level
gdf_lvl1 = gpd.read_file(sample_filename)  #Load the data in a gdf

# Discard lines wit No Data for each level
gdf_lvl2 = gdf_lvl1.dropna(subset=['Code_lvl2'])
gdf_lvl3 = gdf_lvl1.dropna(subset=['Code_lvl3'])

#Attribute a unique code for each polygon
gdf_lvl1['poly_id'] = range(1, len(gdf_lvl1) + 1)
gdf_lvl2['poly_id'] = range(1, len(gdf_lvl2) + 1) 
gdf_lvl3['poly_id'] = range(1, len(gdf_lvl3) + 1) 

# Store the gdfs in a list
gdfs = [gdf_lvl1, gdf_lvl2, gdf_lvl3]

# Export in shp gdfs
lvl = 0
groups_raster = []
# Iterate in each gdf
for level in gdfs:
    lvl += 1
    # Write gdf as shp files
    in_vector = os.path.join(my_folder,'group_{}.shp'.format(lvl))
    level.to_file(in_vector)
    out_name = os.path.join(my_folder,'group_{}.tif'.format(lvl))
    # There is more than 250 polygons, Int16 is used instead of Byte
    encoding = 'Int16'
    # Rasterize the vector file
    cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, 
                                 xmax=xmax, ymax=ymax, encoding = encoding,
                                 out_image=out_name, 
                                 field_name="poly_id",
                                 sptial_resolution=sptial_resolution)
    os.system(cmd)
    
    # Collect rasters name
    groups_raster.append(out_name)
    
    # Remove shp file from directory
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(in_vector):
        driver.DeleteDataSource(in_vector)


# 4 --- Perform classification and validation

# Get pixels values of the image that will be classified
X_img, _, t_img = function.get_samples_from_roi(image_filename, image_filename)

# Get groups from samples
groups_matrix = []
for raster in groups_raster:
    _, groups, _ = function.get_samples_from_roi(image_filename, raster)
    groups_matrix.append(groups)

# Get X, Y, and t for each level and create name 
name_out_classif, X_list, Y_list, t_list = [], [], [], []
for i in range(0, len(samples_raster)):
    # Create name and directory for the classification
    out_classif = os.path.join(my_folder, 'carte_essences{}'.format(
        samples_raster[i][-9:]))
    name_out_classif.append(out_classif)
    # Get X, Y and t and store them in lists
    X, Y, t = function.get_samples_from_roi(image_filename, samples_raster[i])
    X_list.append(X), Y_list.append(Y), t_list.append(t)
    
# Perform classification on each level
for j in range(0, len(X_list)):
    function.classif_final(image_filename, X_list[j], Y_list[j], 
                            name_out_classif[j], X_img, t_img)
    
# Perform validation on each level
std_report_list, mean_report_list = [], []
for k in range(0, len(X_list)):
    name = ' Niveau {}'.format(k+1)
    mean_df_report, std_df_report = function.classif_Kfolds(
        groups_matrix[k], X_list[k], Y_list[k], name)
    std_report_list.append(std_df_report)
    mean_report_list.append(mean_df_report)
    
# 5 --- Regroup, create correspondant map and perform validation
# Create maps
# # Create out name for each regroup
out_lvl3_to_lvl2 = os.path.join(my_folder, 'carte_essences_lvl2_fromlvl3.tif')
out_lvl3_to_lvl1 = os.path.join(my_folder, 'carte_essences_lvl1_fromlvl3.tif')
out_lvl2_to_lvl1 = os.path.join(my_folder, 'carte_essences_lvl1_fromlvl2.tif')

# Level 3 to level 2
lvl3_to_lvl2 = function.load_img_as_array(name_out_classif[2])
lvl3_to_lvl2=function.regroup_classes(1, lvl3_to_lvl2)
ds = function.open_image(name_out_classif[2])
function.write_image(out_lvl3_to_lvl2, lvl3_to_lvl2, gdal_dtype=gdal.GDT_Byte,
                      data_set=ds)

# Level 3 to level 1
lvl3_to_lvl1 = function.load_img_as_array(name_out_classif[2])
lvl3_to_lvl1 = function.regroup_classes(2, lvl3_to_lvl1)
function.write_image(out_lvl3_to_lvl1, lvl3_to_lvl1, gdal_dtype=gdal.GDT_Byte,
                      data_set=ds)

# Level 2 to level 1
lvl2_to_lvl1 = function.load_img_as_array(name_out_classif[1])
lvl2_to_lvl1 = function.regroup_classes(3, lvl2_to_lvl1)
ds2 = function.open_image(name_out_classif[1])
function.write_image(out_lvl2_to_lvl1, lvl2_to_lvl1, gdal_dtype=gdal.GDT_Byte, 
                      data_set=ds2)

# Perform validation
mean_df_report_kf, std_df_report_kf = function.classif_Kfolds(
    groups_matrix[1], X_list[2], Y_list[2], 
    ' Regroupement Niveau 3 vers Niveau 2', 1)

mean_df_report_kf2, std_df_report_kf2 = function.classif_Kfolds(
    groups_matrix[1], X_list[2], Y_list[2], 
    ' Regroupement Niveau 3 vers Niveau 1', 2)

mean_df_report_kf3, std_df_report_kf3 = function.classif_Kfolds(
    groups_matrix[0], X_list[1], Y_list[1], 
    ' Regroupement Niveau 2 vers Niveau 1', 3)
