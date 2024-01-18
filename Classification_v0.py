# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:06:01 2024

@author: Michel Tarby
"""

import sys
import geopandas as gpd
import pandas as pd

import os

from sklearn.model_selection import StratifiedGroupKFold, KFold
import numpy as np
from osgeo import gdal
from osgeo import ogr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from sklearn.model_selection import train_test_split

# personal librairies
sys.path.append('C:\tmp\projet_TLD_SIGMA\GitHub_projet\Rendu_Teledec_Foret')
import my_function as function



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


# 4 --- Perform the classification
for samp in range(0, len(samples_raster)):
    out_classif = os.path.join(my_folder, 'carte_essences{}'.format(
        samples_raster[samp][-9:]))
    function.classif_final(image_filename, samp, out_classif)
"""
# Sample parameters
#test_size = 0.7
nb_iter = 30
nb_folds = 5
is_point = False
# if is_point is True
field_name = 'num'

# outputs
suffix = '_CV{}folds_stratified_group_x{}times'.format(nb_folds, nb_iter)
out_folder = os.path.join(my_folder, 'results')
out_classif = os.path.join(out_folder, 'carte_essencelvl1.tif')



# 2 --- extract samples
if not is_point :
    X, Y, t = cla.get_samples_from_roi(image_filename, sample_filename)
    _, groups, _ = cla.get_samples_from_roi(image_filename, id_filename)
else :
    # get X
    list_row, list_col = rw.get_row_col_from_file(sample_filename, image_filename)
    image = rw.load_img_as_array(image_filename)
    X = image[(list_row, list_col)]

    # get Y
    gdf = gpd.read_file(sample_filename)
    Y = gdf.loc[:, field_name].values
    Y = np.atleast_2d(Y).T

list_cm = []
list_accuracy = []
list_report = []

# Iter on stratified K fold
for _ in range(nb_iter):
  kf = StratifiedGroupKFold(n_splits=nb_folds, shuffle=True)
  for train, test in kf.split(X, Y, groups=groups):
      X_train, X_test = X[train], X[test]
      Y_train, Y_test = Y[train], Y[test]

      # 3 --- Train
      #clf = SVC(cache_size=6000)
      clf = tree.DecisionTreeClassifier()
      clf.fit(X_train, Y_train)

      # 4 --- Test
      Y_predict = clf.predict(X_test)

      # compute quality
      list_cm.append(confusion_matrix(Y_test, Y_predict))
      list_accuracy.append(accuracy_score(Y_test, Y_predict))
      report = classification_report(Y_test, Y_predict,
                                     labels=np.unique(Y_predict),
                                     output_dict=True)

      # store them
      list_report.append(report_from_dict_to_df(report))"""

