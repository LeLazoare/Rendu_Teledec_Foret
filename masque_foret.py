# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:07:49 2023

@authors: Michel Tarby, Tom Mervant
"""
import geopandas as gpd
from osgeo import gdal
import os
import my_function as function

# Définition du dossier de travail et chargement des paramètres d'intérêt
dossier = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/'

BD_foret = os.path.join(dossier, 'data_ori/FORMATION_VEGETALE.shp')
emprise_etude = os.path.join(dossier, 'data_ori/emprise_etude.shp')
out_image = os.path.join(dossier, 'data_ori/masque_foret.tif')

# Couche à créer pour rasteriser par la suite
mask_foret_shp = os.path.join(dossier, 'data_ori/masque_BD_Foret.shp')

# Charger les données sous forme de dataframes
gdf = gpd.read_file(BD_foret)
empr_etu = gpd.read_file(emprise_etude)

# Données ne rentrant pas en compte dans l'étude
suppr = ['Forêt ouverte de feuillus purs', 
         'Forêt ouverte à mélange de feuillus et conifères',
         'Forêt ouverte de conifères purs',
         'Forêt ouverte sans couvert arboré',
         'Lande', 'Formation Herbacée']

# Filtrage du tableau pour supprimer les données ne rentrant pas en compte 
# dans l'étude
gdf_filtre = gdf[~gdf['TFV'].isin(suppr)]

# Création d'un champ Zone prenant la valeur 1 pour la rasterisation
gdf_filtre['Zone'] = 1

# Export en SHP
gdf_filtre.to_file(mask_foret_shp, driver="ESRI Shapefile")

# Rasterisation

# Définition de la résolution spatiale (10m) et récupération de l'emprise de 
# la zone d'intérêt

#issue when confronting results to gdal warp clipping option
input_fpath = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/warp-2/warped.tif'
xmin, ymin, xmax, ymax, sptial_resolution = function.get_raster_extent(input_fpath)

field_name = 'Zone' 

# Définition du pattern de la commande avec les paramètres
# Le raster de sortie sera au format GTiff et encodé en Byte
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

# Utilisation des paramètres dans la commande
cmd = cmd_pattern.format(in_vector=mask_foret_shp, xmin=xmin, ymin=ymin, 
                         xmax=xmax, ymax=ymax, out_image=out_image, 
                         field_name=field_name,
                         sptial_resolution=sptial_resolution)

# Execution du de la commande dans le terminal
os.system(cmd)
