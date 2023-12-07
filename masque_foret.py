# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:07:49 2023

@author: Michel Tarby
"""
import geopandas as gpd
import os

# Définition du dossier de travail et chargement des paramètres d'intérêt
dossier = r'C:\tmp\projet_TLD_SIGMA\data_mask'

BD_foret = os.path.join(dossier, 'FORMATION_VEGETALE.shp')
# Masque créé au format shp:
mask_foret_shp = os.path.join(dossier, 'masque_BD_Foret.shp')
emprise_etude = os.path.join(dossier, 'emprise_etude.shp')
out_image = os.path.join(dossier, 'masque_foret.tif')

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
sptial_resolution = 10
xmin, ymin, xmax, ymax = empr_etu.total_bounds
field_name = 'Zone' 
#
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

# execute the command in the terminal
os.system(cmd)
