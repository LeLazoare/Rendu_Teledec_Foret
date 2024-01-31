# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:12:01 2023

@author: lazar
"""

##Concatenation : 


import sys
sys.path.append('C:/Users/lazar/OneDrive/Bureau/teledetection_avancee/scripts_python/')
import os 
#from osgeo import gdal
#import pandas as pd
#import geopandas
#import sklearn
#import plotly
import numpy as np
#import read_and_write as rw
#import classification as cla
# sklearn import tree
#import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import glob

#path et folders
inputs_folder = 'E:/M2/903_Algo_Avancee/teledetection_avancee/data_S2_forest_proj/24-02-2021'
output_path = 'E:/M2/903_Algo_Avancee/teledetection_avancee/outputs'  
output_concat = os.path.join(output_path, 'concatenate_test_image.tif')
filepaths = glob.glob(inputs_folder + '/*.tif')

def Concatenate_image(filepaths, output_path, output_concate):
    
    # Ouvrir tous les fichiers GeoTIFF et récupérer les bandes dans une liste
    bands = [rasterio.open(fp) for fp in filepaths]
    
    # Utiliser les métadonnées du premier fichier GeoTIFF pour créer le fichier résultant
    first_band = bands[0]
    profile = first_band.profile.copy()
    
    # Mettre à jour les métadonnées pour refléter le nouveau nombre de bandes
    profile.update(count=len(bands))
    
    # Enregistrer le fichier résultant
    with rasterio.open(output_concat, 'w', **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.read(1), i)
    
    print("Concaténation manuelle réussie. Fichier enregistré à :", output_concat)
    
        # Obtenir le nombre de bandes
    with rasterio.open(output_concat) as src:
        num_bands = src.count
    print(f"Le fichier GeoTIFF {output_concat} a {num_bands} bande(s).")
    
    return output_concate