
# Importation des bibliothèques
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats 

import time

# charger les données sous forme de dataframe 
gdf_poly = gpd.read_file('C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/Sample_BD_foret_T31TCJ/Sample_BD_foret_T31TCJ.shp')

# regrouper les données selon la combinaison (unique) des colonnes des niveaux 1, 2 et 3.
# size() pour compter le nombre de polygones
# reset_index() réinitialise l'index du DataFrame résultant et renomme la 
# colonne de comptage comme « count ».
grouped_data = gdf_poly.groupby(['Nom_lvl1', 'Nom_lvl2', 'Nom_lvl3']).size()\
    .reset_index(name='count')

grouped_data

# Liste des niveaux de nomenclature
niveaux = [1, 2, 3]

### Nombre de polygones par classe

# itérer sur la liste; pour chaque niveau, on construit dynamiquement le nom 
# du champ, on récupère les valeurs uniques de ce niveau et on les compte. 
for niveau in niveaux:
    niveau_column = f'Nom_lvl{niveau}'
    unique_values = grouped_data[niveau_column].unique()
    counts = grouped_data.groupby(niveau_column)['count'].sum()\
        .reindex(unique_values, fill_value=0) 
    
    # Générer un diagramme en bâton pour chaque niveau de nomenclature
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(unique_values, counts, color='green')
    
    # Générer le décompte au dessus de chaque bar
    for bar in bars:
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + (100/len(unique_values)),
          round(bar.get_height(), 1),
          horizontalalignment='center',
          color='black',
          weight='light'
      )
      
    ax.set_title(f'Nombre de polygones par classe selon le niveau {niveau}')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Nombre de polygones')
    # angle des classes pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')  
    
    # Ajout et colorisation de lignes verticales grises
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='grey')
    ax.xaxis.grid(False)
    
    # Sauvegarde de la figure eu format png
    plt.savefig(f'diag_baton_nb_poly_lvl{niveau}.png', bbox_inches='tight')  
    


### Nombre de pixels par classe

# charger le fichier vecteur
gdf_pix = gpd.read_file('C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/Sample_BD_foret_T31TCJ/Sample_BD_foret_T31TCJ.shp')

# charger le fichier raster
with rasterio.open('C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/masque_foret.tif') as src:
    # Lire les données raster
    raster_data = src.read(1)

    # Calcul du nombre de pixels (stats="count") dans chaque géométrie du gdf 
    # en fonction des valeurs des pixels du raster
    stats = zonal_stats(gdf_pix.geometry, raster_data, affine=src.transform, 
                        stats="count", nodata=src.nodata)

# ajouter une colonne avec le nombre de pixels par classe dans le gdf
gdf_pix['class_pixels'] = [stat['count'] for stat in stats]

gdf_pix

# liste des niveaux de nomenclature
niveaux = [1, 2, 3]



# itérer sur la liste; pour chaque niveau, on construit dynamiquement le nom du
# champ et on agrège le nb de pixels pour chaque classe de chaque colonne 
for niveau in niveaux:
    niveau_column = f'Nom_lvl{niveau}'
    
    # Générer un diagramme en bâton pour chaque niveau de nomenclature
    fig, ax = plt.subplots(figsize=(14, 7))
    # Grouper les données
    grouped_data = gdf_pix.groupby(niveau_column)['class_pixels'].sum()
    
    # Tracer le diagramme en bâtons
    bars = ax.bar(grouped_data.index, grouped_data.values, color='#de892f')
    
   # Générer le décompte au dessus de chaque bar
    for bar, value in zip(bars, grouped_data.values):
        ax.annotate(str(value),
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2),  # Ajustez cet espace vertical
                    textcoords="offset points",
                    ha='center', va='bottom', color='black', fontweight='light')
    
    # paramètres du diagramme
    ax.set_title(f'Nombre de pixels par classe selon le niveau {niveau}', 
                 color='#333333',
                 weight='bold')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Nombre de pixels')
    # angle des classes pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right') 
    
    
    # Ajout et colorisation de lignes verticales grises
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='grey')
    ax.xaxis.grid(False)
    
    # Sauvegarde de la figure eu format png
    plt.savefig(f'diag_baton_nb_pix_lvl{niveau}.png', bbox_inches='tight')  


 ########## NDVI STATS
#load raster file to compute stats with #NB: raster file has already been clipped to mask
ndvi = rasterio.open(
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/ndvi/maj_ndvi/maj_ndvi.tif'
                     )

#read as array (panchromatic image such as NDVI only)
ndvi_array = ndvi.read(1)

#load polygons to compute stats on
samples_gdf = gpd.read_file(
    'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/Sample_BD_foret_T31TCJ/Sample_BD_foret_T31TCJ.shp'
    )

#compute mean ndvi value for each polygon
start = time.time()

mean_ndvi = zonal_stats(
    samples_gdf.geometry, 
    ndvi_array, 
    affine = ndvi.transform,
    stats='mean',
    nodata = ndvi.nodata
)

end = time.time()
print('processed in:', end - start)

#add values for each polygon
mean_ndvi_df = pd.DataFrame(mean_ndvi)
samples_gdf['mean_ndvi'] = mean_ndvi_df

#compute standard deviation from mean ndvi value for each polygon
start = time.time()

std_ndvi = zonal_stats(
    samples_gdf.geometry, 
    ndvi_array, 
    affine = ndvi.transform,
    stats='std',
    nodata = ndvi.nodata
)

std_ndvi_df = pd.DataFrame(std_ndvi)
samples_gdf['std_ndvi'] = std_ndvi_df

end = time.time()
print('processed in:', end - start)

samples_gdf

