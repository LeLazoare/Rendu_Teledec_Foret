
#lib import
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

gdf_poly
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
##############################################
#REFERENCE IMAGE FOR RASTERIZATION
input_fpath = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/ndvi/2021-02-24_ndvi.tif'
reference_img = gdal.Open(input_fpath)
geotransform = reference_img.GetGeoTransform()
print(geotransform)

spatial_resolution = geotransform[1]
xmin = geotransform[0]
ymax = geotransform[3]
xmax = xmin + geotransform[1] * reference_img.RasterXSize
ymin = ymax + geotransform[5] * reference_img.RasterYSize
##############################################

##############################################
#RASTERIZATION PARAMETER
folder = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class'
input_vectors = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/Sample_BD_foret_T31TCJ/Sample_BD_foret_T31TCJ.shp'
ndvi_maj = 'C:/Users/Xenerios/Desktop/adv_remote-sensing/forest_class/res/ndvi/maj_ndvi/maj_ndvi.tif'
##############################################
output_fpath = os.path.join(folder, 'res/samples/lvl1.tif')

##############################################
#RASTERIZATION LEVEL 1
start = time.time()

#create empty dataset using input_dataset main properties with chosen format
output_dataset = gdal.GetDriverByName('GTiff').Create(
    output_fpath,
    reference_img.RasterXSize,
    reference_img.RasterYSize,
    reference_img.RasterCount, #ndvi == 1 band
    gdal.GDT_Byte
)

output_dataset.SetGeoTransform(reference_img.GetGeoTransform())
output_dataset.SetProjection(reference_img.GetProjection())

#default background value
background_value = 0
output_dataset.GetRasterBand(1).Fill(background_value)

#get corresponding field name from dictionary
field_name = 'Code_lvl1'

print(f"Processing {'lvl1'} with input vector: {input_vectors} and field name: {field_name}")

#define command parameters
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

#set parameters
cmd = cmd_pattern.format(in_vector=input_vectors, xmin=xmin, ymin=ymin, 
                         xmax=xmax, ymax=ymax, out_image=output_fpath, 
                         field_name=field_name,
                         sptial_resolution=spatial_resolution)

#print command before launching
print(f"Executing command: {cmd}")

#launch command
os.system(cmd)

print(f"Rasterization completed for {'lvl1'}")
print(f"Output file path: {output_fpath}")

end = time.time()
print('processed in:', end - start)
##############################################
#GET SAMPLES FROM ROI
X, Y, t = cla.get_samples_from_roi(ndvi_maj, 
                                   output_fpath, 
                                   value_to_extract=None,
                                   bands=None, 
                                   output_fmt='by_label')

##############################################
#DISPLAY LVL1 MEAN AND STD NDVI VALUES
bands_names = ['2021-02-24', '2021-03-31', '2021-04-15', '2021-07-19', '2021-10-17', '2021-12-16']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

for label, ndvi_values in X.items():
    means = np.mean(ndvi_values, axis=0)
    stds = np.std(ndvi_values, axis=0)

    ax.plot(means, label=f'Mean - Class {label}')
    ax.fill_between(range(len(bands_names)), means + stds, means - stds, alpha=0.3, label=f'Std Dev - Class {label}')

ax.set_xticks(range(len(bands_names)))
ax.set_xticklabels(bands_names)


#shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

ax.set_xlabel('Bands')
ax.set_ylabel('NDVI Values')
ax.set_title('LVL 1 - Temporal Signature of Mean NDVI Values with Standard Deviation by Class')

#put legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#save fig
plt.savefig(f'temp_mean_ndvi_lvl1.png', bbox_inches='tight') 

plt.show()

##############################################
#RASTERIZATION LEVEL 2
output_fpath = os.path.join(folder, 'res/samples/lvl2.tif')

start = time.time()

#create empty dataset using input_dataset main properties with chosen format
output_dataset = gdal.GetDriverByName('GTiff').Create(
    output_fpath,
    reference_img.RasterXSize,
    reference_img.RasterYSize,
    reference_img.RasterCount, #ndvi == 1 band
    gdal.GDT_Byte
)

output_dataset.SetGeoTransform(reference_img.GetGeoTransform())
output_dataset.SetProjection(reference_img.GetProjection())

#default background value
background_value = 0
output_dataset.GetRasterBand(1).Fill(background_value)

#get corresponding field name from dictionary
field_name = 'Code_lvl2'

print(f"Processing {'lvl2'} with input vector: {input_vectors} and field name: {field_name}")

#define command parameters
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

#set parameters
cmd = cmd_pattern.format(in_vector=input_vectors, xmin=xmin, ymin=ymin, 
                         xmax=xmax, ymax=ymax, out_image=output_fpath, 
                         field_name=field_name,
                         sptial_resolution=spatial_resolution)

#print command before launching
print(f"Executing command: {cmd}")

#launch command
os.system(cmd)

print(f"Rasterization completed for {'lvl2'}")
print(f"Output file path: {output_fpath}")

end = time.time()
print('processed in:', end - start)

##############################################
#GET SAMPLES FROM ROI
X, Y, t = cla.get_samples_from_roi(ndvi_maj, 
                                   output_fpath, 
                                   value_to_extract=None,
                                   bands=None, 
                                   output_fmt='by_label')

##############################################
#DISPLAY LVL2 MEAN AND STD NDVI VALUES
bands_names = ['2021-02-24', '2021-03-31', '2021-04-15', '2021-07-19', '2021-10-17', '2021-12-16']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

for label, ndvi_values in X.items():
    means = np.mean(ndvi_values, axis=0)
    stds = np.std(ndvi_values, axis=0)

    ax.plot(means, label=f'Mean - Class {label}')
    ax.fill_between(range(len(bands_names)), means + stds, means - stds, alpha=0.3, label=f'Std Dev - Class {label}')

ax.set_xticks(range(len(bands_names)))
ax.set_xticklabels(bands_names)


#shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

ax.set_xlabel('Bands')
ax.set_ylabel('NDVI Values')
ax.set_title('LVL 2 - Temporal Signature of Mean NDVI Values with Standard Deviation by Class')

#put legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#save fig
plt.savefig(f'temp_mean_ndvi_lvl2.png', bbox_inches='tight') 

plt.show()

##############################################
#RASTERIZATION LVL3
output_fpath = os.path.join(folder, 'res/samples/lvl3.tif')

start = time.time()

#create empty dataset using input_dataset main properties with chosen format
output_dataset = gdal.GetDriverByName('GTiff').Create(
    output_fpath,
    reference_img.RasterXSize,
    reference_img.RasterYSize,
    reference_img.RasterCount, #ndvi == 1 band
    gdal.GDT_Byte
)

output_dataset.SetGeoTransform(reference_img.GetGeoTransform())
output_dataset.SetProjection(reference_img.GetProjection())

#default background value
background_value = 0
output_dataset.GetRasterBand(1).Fill(background_value)

#get corresponding field name from dictionary
field_name = 'Code_lvl3'

print(f"Processing {'lvl3'} with input vector: {input_vectors} and field name: {field_name}")

#define command parameters
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

#set parameters
cmd = cmd_pattern.format(in_vector=input_vectors, xmin=xmin, ymin=ymin, 
                         xmax=xmax, ymax=ymax, out_image=output_fpath, 
                         field_name=field_name,
                         sptial_resolution=spatial_resolution)

#print command before launching
print(f"Executing command: {cmd}")

#launch command
os.system(cmd)

print(f"Rasterization completed for {'lvl3'}")
print(f"Output file path: {output_fpath}")

end = time.time()
print('processed in:', end - start)

##############################################
#GET SAMPLES FROM ROI
X, Y, t = cla.get_samples_from_roi(ndvi_maj, 
                                   output_fpath, 
                                   value_to_extract=None,
                                   bands=None, 
                                   output_fmt='by_label')

##############################################
#DISPLAY LVL3 MEAN AND STD NDVI VALUES
bands_names = ['2021-02-24', '2021-03-31', '2021-04-15', '2021-07-19', '2021-10-17', '2021-12-16']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

for label, ndvi_values in X.items():
    means = np.mean(ndvi_values, axis=0)
    stds = np.std(ndvi_values, axis=0)

    ax.plot(means, label=f'Mean - Class {label}')
    ax.fill_between(range(len(bands_names)), means + stds, means - stds, alpha=0.3, label=f'Std Dev - Class {label}')

ax.set_xticks(range(len(bands_names)))
ax.set_xticklabels(bands_names)

#shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

ax.set_xlabel('Bands')
ax.set_ylabel('NDVI Values')
ax.set_title('LVL 3 - Temporal Signature of Mean NDVI Values with Standard Deviation by Class')

#put legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#save fig
plt.savefig(f'temp_mean_ndvi_lvl3.png', bbox_inches='tight') 

plt.show()

 
