# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:58:45 2024

@author: HP OMEN
"""
from osgeo import gdal #main img handling lib
import numpy as np #overall calculations
import subprocess
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
import matplotlib.pyplot as plt
from museotoolbox.charts import PlotConfusionMatrix
from matplotlib.pyplot import cm as colorMap
def open_image(filename, verbose=False):
  """
  Open an image file with gdal

  Paremeters
  ----------
  filename : str
      Image path to open

  Return
  ------
  osgeo.gdal.Dataset
  """
  data_set = gdal.Open(filename, gdal.GA_ReadOnly)

  if data_set is None:
      print('Impossible to open {}'.format(filename))
  elif data_set is not None and verbose:
      print('{} is open'.format(filename))

  return data_set

def get_image_dimension(data_set, verbose=False):
    """
    get image dimensions

    Parameters
    ----------
    data_set : osgeo.gdal.Dataset

    Returns
    -------
    nb_lignes : int
    nb_col : int
    nb_band : int
    """

    nb_col = data_set.RasterXSize
    nb_lignes = data_set.RasterYSize
    nb_band = data_set.RasterCount
    if verbose:
        print('Number of columns :', nb_col)
        print('Number of lines :', nb_lignes)
        print('Number of bands :', nb_band)

    return nb_lignes, nb_col, nb_band


def get_origin_coordinates(data_set, verbose=False):
    """
    get origin coordinates

    Parameters
    ----------
    data_set : osgeo.gdal.Dataset

    Returns
    -------
    origin_x : float
    origin_y : float
    """
    geotransform = data_set.GetGeoTransform()
    origin_x, origin_y = geotransform[0], geotransform[3]
    if verbose:
        print('Origin = ({}, {})'.format(origin_x, origin_y))

    return origin_x, origin_y

def get_pixel_size(data_set, verbose=False):
    """
    get pixel size

    Parameters
    ----------
    data_set : osgeo.gdal.Dataset

    Returns
    -------
    psize_x : float
    psize_y : float
    """
    geotransform = data_set.GetGeoTransform()
    psize_x, psize_y = geotransform[1],geotransform[5]
    if verbose:
        print('Pixel Size = ({}, {})'.format(psize_x, psize_y))

    return psize_x, psize_y

def convert_data_type_from_gdal_to_numpy(gdal_data_type):
    """
    convert data type from gdal to numpy style

    Parameters
    ----------
    gdal_data_type : str
        Data type with gdal syntax
    Returns
    -------
    numpy_data_type : str
        Data type with numpy syntax
    """
    if gdal_data_type == 'Byte':
        numpy_data_type = 'uint8'
    else:
        numpy_data_type = gdal_data_type.lower()
    return numpy_data_type

def load_img_as_array(filename, verbose=False):
    """
    Load the whole image into an numpy array with gdal

    Paremeters
    ----------
    filename : str
        Path of the input image

    Returns
    -------
    array : numpy.ndarray
        Image as array
    """

    # Get size of output array
    data_set = open_image(filename, verbose=verbose)
    nb_lignes, nb_col, nb_band = get_image_dimension(data_set, verbose=verbose)

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
    """
    Write a array into an image file.

    Parameters
    ----------
    out_filename : str
        Path of the output image.
    array : numpy.ndarray
        Array to write
    nb_col : int (optional)
        If not indicated, the function consider the `array` number of columns
    nb_ligne : int (optional)
        If not indicated, the function consider the `array` number of rows
    nb_band : int (optional)
        If not indicated, the function consider the `array` number of bands
    data_set : osgeo.gdal.Dataset
        `gdal_dtype`, `transform`, `projection` and `driver_name` values
        are infered from `data_set` in case there are not indicated.
    gdal_dtype : int (optional)
        Gdal data type (e.g. : gdal.GDT_Int32).
    transform : tuple (optional)
        GDAL Geotransform information same as return by
        data_set.GetGeoTransform().
    projection : str (optional)
        GDAL projetction information same as return by
        data_set.GetProjection().
    driver_name : str (optional)
        Any driver supported by GDAL. Ignored if `data_set` is indicated.
    Returns
    -------
    None
    """
    # Get information from array if the parameter is missing
    nb_col = nb_col if nb_col is not None else array.shape[1]
    nb_ligne = nb_ligne if nb_ligne is not None else array.shape[0]
    array = np.atleast_3d(array)
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


def xy_to_rowcol(x, y, image_filename):
    """
    Convert geographic coordinates into row/col coordinates

    Paremeters
    ----------
    x : float
      x geographic coordinate
    y : float
        y geographic coordinate
    image_filename : str
        Path of the image.

    Returns
    -------
    row : int
    col : int
    """
    # get image infos
    data_set = open_image(image_filename)
    origin_x, origin_y = get_origin_coordinates(data_set)
    psize_x, psize_y = get_pixel_size(data_set)

    # convert x y to row col
    col = int((x - origin_x) / psize_x)
    row = - int((origin_y - y) / psize_y)

    return row, col


def get_xy_from_file(filename):
    """
    Get x y coordinates from a vector point file

    Parameters
    ----------
    filename : str
        Path of the vector point file

    Returns
    -------
    list_x : np.array
    list_y : np.array
    """
    gdf = gpd.read_file(filename)
    geometry = gdf.loc[:, 'geometry']
    list_x = geometry.x.values
    list_y = geometry.y.values

    return list_x, list_y

def get_row_col_from_file(point_file, image_file):
    """
    Getrow col image coordinates from a vector point file
    and image file

    Parameters
    ----------
    point_file : str
        Path of the vector point file
    image_file : str
        Path of the raster image file

    Returns
    -------
    list_row : np.array
    list_col : np.array
    """
    list_row = []
    list_col = []
    list_x, list_y = get_xy_from_file(point_file)
    for x, y in zip(list_x, list_y):
        row, col = xy_to_rowcol(x, y, image_file)
        list_row.append(row)
        list_col.append(col)
    return list_row, list_col

def get_data_for_scikit(point_file, image_file, field_name):
    """
    Get a sample matrix and a label matrix from a point vector file and an
    image.

    Parameters
    ----------
    point_file : str
        Path of the vector point file
    image_file : str
        Path of the raster image file
    field_name : str
        Field name containing the numeric label of the sample.

    Returns
    -------
     X : ndarray or dict of ndarra
        The sample matrix. A nXd matrix, where n is the number of referenced
        pixels and d is the number of variables. Each line of the matrix is a
        pixel.
    Y : ndarray
        the label of the pixel
    """

    list_row, list_col = get_row_col_from_file(point_file, image_file)
    image = load_img_as_array(image_file)
    X = image[(list_row, list_col)]

    gdf = gpd.read_file(point_file)
    Y = gdf.loc[:, field_name].values
    Y = np.atleast_2d(Y).T

    return X, Y

def get_samples_from_roi(raster_name, roi_name, value_to_extract=None,
                         bands=None, output_fmt='full_matrix'):
    '''
    The function get the set of pixel of an image according to an roi file
    (raster). In case of raster format, both map should be of same
    size.

    Parameters
    ----------
    raster_name : string
        The name of the raster file, could be any file GDAL can open
    roi_name : string
        The path of the roi image.
    value_to_extract : float, optional, defaults to None
        If specified, the pixels extracted will be only those which are equal
        this value. By, defaults all the pixels different from zero are
        extracted.
    bands : list of integer, optional, defaults to None
        The bands of the raster_name file whose value should be extracted.
        Indexation starts at 0. By defaults, all the bands will be extracted.
    output_fmt : {`full_matrix`, `by_label` }, (optional)
        By default, the function returns a matrix with all pixels present in the
        ``roi_name`` dataset. With option `by_label`, a dictionnary
        containing as many array as labels present in the ``roi_name`` data
        set, i.e. the pixels are grouped in matrices corresponding to one label,
        the keys of the dictionnary corresponding to the labels. The coordinates
        ``t`` will also be in dictionnary format.

    Returns
    -------
    X : ndarray or dict of ndarra
        The sample matrix. A nXd matrix, where n is the number of referenced
        pixels and d is the number of variables. Each line of the matrix is a
        pixel.
    Y : ndarray
        the label of the pixel
    t : tuple or dict of tuple
        tuple of the coordinates in the original image of the pixels
        extracted. Allow to rebuild the image from `X` or `Y`
    '''

    # Get size of output array
    raster = open_image(raster_name)
    nb_col, nb_row, nb_band = get_image_dimension(raster)

    # Get data type
    band = raster.GetRasterBand(1)
    gdal_data_type = gdal.GetDataTypeName(band.DataType)
    numpy_data_type = convert_data_type_from_gdal_to_numpy(gdal_data_type)

    # Check if is roi is raster or vector dataset
    roi = open_image(roi_name)

    if (raster.RasterXSize != roi.RasterXSize) or \
            (raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        print('Raster : {}'.format(raster_name))
        print('Roi : {}'.format(roi_name))
        exit()

    if not bands:
        bands = list(range(nb_band))
    else:
        nb_band = len(bands)

    #  Initialize the output
    ROI = roi.GetRasterBand(1).ReadAsArray()
    if value_to_extract:
        t = np.where(ROI == value_to_extract)
    else:
        t = np.nonzero(ROI)  # coord of where the samples are different than 0

    Y = ROI[t].reshape((t[0].shape[0], 1)).astype('int32')

    del ROI
    roi = None  # Close the roi file

    try:
        X = np.empty((t[0].shape[0], nb_band), dtype=numpy_data_type)
    except MemoryError:
        print('Impossible to allocate memory: roi too large')
        exit()

    # Load the data
    for i in bands:
        temp = raster.GetRasterBand(i + 1).ReadAsArray()
        X[:, i] = temp[t]
    del temp
    raster = None  # Close the raster file

    # Store data in a dictionnaries if indicated
    if output_fmt == 'by_label':
        labels = np.unique(Y)
        dict_X = {}
        dict_t = {}
        for lab in labels:
            coord = np.where(Y == lab)[0]
            dict_X[lab] = X[coord]
            dict_t[lab] = (t[0][coord], t[1][coord])

        return dict_X, Y, dict_t
    else:
        return X, Y, t,

def custom_bg(ax, x_label=None, y_label=None, fontsize=18, labelsize=14,
              x_grid=True, y_grid=True, minor=True):

    ax.set_facecolor('ivory')

    # custom label
    x_label = ax.get_xlabel() if not x_label else x_label
    ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=fontsize)
    y_label = ax.get_ylabel() if not y_label else y_label
    ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=fontsize)

    # custom border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='x', colors='darkslategrey', labelsize=labelsize)
    ax.tick_params(axis='y', colors='darkslategrey', labelsize=labelsize)

    # custom grid
    if minor:
        ax.minorticks_on()
    if y_grid:
        ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                      linewidth=0.5, zorder=1)
        ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                      linewidth=0.3, zorder=1)
    if x_grid:
        ax.xaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                      linewidth=0.5, zorder=1)

        ax.xaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                      linewidth=0.3, zorder=1)
    return ax

def plot_cm(cm, labels, out_filename=None):
    """
    Plot a confusion matrix

    Parameters
    ----------
    cm : np.array
        Confusion matrix, reference are expected in rows and prediction in
        columns
    labels : list of string,
        Labels of the classes.
    out_filename : str (optional)
        If indicated, the chart is saved at the `out_filename` location
    """

    pltCm = PlotConfusionMatrix(cm, cmap=colorMap.YlGn)

    pltCm.add_text(font_size=12)
    pltCm.add_x_labels(labels, rotation=45)
    pltCm.add_y_labels(labels)
    pltCm.color_diagonal(diag_color=colorMap.YlGn,
                          matrix_color=colorMap.Reds)
    pltCm.add_accuracy(invert_PA_UA=False, user_acc_label='Recall',
                        prod_acc_label='Precision')
    pltCm.add_f1()
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight')

def plot_class_quality(report, accuracy, out_filename=None):
    """
    Display a plot bar of quality metrics of each class.

    Parameters
    ----------
    report : dict
        Classification report (output of the `classification_report` function
        of scikit-learn.
    accuracy : float
        Overall accuracy.
    out_filename : str (optional)
        If indicated, the chart is saved at the `out_filename` location
    """
    report_df = pd.DataFrame.from_dict(report)
    # drop columns (axis=1) same as numpy
    try :
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'],
                                   axis=1)
    except KeyError:
        report_df = report_df.drop(['macro avg', 'weighted avg'],
                                   axis=1)
    # drop rows (axis=0) same as numpy
    report_df = report_df.drop(['support'], axis=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = report_df.T.plot.bar(ax=ax, zorder=2)

    # custom : information
    ax.text(0.05, 0.95, 'OA : {:.2f}'.format(accuracy), fontsize=14)
    ax.set_title('Class quality estimation')

    # custom : cuteness
    # background color
    ax.set_facecolor('ivory')
    # labels
    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    y_label = ax.get_ylabel()
    ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    # borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
    ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
    # grid
    ax.minorticks_on()
    ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                  linewidth=0.5, zorder=1)
    ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                  linewidth=0.3, zorder=1)
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight')


def plot_mean_class_quality(list_df_report, list_accuracy, out_filename=None):
    """
    Display a plot bar of quality metrics of each class.

    Parameters
    ----------
    report : dict
        Classification report (output of the `classification_report` function
        of scikit-learn.
    accuracy : float
        Overall accuracy.
    out_filename : str (optional)
        If indicated, the chart is saved at the `out_filename` location
    """

    # compute mean of accuracy
    array_accuracy = np.asarray(list_accuracy)
    mean_accuracy = array_accuracy.mean()
    std_accuracy = array_accuracy.std()

    array_report = np.array(list_df_report)
    mean_report = array_report.mean(axis=0)
    std_report = array_report.std(axis=0)
    a_report = list_df_report[0]
    mean_df_report = pd.DataFrame(mean_report, index=a_report.index,
                                  columns=a_report.columns)
    std_df_report = pd.DataFrame(std_report, index=a_report.index,
                                 columns=a_report.columns)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax = mean_df_report.T.plot.bar(ax=ax,
                                   yerr=std_df_report.T, zorder=2)
    # custom : information
    ax.set_ylim(0.5, 1)
    ax.text(1.5, 0.95, 'OA : {:.2f} +- {:.2f}'.format(mean_accuracy,
                                                      std_accuracy),
            fontsize=14)
    ax.set_title('Class quality estimation')

    # custom : cuteness
    # background color
    ax.set_facecolor('ivory')
    # labels
    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    y_label = ax.get_ylabel()
    ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    # borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
    ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
    # grid
    ax.minorticks_on()
    ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                  linewidth=0.5, zorder=1)
    ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                  linewidth=0.3, zorder=1)
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight')

def report_from_dict_to_df(dict_report):

    # convert report into dataframe
    report_df = pd.DataFrame.from_dict(dict_report)

    # drop unnecessary rows and columns
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    report_df = report_df.drop(['support'], axis=0)

    return report_df

def classif_Kfolds(image_filename, sample_filename, id_filename):
    # Sample parameters
    #test_size = 0.7
    nb_iter = 30
    nb_folds = 5
    
    
    # 2 --- extract samples
    X, Y, t = get_samples_from_roi(image_filename, sample_filename)
    _, groups, _ = get_samples_from_roi(image_filename, id_filename)
    
    
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
          clf = RF(max_depth=20, oob_score=True,max_samples=0.75, 
                   class_weight='balanced')
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
          list_report.append(report_from_dict_to_df(report))
    
    # compute mean of cm
    array_cm = np.array(list_cm)
    mean_cm = array_cm.mean(axis=0)
    
    # compute mean and std of overall accuracy
    array_accuracy = np.array(list_accuracy)
    mean_accuracy = array_accuracy.mean()
    std_accuracy = array_accuracy.std()
    
    # compute mean and std of classification report
    array_report = np.array(list_report)
    mean_report = array_report.mean(axis=0)
    std_report = array_report.std(axis=0)
    a_report = list_report[0]
    mean_df_report = pd.DataFrame(mean_report, index=a_report.index,
                                  columns=a_report.columns)
    std_df_report = pd.DataFrame(std_report, index=a_report.index,
                                 columns=a_report.columns)
    
    # Display confusion matrix
    plot_cm(mean_cm, np.unique(Y_predict))
    
    # Display class metrics
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = mean_df_report.T.plot.bar(ax=ax, yerr=std_df_report.T, zorder=2)
    ax.set_ylim(0.5, 1)
    _ = ax.text(1.5, 0.95, 'OA : {:.2f} +- {:.2f}'.format(mean_accuracy,
                                                          std_accuracy),
                fontsize=14)
    ax.set_title('Class quality estimation')
    
    # custom : cuteness
    # background color
    ax.set_facecolor('ivory')
    # labels
    x_label = ax.get_xlabel()
    ax.set_xlabel(x_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    y_label = ax.get_ylabel()
    ax.set_ylabel(y_label, fontdict={'fontname': 'Sawasdee'}, fontsize=14)
    # borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='x', colors='darkslategrey', labelsize=14)
    ax.tick_params(axis='y', colors='darkslategrey', labelsize=14)
    # grid
    ax.minorticks_on()
    ax.yaxis.grid(which='major', color='darkgoldenrod', linestyle='--',
                  linewidth=0.5, zorder=1)
    ax.yaxis.grid(which='minor', color='darkgoldenrod', linestyle='-.',
                  linewidth=0.3, zorder=1)

def classif_final(image_filename, sample_filename, out_classif):
    # 1 --- extract samples
    X, Y, t = get_samples_from_roi(image_filename, sample_filename)
    
    # 2 --- Apply the model
    clf = RF(max_depth=10, oob_score=True,max_samples=0.10, 
                   class_weight='balanced')
    clf.fit(X, Y)

    
    # 3 --- apply on the whole image
    # load image
    X_img, _, t_img = get_samples_from_roi(image_filename, image_filename)
    
    # predict image
    Y_predict = clf.predict(X_img)
    
    # reshape
    ds = open_image(image_filename)
    nb_row, nb_col, _ = get_image_dimension(ds)
    
    img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
    img[t_img[0], t_img[1], 0] = Y_predict
    
    # write image
    ds = open_image(image_filename)
    write_image(out_classif, img, data_set=ds, gdal_dtype=None,
                transform=None, projection=None, driver_name=None,
                nb_col=None, nb_ligne=None, nb_band=1)