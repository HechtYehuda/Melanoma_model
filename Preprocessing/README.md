# Preprocessing
This directory contains all functions and pipelines for preprocessing of images and metadata. 
* The _Imputation analysis_ notebook provides analysis and justification for imputation performed in the `preprocess_meta()` function. The `meta_preprocessing` script carries out this function on both the training and test metadata.
* The `images_to_hist` script creates a metadata-histogram CSV file for EDA. This is the pipeline for the _Multivariate EDA_ notebook in the _EDA_ directory.
* The `images_to_array` script runs the `files_to_array()` function on both the _train_ and _test_ JPEG image data. This transforms all images to a single Numpy array, and then pickles it. It utilizes the `concurrent.futures` module for multi-core processing, to improve speeds.
* The `image_preprocessing` script runs the `denoise` function across the raw JPEG data. It uses multicore processing for increased speeds. Note that this script takes many hours to run, despite optimizations.