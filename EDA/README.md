# EDA
This directory contains the notebooks and functions for EDA.

The _Univariate EDA_ notebook examines the metadata on a univariate level. The _Multivariate EDA_ notebook examines the data on a more complex level, through groupings of patients, locations of images on the body, and target diagnosis.

`eda_functions.py` contains a preprocessing function, creating the descriptive statistics used in the EDA. All statistics examine the grayscale intensity histograms of all images, matched to the corresponding metadata.
