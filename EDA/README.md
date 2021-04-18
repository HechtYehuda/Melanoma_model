# EDA
This directory contains the notebooks and functions for EDA.

The _Univariate EDA_ notebook examines the metadata on a univariate basis. The _Multivariate EDA_ examines the data on a more complex basis, through groupings of patients, locations of images on the body, and target diagnosis.

`eda_functions.py` contains a preprocessing function, creating the descriptive statistics used in the EDA. All statistics examine the three-channel intensity histograms of all images, matched to the corresponding metadata.
