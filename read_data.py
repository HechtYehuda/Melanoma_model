import numpy as np
import dask.array as da
from pydicom import dcmread
import glob

PATH = 'train'
dcm_files = glob.glob(PATH+'/*.dcm')

dask_arrays = []
for dcm_file in dcm_files:
    dcm = dcmread(dcm_file)
    img = dcm.pixel_array
    array = da.from_array(img, chunks=1000)
    dask_arrays.append(array)

X = da.stack(dask_arrays)
# da.compute(X)
# print(X.shape)
