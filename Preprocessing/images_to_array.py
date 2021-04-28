import preprocessing_functions
import pandas as pd
import numpy as np
import pickle

path = '../data/jpeg/train/'
meta_df = pd.read_csv('../data/train.csv')

files = [file+'.jpg' for file in meta_df.loc[:,'image_name']]
img_map = preprocessing_functions.files_to_array(path, files)

img_array = np.array([i for i in img_map])

pickle_path = '../processed_data/img_array.pkl'
with open(pickle_path, 'wb') as file:
    pickle.dump(img_array, file)

with open(pickle_path, 'rb') as file:
    test = pickle.load(file)

if np.array_equal(img_array, test):
    print('Pickling successful.')
else:
    print('Pickling failed.')
