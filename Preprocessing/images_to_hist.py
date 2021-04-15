import os
import preprocessing_functions
import pandas as pd

path = 'data/jpeg/train/'
files = [image for image in os.listdir(path)]

hist_df = preprocessing_functions.files_to_hist_df(path, files)
meta_df = pd.read_csv('data/train.csv')
eda_df = meta_df.merge(hist_df, on='image_name')

print(eda_df.head())
eda_df.to_csv('data/eda.csv')
