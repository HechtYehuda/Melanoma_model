import os
import test
import pandas as pd
import preprocessing_functions

path = '../data/jpeg/train/'
files = [image for image in os.listdir(path)]

# Combine histogram and metadata
hist_df = preprocessing_functions.files_to_hist_df(path, files)
meta_df = pd.read_csv('../data/train.csv')
df = meta_df.merge(hist_df, on='image_name')

print(df.head())
df.to_csv('../processed_data/eda.csv')
