import os
import preprocessing_functions
import pandas as pd

path = 'data/jpeg/train/'
files = [image for image in os.listdir(path)]

# Combine histogram and metadata
hist_df = preprocessing_functions.files_to_hist_df(path, files)
meta_df = pd.read_csv('data/train.csv')
df = meta_df.merge(hist_df, on='image_name') # Necessary for correct indexing

# Rename channels with color names
numbers = df.loc[:,'0':'255'].columns
rename_1 = dict(zip(df.loc[:,'256':'511'].columns, numbers))
rename_2 = dict(zip(df.loc[:,'512':'767'].columns, numbers))

# Blue
blue = df.loc[:,'0':'255'].add_prefix('blue_')

# Green
df.loc[:,'256':'511'].rename(rename_1, axis=1, inplace=True)
green = df.loc[:,'0':'255'].add_prefix('green_')

# Red
df.loc[:,'512':'767'].rename(rename_2, axis=1, inplace=True)
red = df.loc[:,'0':'255'].add_prefix('red_')

df = pd.concat([df.iloc[:,0:9],blue,green,red], axis=1)

df.to_csv('data/eda.csv')
