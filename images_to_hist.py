import os
import project_functions
import pickle

path = 'data/jpeg/train/'
files = [path+image for image in os.listdir(path)]

hist_df = project_functions.files_to_hist_df(files)
with open('data/dataframes/hist_df.pkl', 'wb') as file:
    pickle.dump(hist_df, file)
