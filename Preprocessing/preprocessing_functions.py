import tqdm
import pandas as pd
import numpy as np
import cv2

# Files to hist df
def files_to_hist_df(path, files_list):
    '''
    Reads a list of image files and creates a pandas DataFrame of the three-channel histograms.
    '''
    # Instantiate arrays
    files = [path+file for file in files_list]
    hist_array = np.zeros(256)
    image_names = []

    for file in tqdm.tqdm(files_list):
        full_file = path+file
        image_names.append(file.split('.')[0])
        img = cv2.imread(full_file, 1)
        # Calculate histograms
        hist = cv2.calcHist(img, channels=[0], mask=None, histSize=[256], ranges=[0,256]).flatten()

        # Append histograms to arrays
        hist_array = np.vstack([hist_array, hist])

    # Remove instantiating zeros
    hist_array = hist_array[1:]

    # Create dataframes
    hist_df = pd.DataFrame(hist_array).add_prefix('intensity_')
    hist_df['image_name'] = image_names

    return hist_df
