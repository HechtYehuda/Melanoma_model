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
    blue_array = np.zeros(256)
    green_array = np.zeros(256)
    red_array = np.zeros(256)
    image_names = []

    for file in tqdm.tqdm(files_list):
        full_file = path+file
        image_names.append(file.split('.')[0])
        img = cv2.imread(full_file, 3)
        # Calculate histograms
        blue = cv2.calcHist(img, channels=[0], mask=None, histSize=[256], ranges=[0,256]).flatten()
        green = cv2.calcHist(img, channels=[1], mask=None, histSize=[256], ranges=[0,256]).flatten()
        red = cv2.calcHist(img, channels=[2], mask=None, histSize=[256], ranges=[0,256]).flatten()
        # Append histograms to arrays
        blue_array = np.vstack([blue_array, blue])
        green_array = np.vstack([green_array, green])
        red_array = np.vstack([red_array, red])

    # Remove instantiating zeros
    blue_array = blue_array[1:]
    green_array = green_array[1:]
    red_array = red_array[1:]

    # Create dataframes
    hist_df = pd.DataFrame(np.hstack([blue_array, green_array, red_array]))
    hist_df['image_name'] = image_names

    return hist_df
