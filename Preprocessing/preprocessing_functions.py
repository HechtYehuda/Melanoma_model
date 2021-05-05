import os
import tqdm
import concurrent.futures
import pandas as pd
import numpy as np
import PIL
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import pickle

# Files to hist df
def files_to_hist_df(path, files_list):
    '''
    Reads a list of image files and creates a pandas DataFrame of the histograms.
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

# Files to array pipeline and function
def keras_pipeline(file):
    TARGET_SIZE=(80,120)
    img = load_img(file, target_size=TARGET_SIZE)
    img_array = img_to_array(img)
    return img_array

def files_to_array(path, files_list):
    files = [path+file for file in files_list]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        img_map = list(tqdm.tqdm(executor.map(keras_pipeline, files)))
    return img_map

# Metadata preprocessing
def preprocess_meta(data):
    RAW_PATH = f'../data/{data}.csv'
    PROCESSED_PATH = f'../processed_data/{data}.csv'
    df = pd.read_csv(RAW_PATH)
    df.loc[df['age_approx'].isnull(), 'age_approx'] = 45.0
    df.loc[df['sex'].isnull(), 'sex'] = 'Unknown'
    df.loc[df['anatom_site_general_challenge'].isnull(), 'anatom_site_general_challenge'] = 'torso'
    
    sex = pd.get_dummies(df['sex'])
    location = pd.get_dummies(df['anatom_site_general_challenge'])
    new_df = pd.concat([df['age_approx'], sex, location], axis=1)
    if data == 'train':
        new_df = pd.concat([new_df, df['target']], axis=1)
    new_df.to_csv(PROCESSED_PATH)
    
# Transforms images to pickled data
def img_to_pickle(data):
    path = f'../processed_data/jpeg/{data}/'
    files_df = pd.read_csv(f'../data/{data}.csv')
    
    files = [file+'.jpg' for file in files_df.loc[:,'image_name']]
    img_map = files_to_array(path, files)
    
    img_array = np.array([i for i in img_map])
    
    pickle_path = f'../processed_data/{data}_img_array.pkl'
    with open(pickle_path, 'wb') as file:
        pickle.dump(img_array, file)
    
    with open(pickle_path, 'rb') as file:
        test = pickle.load(file)
    
    if np.array_equal(img_array, test):
        print('Pickling successful.')
    else:
        print('Pickling failed.')

# Denoising functions
def denoise_single_image(img_path):
    img = cv2.imread(f'../data/jpeg/{img_path}')
    dst = cv2.fastNlMeansDenoising(img, 10,10,7,21)
    cv2.imwrite(f'../processed_data/jpeg/{img_path}', dst)
    print(f'{img_path} denoised.')

def denoise(data):
    img_list = os.listdir(f'../data/jpeg/{data}')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tqdm.tqdm(executor.map(denoise_single_image, (f'{data}/{img_path}' for img_path in img_list)))

