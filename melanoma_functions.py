# Files to hist df
def files_to_hist_df(files_list):
    '''
    Reads a list of image files and creates a pandas DataFrame of the three-channel histograms.
    '''
    # Instantiate arrays
    blue_array = np.zeros(256)
    green_array = np.zeros(256)
    red_array = np.zeros(256)

    for file in files:
        img = cv2.imread(file, 3)
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
    red_df = pd.DataFrame(red_array).add_prefix('red_')
    blue_df = pd.DataFrame(blue_array).add_prefix('blue_')
    green_df = pd.DataFrame(green_array).add_prefix('green_')
    hist_df = pd.concat([red_df, blue_df, green_df], axis=1)
