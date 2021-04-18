import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning

# Preprocessing
def eda_preprocessing(path):
    warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
    df = pd.read_csv(path, index_col=0)
    
    eda_df = df.iloc[:,:8]
    blue = df.loc[:,df.columns.str.contains('blue')]
    green = df.loc[:,df.columns.str.contains('green')]
    red = df.loc[:,df.columns.str.contains('red')]
    
    blue_eda = blue.copy()
    green_eda = green.copy()
    red_eda = red.copy()
    
    # Range features
    blue_eda['blue_range'] = blue.max(axis=1) - blue.min(axis=1)
    green_eda['green_range'] = green.max(axis=1) - green.min(axis=1)
    red_eda['red_range'] = red.max(axis=1) - red.min(axis=1)
    
    # IQR features
    blue_quartiles = blue.apply(np.percentile, axis=1, q=[25,75])
    blue_eda['blue_iqr'] = blue_quartiles.apply(lambda x: x[1] - x[0])
    
    green_quartiles = green.apply(np.percentile, axis=1, q=[25,75])
    green_eda['green_iqr'] = green_quartiles.apply(lambda x: x[1] - x[0])
    
    red_quartiles = red.apply(np.percentile, axis=1, q=[25,75])
    red_eda['red_iqr'] = red_quartiles.apply(lambda x: x[1] - x[0])

    # Skew and kurtosis features
    blue_eda['blue_skew'] = blue.skew(axis=1)
    green_eda['green_skew'] = green.skew(axis=1)
    red_eda['red_skew'] = red.skew(axis=1)

    blue_eda['blue_kurtosis'] = blue.kurt(axis=1)
    green_eda['green_kurtosis'] = green.kurt(axis=1)
    red_eda['red_kurtosis'] = red.kurt(axis=1)

    # Mean, median features
    blue_eda['blue_mean'] = blue.mean(axis=1)
    green_eda['green_mean'] = green.mean(axis=1)
    red_eda['red_mean'] = red.mean(axis=1)

    blue_eda['blue_median'] = blue.median(axis=1)
    green_eda['green_median'] = green.median(axis=1)
    red_eda['red_median'] = red.median(axis=1)

    # Compile final dataframe
    contains_blue = blue_eda.columns.str.contains('|'.join([str(i) for i in range(256)]))
    blue_eda = blue_eda.drop(blue_eda.columns[contains_blue], axis=1)
    
    contains_green = green_eda.columns.str.contains('|'.join([str(i) for i in range(256)]))
    green_eda = green_eda.drop(green_eda.columns[contains_green], axis=1)
    
    contains_red = red_eda.columns.str.contains('|'.join([str(i) for i in range(256)]))
    red_eda = red_eda.drop(red_eda.columns[contains_red], axis=1)
    
    eda_df = pd.concat([eda_df, blue_eda, green_eda, red_eda], axis=1)

    return eda_df
