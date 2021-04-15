import modin.pandas as pd
import numpy as np

# Preprocessing
def eda_preprocessing(dataframe):
    client = Client()
    df = pd.read_csv('../data/eda.csv', index_col=0)
    
    meta = df.iloc[:,:8]
    blue = df.loc[:,df.columns.str.contains('blue')]
    green = df.loc[:,df.columns.str.contains('green')]
    red = df.loc[:,df.columns.str.contains('red')]
    
    blue_eda = pd.concat([meta, blue], axis=1)
    green_eda = pd.concat([meta, green], axis=1)
    red_eda = pd.concat([meta, red], axis=1)
    
    # Range features
    blue_eda['range'] = blue.max(axis=1) - blue.min(axis=1)
    green_eda['range'] = green.max(axis=1) - green.min(axis=1)
    red_eda['range'] = red.max(axis=1) - red.min(axis=1)
    
    # IQR features
    blue_quartiles = blue.apply(np.percentile, axis=1, q=[25,75])
    blue_eda['iqr'] = blue_quartiles.apply(lambda x: x[1] - x[0])
    
    green_quartiles = green.apply(np.percentile, axis=1, q=[25,75])
    green_eda['iqr'] = green_quartiles.apply(lambda x: x[1] - x[0])
    
    red_quartiles = red.apply(np.percentile, axis=1, q=[25,75])
    red_eda['iqr'] = red_quartiles.apply(lambda x: x[1] - x[0])

    return blue_eda, green_eda, red_eda
