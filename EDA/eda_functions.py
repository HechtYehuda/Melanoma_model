import pandas as pd
import numpy as np
from scipy import stats as ss
import warnings
from pandas.core.common import SettingWithCopyWarning

# Preprocessing
def eda_preprocessing(path):
    '''
    Creates a dataframe of pixel intensity descriptive statistics.
    '''
    warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
    df = pd.read_csv(path, index_col=0)
    
    blue = df.loc[:,df.columns.str.contains('blue')]
    green = df.loc[:,df.columns.str.contains('green')]
    red = df.loc[:,df.columns.str.contains('red')]

    meta_features = [
            'image_name',
            'patient_id',
            'sex',
            'age_approx',
            'anatom_site_general_challenge',
            'target'
            ]
    eda_df = df.loc[:,meta_features]
    
    # Range features
    eda_df['blue_range'] = blue.max(axis=1) - blue.min(axis=1)
    eda_df['green_range'] = green.max(axis=1) - green.min(axis=1)
    eda_df['red_range'] = red.max(axis=1) - red.min(axis=1)
    
    # IQR features
    blue_quartiles = blue.apply(np.percentile, axis=1, q=[25,75])
    eda_df['blue_iqr'] = blue_quartiles.apply(lambda x: x[1] - x[0])
    
    green_quartiles = green.apply(np.percentile, axis=1, q=[25,75])
    eda_df['green_iqr'] = green_quartiles.apply(lambda x: x[1] - x[0])
    
    red_quartiles = red.apply(np.percentile, axis=1, q=[25,75])
    eda_df['red_iqr'] = red_quartiles.apply(lambda x: x[1] - x[0])

    # Skew and kurtosis features
    eda_df['blue_skew'] = blue.skew(axis=1)
    eda_df['green_skew'] = green.skew(axis=1)
    eda_df['red_skew'] = red.skew(axis=1)

    eda_df['blue_kurtosis'] = blue.kurt(axis=1)
    eda_df['green_kurtosis'] = green.kurt(axis=1)
    eda_df['red_kurtosis'] = red.kurt(axis=1)

    # Mean, median features
    eda_df['blue_mean'] = blue.mean(axis=1)
    eda_df['green_mean'] = green.mean(axis=1)
    eda_df['red_mean'] = red.mean(axis=1)

    eda_df['blue_median'] = blue.median(axis=1)
    eda_df['green_median'] = green.median(axis=1)
    eda_df['red_median'] = red.median(axis=1)

    return eda_df

# ANOVA report
def anova_report(dataframe, grouping, comparison, aggregator):
    '''
    Prints F and P values of all three channels based on a grouping feature and comparision feature.
    Thanks to @ayhan for his StackOverflow answer: https://stackoverflow.com/a/44066097/7287543
    '''
    for site in dataframe.groupby(grouping):
        blue_data = [i[1] for i in site[1].groupby(comparison)['blue_'+aggregator]]
        green_data = [i[1] for i in site[1].groupby(comparison)['green_'+aggregator]]
        red_data = [i[1] for i in site[1].groupby(comparison)['red_'+aggregator]]
        try:
            f_blue, p_blue = ss.f_oneway(*blue_data)
            f_green, p_green = ss.f_oneway(*green_data)
            f_red, p_red = ss.f_oneway(*red_data)
    
            print(f'\nLocation: {site[0]}')
            print(f'  Channel: Blue\n     F value: {f_blue:{.3}}\n     p value: {p_blue:{.3}}')
            print(f'  Channel: Green\n     F value: {f_green:{.3}}\n     p value: {p_green:{.3}}')
            print(f'  Channel: Red\n     F value: {f_red:{.3}}\n     p value: {p_red:{.3}}')
    
        except:
            continue
