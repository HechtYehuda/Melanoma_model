import pandas as pd
import numpy as np
from scipy import stats as ss
import matplotlib.pyplot as plt
import warnings
from pandas.core.common import SettingWithCopyWarning

# Preprocessing
def eda_preprocessing(path):
    '''
    Creates a dataframe of pixel intensity descriptive statistics.
    '''
    warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
    df = pd.read_csv(path, index_col=0)
    
    hist_df = df.loc[:,df.columns.str.contains('intensity')]

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
    eda_df['hist_range'] = hist_df.max(axis=1) - hist_df.min(axis=1)
    
    # IQR features
    hist_quartiles = hist_df.apply(np.percentile, axis=1, q=[25,75])
    eda_df['hist_iqr'] = hist_quartiles.apply(lambda x: x[1] - x[0])
    
    # Skew and kurtosis features
    eda_df['hist_skew'] = hist_df.skew(axis=1)
    eda_df['hist_kurtosis'] = hist_df.kurt(axis=1)

    # Mean, median features
    eda_df['hist_mean'] = hist_df.mean(axis=1)
    eda_df['hist_median'] = hist_df.median(axis=1)

    return eda_df

# ANOVA report
def anova_report(dataframe, grouping, comparison, aggregator):
    '''
    Prints F and P values based on a grouping feature and comparision feature.
    Thanks to @ayhan for his StackOverflow answer: https://stackoverflow.com/a/44066097/7287543
    '''
    for site in dataframe.groupby(grouping):
        hist_data = [i[1] for i in site[1].groupby(comparison)['hist_'+aggregator]]
        try:
            f_hist, p_hist = ss.f_oneway(*hist_data) 
            print(f'\nLocation: {site[0]}')
            print(f'     F value: {f_hist:{.3}}\n     p value: {p_hist:{.3}}')
        except:
            continue

# Compare histograms
def compare_histograms(dataframes, feature):
    df1, df2 = dataframes
    fig, ax = plt.subplots(2,1)
    ax1 = ax[0].hist(df1[f'hist_{feature}'])
    ax[0].set_title(f'Melanoma {feature}')
    ax2 = ax[1].hist(df2[f'hist_{feature}'])
    ax[1].set_title(f'No melanoma {feature}')
    plt.tight_layout()