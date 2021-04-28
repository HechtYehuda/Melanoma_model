import pandas as pd
RAW_PATH = '../data/train.csv'
PROCESSED_PATH = '../processed_data/train.csv'
df = pd.read_csv(RAW_PATH)
df.loc[df['age_approx'].isnull(), 'age_approx'] = 45.0
df.loc[df['sex'].isnull(), 'sex'] = 'Unknown'
df.loc[df['anatom_site_general_challenge'].isnull(), 'anatom_site_general_challenge'] = 'torso'
df.to_csv(PROCESSED_PATH)
