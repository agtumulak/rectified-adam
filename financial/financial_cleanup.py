#!/usr/bin/env python
from ipdb import set_trace as st
import os
import os.path
import pandas as pd
from tqdm import tqdm

min_days = 9000 # Required number of days in symbol
full_df = None
directory = 'full_history'

for filename in tqdm(os.listdir(directory)):
    df = pd.read_csv(os.path.join(directory, filename), index_col='date')
    # daily_return = (df['adjclose'] - df['open']) / df['open']
    daily_return = df['open'].diff(periods=-1)[:-1]
    daily_return.name = filename.split('.')[0]
    n_days = len(daily_return)
    if n_days < min_days:
        continue
    elif full_df is None:
        full_df = daily_return
    else:
        try:
            full_df = pd.concat([full_df, daily_return], axis=1, join='inner')
        except:
            print('Skipping {}'.format(filename))
            continue
full_df.to_hdf('day_returns.hdf5', 'returns')
full_df.cov().to_hdf('day_cov.hdf5', 'cov')
full_df.mean().to_hdf('day_mean.hdf5', 'mean')
st()
pass
