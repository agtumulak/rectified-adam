#!/usr/bin/env python
import pandas as pd

df = pd.read_csv('data/amex-nyse-nasdaq-stock-histories/history_60d.csv')
df = df.loc[~df.duplicated(subset=['date', 'symbol']),:] # drop duplicates
df = df.pivot(index='date', columns='symbol', values='close')
print(df)
