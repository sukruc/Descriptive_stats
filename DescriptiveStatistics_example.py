import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DescriptiveStatistics import DescriptiveStatistics
del sys.modules['DescriptiveStatistics']
from DescriptiveStatistics import DescriptiveStatistics

# Create some toy data to play with:
x1 = np.random.randn(200)
x2 = np.random.randn(200)

# Add some disortion:
def distorter(x,window_len,mu,sigma):
    for i in range(len(x)-window_len):
        x[i:i+window_len] = x[i:i+window_len]*np.random.normal(mu,sigma,window_len)
    return x

# Create a DataFrame
x1 = distorter(x1,3,2,5)
x2 = distorter(x2,4,1,3)
df = pd.DataFrame({'x1':x1,'x2':x2})

ds = DescriptiveStatistics()

df_ds = pd.DataFrame()
for tag in df.columns:
    x = df[tag].copy()
    print(tag)
    df_ds_temp = ds.plot(x,tag,prefix='stats')
    df_ds = df_ds.append(df_ds_temp, ignore_index=True)

df_ds.to_csv('summary_All_stats.csv', sep=',' ,index=False)
