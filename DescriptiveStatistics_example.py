import sys
import pandas as pd
import numpy as np
from DescriptiveStatistics import DescriptiveStatistics
del sys.modules['DescriptiveStatistics']
from DescriptiveStatistics import DescriptiveStatistics
ds = DescriptiveStatistics()
DF = pd.read_csv('data.csv')

df_ds = pd.DataFrame()
for tag in DF.columns:
    x   = DF[tag].copy()
    print(tag)
    df_ds_temp = ds.plot(x,tag,prefix='stat')
    df_ds = df_ds.append(df_ds_temp, ignore_index=True)

df_ds.to_csv('summary_All.csv', sep=',' ,index=False)
