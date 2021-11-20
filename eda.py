import os
from glob import glob
import pandas as pd

DATA_DIR = 'data'

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
pred_df = pd.read_csv('')
# pred_df['disease_code'].value_counts()

print(train_df.head())
print(train_df.shape)
# 250 4
print(test_df.shape)
# 4750 2

print(train_df['disease_code'].value_counts())
print(pred_df['disease_code'].value_counts())
# print(test_df['disease_code'].value_counts())
"""
0    106
1     46
2     30
3     29
4     17
5     12
6     10
"""
"""
0    1745
1     846
2     640
3     611
4     399
5     303
6     206
"""
