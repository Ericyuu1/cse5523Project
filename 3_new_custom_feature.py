
import numpy as np 
import pandas as pd 
import os, sys, gc, warnings, psutil, random
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


grid_df = pd.read_pickle('grid_part_1.pkl')
grid_df['sales'][grid_df['d']>(1913)] = np.nan
train = grid_df[grid_df['d']<=(1913)]
test = grid_df[grid_df['d']>(1913)]
cols = ['item_id', 'cat_id', 'dept_id']
n_folds = 10
n_inner_folds = 5
target = 'sales'
for feature in cols:
    target_encoding = 'enc_' + feature + '_mean'
    
    train[target_encoding] = np.nan
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2020)
    split = 0
    default_mean = train[target].mean()

    for infold, oof in kf.split(train[feature]):
        tr, val = train.iloc[infold], train.iloc[oof]
        inner_kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=2020)
        temp = train.iloc[infold]
        inner_default_mean = tr[target].mean()
        for inner_infold, inner_oof in inner_kf.split(temp):
            # inner out of fold mean
            temp_tr, temp_val = temp.iloc[inner_infold], temp.iloc[inner_oof]
            temp.loc[temp.index[inner_oof], target_encoding] = temp_val[feature].map(temp_tr.groupby([feature])[target].mean())
            temp[target_encoding].fillna(inner_default_mean, inplace=True)
        train.loc[train.index[oof], target_encoding] = val[feature].map(temp.groupby([feature])[target_encoding].mean())
        train[target_encoding].fillna(default_mean, inplace=True)
    test[target_encoding] = test[feature].map(train.groupby([feature])[target_encoding].mean())
    
    target_encoding = 'enc_' + feature + '_std'
    train[target_encoding] = np.nan
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2020)
    split = 0
    default_mean = train[target].std()

    for infold, oof in kf.split(train[feature]):
        tr, val = train.iloc[infold], train.iloc[oof]
        inner_kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=2020)
        temp = tr.copy()
        inner_default_mean = tr[target].std()
        for inner_infold, inner_oof in inner_kf.split(tr):
            # inner out of fold mean
            temp_tr, temp_val = temp.iloc[inner_infold], temp.iloc[inner_oof]
            temp.loc[temp.index[inner_oof], target_encoding] = temp_val[feature].map(temp_tr.groupby([feature])[target].std())
            temp[target_encoding].fillna(inner_default_mean, inplace=True)
        train.loc[train.index[oof], target_encoding] = val[feature].map(temp.groupby([feature])[target_encoding].mean())
        train[target_encoding].fillna(default_mean, inplace=True)
    test[target_encoding] = test[feature].map(train.groupby([feature])[target_encoding].mean())
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    



grid_df = pd.concat([train, test])
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
grid_df = reduce_mem_usage(grid_df)


grid_df


keep_cols = ['id', 'd', 'enc_item_id_mean', 'enc_item_id_std',
       'enc_cat_id_mean', 'enc_cat_id_std', 'enc_dept_id_mean', 'enc_dept_id_std']
grid_df = grid_df[keep_cols]
grid_df


import os
path_output = '../input/m5-custom-features/'
os.makedirs(path_output, exist_ok = True)

grid_df.to_pickle('../input/m5-custom-features/mean_encoding_10_5kfold.pkl')


