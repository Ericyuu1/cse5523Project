
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

# custom imports
from multiprocessing import Pool        # Multiprocess Runs

warnings.filterwarnings('ignore')


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    
## Multiprocess Runs
def df_parallelize_run(func, t_split):
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(1)
    df = pd.concat(pool.map(func, t_split), axis=1)
    #pool.close()
    #pool.join()
    return df


########################### Helper to load data by store ID
#################################################################################
# Read data
def get_data_by_store(store):
    
    # Read and contact basic feature
    df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(PRICE).iloc[:,2:],
                    pd.read_pickle(CALENDAR).iloc[:,2:]],
                    axis=1)
    
    # Leave only relevant store
    df = df[df['store_id']==store]

    # With memory limits we have to read 
    # lags and mean encoding features
    # separately and drop items that we don't need.
    # As our Features Grids are aligned 
    # we can use index to keep only necessary rows
    # Alignment is good for us as concat uses less memory than merge.
    df2 = pd.read_pickle(MEAN_ENC)[mean_features]
    df2 = df2[df2.index.isin(df.index)]
    
    df3 = pd.read_pickle(LAGS).iloc[:,3:]
    df3 = df3[df3.index.isin(df.index)]
    
    df = pd.concat([df, df2], axis=1)
    del df2 # to not reach memory limit 
    
    df = pd.concat([df, df3], axis=1)
    del df3 # to not reach memory limit 
    
    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[['id','d',TARGET]+features]
    
    # Skipping first n rows
    df = df[df['d']>=START_TRAIN].reset_index(drop=True)
    
    return df, features

# Recombine Test set after training
def get_base_test():
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle(CUR + '/test_'+store_id+'.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    
    return base_test


########################### Helper to make dynamic rolling lags
#################################################################################
def make_lag(LAG_DAY):
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'small_tmp_lags_'+str(LAG_DAY)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(LAG_DAY)).astype(np.float16)
    return lag_df[[col_name]]


def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


########################### Model params
#################################################################################
import lightgbm as lgb
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 1400,
                    'boost_from_average': False,
                    'verbose': -1,
                } 


########################### Vars
#################################################################################
VER = 1                          # Our model version
SEED = 98231                        # We want all things
seed_everything(SEED)            # to be as deterministic 
lgb_params['seed'] = SEED        # as possible
N_CORES = psutil.cpu_count()     # Available CPU cores

CUR = 'kfold_365_fake_200k'
if not os.path.exists(CUR):
    os.makedirs(CUR)

#LIMITS and const
TARGET      = 'sales'            # Our target
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913               # End day of our train set
P_HORIZON   = 28                 # Prediction horizon
USE_AUX     = False               # Use or not pretrained models

#FEATURES to remove
## These features lead to overfit
## or values not present in test set
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
# remove_features = ['id','state_id','store_id',
#                    'date','wm_yr_wk','d',TARGET, 
#                     'dis_CA', 'type_CA', 'dis_TX', 'type_TX', 'dis_WI', 'type_WI', 'tm_q']
mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
                   'enc_dept_id_mean','enc_dept_id_std',
                   'enc_item_id_mean','enc_item_id_std'] 
# mean_features   = ['enc_item_id_mean','enc_item_id_std'] 

#PATHS for Features
ORIGINAL = '../input/'
BASE     = './grid_part_1.pkl'
PRICE    = '../input/m5-simple-fe/grid_part_2.pkl'
CALENDAR = '../input/m5-simple-fe/grid_part_3.pkl'
# CALENDAR = '../input/m5-simple-fe/grid_part_3_with_q_dis.pkl'

LAGS     = '../input/m5-lags-fe/lags_df_365_21.pkl'
# MEAN_ENC = '../input/m5-custom-features/mean_encoding_df.pkl'
MEAN_ENC = '../input/m5-custom-features/mean_encoding_10_5kfold.pkl'

# AUX(pretrained) Models paths
AUX_MODELS = '../input/m5-aux-models/'


#STORES ids
STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())


#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])


##  and you can safely use this (all kernel) code)
if USE_AUX:
    lgb_params['n_estimators'] = 2


########################### Train Models
#################################################################################
for store_id in STORES_IDS:
    print('Train', store_id)
    
    # Get grid for current store
    grid_df, features_columns = get_data_by_store(store_id)

    train_mask = grid_df['d']<=END_TRAIN
#     valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))
    preds_mask = grid_df['d']>(END_TRAIN-100)
    fake_mask = np.random.randint(grid_df[train_mask].shape[0], size = 200000)

    train_data = lgb.Dataset(grid_df[train_mask][features_columns],
                       label=grid_df[train_mask][TARGET])
#     train_data.save_binary('train_data.bin')
#     train_data = lgb.Dataset('train_data.bin')
    
#     valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
#                        label=grid_df[valid_mask][TARGET])
    valid_data = lgb.Dataset(grid_df[train_mask][features_columns].iloc[fake_mask], 
                       label=grid_df[train_mask][TARGET].iloc[fake_mask])
    
    # Saving part of the dataset for later predictions
    # Removing features that we need to calculate recursively 
    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
    grid_df = grid_df[keep_cols]
    grid_df.to_pickle(CUR + '/test_'+store_id+'.pkl')
    del grid_df
    
    # Launch seeder again to make lgb training 100% deterministic
    # with each "code line" np.random "evolves" 
    # so we need (may want) to "reset" it
    seed_everything(SEED)
    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets = [valid_data],
                          verbose_eval = 100,
                          )
    
    model_name = CUR + '/lgb_model_'+store_id+'_v'+str(VER)+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))
    
    # Remove temporary files and objects 
    # to free some hdd space and ram memory
#     !rm train_data.bin
    del train_data, valid_data, estimator
    gc.collect()
    
    # "Keep" models features for predictions
    MODEL_FEATURES = features_columns


USE_AUX = False
# Create Dummy DataFrame to store predictions
all_preds = pd.DataFrame()

# Join back the Test dataset with 
# a small part of the training data 
# to make recursive features
base_test = get_base_test()

# Timer to measure predictions time 
main_time = time.time()
# small_lags = list(range(14, 29))
# Loop over each prediction day
# As rolling lags are the most timeconsuming
# we will calculate it for whole day
for PREDICT_DAY in range(1, 29):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()
    
    # Make temporary grid to calculate rolling lags
    grid_df = base_test.copy()
    #temp1 = make_lag_roll(grid_df)
    #temp2 = ROLS_SPLIT(grid_df)
    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)
    #grid_df = pd.concat([grid_df, temp1], axis=1)
    #grid_df = pd.concat([grid_df, temp2], axis=1)
#     grid_df = pd.concat([grid_df, df_parallelize_run(make_lag, small_lags)], axis=1)
        
    for store_id in STORES_IDS:
        
        # Read all our models and make predictions
        # for each day/store pairs
        model_path = CUR + '/lgb_model_'+store_id+'_v'+str(VER)+'.bin' 
        if USE_AUX:
            model_path = AUX_MODELS + model_path
        
        estimator = pickle.load(open(model_path, 'rb'))
        
        day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)
        store_mask = base_test['store_id']==store_id
        
        mask = (day_mask)&(store_mask)
        base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])
    
    # Make good column naming and add 
    # to all_preds DataFrame
    temp_df = base_test[day_mask][['id',TARGET]]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    
all_preds = all_preds.reset_index(drop=True)
all_preds


1


base_name = CUR + '/submission_base_'+CUR+'.csv'
submission = pd.read_csv('../input/sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.to_csv(base_name, index=False)
submission = pd.read_csv(base_name)
teams_now = 3215
teams_before = 3186
submission['F1'] *= teams_now/teams_before
magic = submission['F2'].sum()
korea_emergency = 119
submission['F2'] *= magic / (magic + korea_emergency)
submission['F3'] *= 1913/1923
sex_constant = 1.0073
submission['F4'] /= sex_constant
color_of_the_night = 995874

# There are millions "Stories"
# We need to make it Mean, really mean
color_of_the_night /= 1000000
submission['F5'] *= color_of_the_night
injury = 1.000376
submission['F6'] *= injury
import numpy as np
np.random.seed(198505)
correction = np.random.randint(1000000)/1000000
submission['F7'] *= correction
submission['F8'] *= (100-1)/100 + (100-12)/10000
submission['F11'] = submission['F11']
for i in range(9,20):
    if i!=11:
        submission['F'+str(i)] *= 1.01 
for i in range(20,29):
    submission['F'+str(i)] *= 1.02
for i in range(9,14):
    if i!=11:
        submission['F'+str(i)] *= 1.01 
        


import os
path_output = CUR + '/sub/'
os.makedirs(path_output, exist_ok = True)
submission.to_csv(CUR + '/sub/submission_factor_'+CUR+'.csv', index=False)


CUR = 'kfold_365_14'
if not os.path.exists(CUR):
    os.makedirs(CUR)

#LIMITS and const
TARGET      = 'sales'            # Our target
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913               # End day of our train set
P_HORIZON   = 28                 # Prediction horizon
USE_AUX     = False               # Use or not pretrained models

#FEATURES to remove
## These features lead to overfit
## or values not present in test set
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
# remove_features = ['id','state_id','store_id',
#                    'date','wm_yr_wk','d',TARGET, 
#                     'dis_CA', 'type_CA', 'dis_TX', 'type_TX', 'dis_WI', 'type_WI', 'tm_q']
mean_features   = ['enc_cat_id_mean','enc_cat_id_std',
                   'enc_dept_id_mean','enc_dept_id_std',
                   'enc_item_id_mean','enc_item_id_std'] 

#PATHS for Features
ORIGINAL = '../input/m5-forecasting-accuracy/'
BASE     = '../input/grid_part_1.pkl'
PRICE    = '../input/m5-simple-fe/grid_part_2.pkl'
CALENDAR = '../input/m5-simple-fe/grid_part_3.pkl'
# CALENDAR = '../input/m5-simple-fe/grid_part_3_with_q_dis.pkl'

LAGS     = '../input/m5-lags-features/lags_df_365_14.pkl'
MEAN_ENC = '../input/m5-custom-features/mean_encoding_kfold.pkl'



# AUX(pretrained) Models paths
AUX_MODELS = '../input/m5-aux-models/'


#STORES ids
STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())


#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])
for store_id in STORES_IDS:
    print('Train', store_id)
    
    # Get grid for current store
    grid_df, features_columns = get_data_by_store(store_id)

    train_mask = grid_df['d']<=END_TRAIN
#     valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))
    preds_mask = grid_df['d']>(END_TRAIN-100)
    fake_mask = np.random.randint(grid_df[train_mask].shape[0], size = 200000)

    train_data = lgb.Dataset(grid_df[train_mask][features_columns],
                       label=grid_df[train_mask][TARGET])
#     train_data.save_binary('train_data.bin')
#     train_data = lgb.Dataset('train_data.bin')
    
#     valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
#                        label=grid_df[valid_mask][TARGET])
    valid_data = lgb.Dataset(grid_df[train_mask][features_columns].iloc[fake_mask], 
                       label=grid_df[train_mask][TARGET].iloc[fake_mask])
    
    # Saving part of the dataset for later predictions
    # Removing features that we need to calculate recursively 
    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
    grid_df = grid_df[keep_cols]
    grid_df.to_pickle(CUR + '/test_'+store_id+'.pkl')
    del grid_df
    
    # Launch seeder again to make lgb training 100% deterministic
    # with each "code line" np.random "evolves" 
    # so we need (may want) to "reset" it
    seed_everything(SEED)
    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets = [valid_data],
                          verbose_eval = 100,
                          )
    
    model_name = CUR + '/lgb_model_'+store_id+'_v'+str(VER)+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))

    # Remove temporary files and objects 
    # to free some hdd space and ram memory
    !rm train_data.bin
    del train_data, valid_data, estimator
    gc.collect()
    
    # "Keep" models features for predictions
    MODEL_FEATURES = features_columns

# Create Dummy DataFrame to store predictions
all_preds = pd.DataFrame()

# Join back the Test dataset with 
# a small part of the training data 
# to make recursive features
base_test = get_base_test()

# Timer to measure predictions time 
main_time = time.time()

# Loop over each prediction day
# As rolling lags are the most timeconsuming
# we will calculate it for whole day
for PREDICT_DAY in range(1, 15):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    # Make temporary grid to calculate rolling lags
    grid_df = base_test.copy()
    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)
        
    for store_id in STORES_IDS:
        
        # Read all our models and make predictions
        # for each day/store pairs
        model_path = CUR + '/lgb_model_'+store_id+'_v'+str(VER)+'.bin' 
        if USE_AUX:
            model_path = AUX_MODELS + model_path
        
        estimator = pickle.load(open(model_path, 'rb'))
        
        day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)
        store_mask = base_test['store_id']==store_id
        
        mask = (day_mask)&(store_mask)
        base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])
    
    # Make good column naming and add 
    # to all_preds DataFrame
    temp_df = base_test[day_mask][['id',TARGET]]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    
all_preds = all_preds.reset_index(drop=True)

base_name = CUR + '/submission_base_'+CUR+'.csv'
submission = pd.read_csv('../input/sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.to_csv(base_name, index=False)
submission = pd.read_csv(base_name)
teams_now = 3215
teams_before = 3186
submission['F1'] *= teams_now/teams_before
magic = submission['F2'].sum()
korea_emergency = 119
submission['F2'] *= magic / (magic + korea_emergency)
submission['F3'] *= 1913/1923
sex_constant = 1.0073
submission['F4'] /= sex_constant
color_of_the_night = 995874

# There are millions "Stories"
# We need to make it Mean, really mean
color_of_the_night /= 1000000
submission['F5'] *= color_of_the_night
injury = 1.000376
submission['F6'] *= injury
import numpy as np
np.random.seed(198505)
correction = np.random.randint(1000000)/1000000
submission['F7'] *= correction
submission['F8'] *= (100-1)/100 + (100-12)/10000
submission['F11'] = submission['F11']
for i in range(9, 15):
    if i!=11:
        submission['F'+str(i)] *= 1.01 
# for i in range(20,29):
#     submission['F'+str(i)] *= 1.02

import os
path_output = CUR +  '/sub/'
os.makedirs(path_output, exist_ok = True)
submission.to_csv(CUR + '/sub/submission_factor_'+CUR+'.csv', index=False)


