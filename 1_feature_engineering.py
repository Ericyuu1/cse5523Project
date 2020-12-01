
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
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


## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


########################### Vars ####################################
TARGET = 'sales'         # Our main target
END_TRAIN = 1913         # Last day in train set
MAIN_INDEX = ['id','d']  # We can identify item by these columns


########################### Load Data #######################################
print('Load Main Data')

# Here are reafing all our data without any limitations and dtype modification
train_df = pd.read_csv('../input/sales_train_validation.csv')
prices_df = pd.read_csv('../input/sell_prices.csv')
calendar_df = pd.read_csv('../input/calendar.csv')


########################### Make Grid ###################################################
print('Create Grid')

# We can tranform horizontal representation to vertical "view"
# Our "index" will be 'id','item_id','dept_id','cat_id','store_id','state_id' and labels are 'd_' coulmns

index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
# 原本每一行是一个商品，在所有不同日子里的销售情况
# 将其转换成单个商品单日的销售量，也就是我们的预测目标(原来的列名d_1这种日期会变成行的索引)
grid_df = pd.melt(train_df, 
                  id_vars = index_columns, 
                  var_name = 'd',
                  value_name = TARGET)

# If we look on train_df we se that we don't have a lot of traning rows but each day can provide more train data
print('Train rows change from:', len(train_df), 'to', len(grid_df))

# To be able to make predictions we need to add "test set" to our grid
# 将测试集也加进去
add_grid = pd.DataFrame()
for i in range(1,29):
    temp_df = train_df[index_columns]
    temp_df = temp_df.drop_duplicates()
    temp_df['d'] = 'd_'+ str(END_TRAIN+i)
    temp_df[TARGET] = np.nan
    add_grid = pd.concat([add_grid,temp_df])

grid_df = pd.concat([grid_df,add_grid])
grid_df = grid_df.reset_index(drop=True)

# Remove some temoprary DFs, and train_df
del temp_df, add_grid, train_df

# You don't have to use df = df construction, you can use inplace=True instead.
# like this  grid_df.reset_index(drop=True, inplace=True)

# Let's check our memory usage
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

# We can free some memory  by converting "strings" to categorical
# it will not affect merging and  we will not lose any valuable data
# 通过将strings类型的数据变成categorical可以减少一些内存占用
for col in index_columns:
    grid_df[col] = grid_df[col].astype('category')

# Let's check again memory usage
print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))


########################### Product Release date ######################################################
print('Release week')

# It seems that leadings zero values in each train_df item row are not real 0 sales but mean
# absence for the item in the store we can save some memory by removing such zeros

# Prices are set by week so it will have not very accurate release week
# 对于单个商店单个商品的单周价格，应该只有一个，这里去重 取min
# groupby的时候取了单个商品所有周ID的最小值，结果就是该商品价格最早发布的week id
release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
release_df.columns = ['store_id','item_id','release']

# Now we can merge release_df
grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
del release_df

# We want to remove some "zeros" rows from grid_df 
# to do it we need wm_yr_wk column, let's merge partly calendar_df to have it
grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
                      
# Now we can cutoff some rows and safe memory 
# 当商品的上架日期大于商品的出售所属的日期时，代表该商品实际上没有在售卖
# 所以仅取那些真实在售卖的产品
grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
grid_df = grid_df.reset_index(drop=True)

# Let's check our memory usage
print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

# Should we keep release week as one of the features?
# Only good CV can give the answer. Let's minify the release values.
# Min transformation will not help here as int16 -> Integer (-32768 to 32767)
# and our grid_df['release'].max() serves for int16 but we have have an idea 
# how to transform  other columns in case we will need it
# 对release列transform，减去其最小值
grid_df['release'] = grid_df['release'] - grid_df['release'].min()
grid_df['release'] = grid_df['release'].astype(np.int16)

# Let's check again memory usage
print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))


########################### Save part 1 ###################################
print('Save Part 1')
import os
path_output = '../input/m5-simple-fe/'
os.makedirs(path_output, exist_ok = True)

# We have our BASE grid ready and can save it as pickle file for future use (model training)
grid_df.to_pickle('../input/m5-simple-fe/grid_part_1.pkl')
print('Size:', grid_df.shape)


########################### Prices ####################################
print('Prices')

# We can do some basic aggregations
# 把同一商店里同一商品的price分别计算max, min, std, mean，然后加到prices_df里面
prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')
prices_df['price_median'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('median')
prices_df['price_dif'] = prices_df['price_median'] - prices_df['price_mean']
# and do price normalization (min/max scaling)
prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

# Some items are can be inflation dependent and some items are very "stable"
# 对于计算商品的存在的不同价格数，如果说价格只有一个，则说明价格十分稳定
# 对于同一商店内有着一样商品的价格的商品有多少个

prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')
# I would like some "rolling" aggregations but would like months and years as "window"
calendar_prices = calendar_df[['wm_yr_wk','month','year']]
calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
# 让prices_df每一行有年、月，方便后面计算趋势
prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
del calendar_prices

# Now we can add price "momentum" (some sort of)
# Shifted by week  by month mean by year mean
# 通过除以自己的上一周的价格计算价格每周/月/年变化趋势
prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

del prices_df['month'], prices_df['year']


########################### Merge prices and save part 2 ###################################
print('Merge prices and save part 2')

# Merge Prices，将多种价格整合进grid_df里
original_columns = list(grid_df)
grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
# 保留价格和所需要的唯一的ID以及天数，该ID有store_id, item_id
keep_columns = [col for col in list(grid_df) if col not in original_columns]
grid_df = grid_df[MAIN_INDEX+keep_columns]
grid_df = reduce_mem_usage(grid_df)

# Safe part 2
grid_df.to_pickle('../input/m5-simple-fe/grid_part_2.pkl')
print('Size:', grid_df.shape)

# We don't need prices_df anymore
del prices_df

# We can remove new columns
# or just load part_1
grid_df = pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')


########################### Merge calendar #################################
grid_df = grid_df[MAIN_INDEX]

# Merge calendar partly，将和日期相关的属性，比如是否有重要活动以及政府补助出现，整合进grid_df
icols = ['date',
         'd',
         'event_name_1',
         'event_type_1',
         'event_name_2',
         'event_type_2',
         'snap_CA',
         'snap_TX',
         'snap_WI']

grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

# Minify data 'snap_' columns we can convert to bool or int8

icols = ['event_name_1',
         'event_type_1',
         'event_name_2',
         'event_type_2',
         'snap_CA',
         'snap_TX',
         'snap_WI']
# 将一些列的值转换成category类型
for col in icols:
    grid_df[col] = grid_df[col].astype('category')

# Convert to DateTime
grid_df['date'] = pd.to_datetime(grid_df['date'])

# Make some features from date
# tm_w时那一年的第几周，tm_wm是那个月的第几周
grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
grid_df['tm_q'] = grid_df['date'].dt.quarter.astype(np.int8)
grid_df['tm_y'] = grid_df['date'].dt.year
grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

# 一周的第几天，是否是周末
# dt.dayofweek，周一是0，z
grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)

# Remove date
del grid_df['date']


########################### Save part 3 (Dates) ##########################################
print('Save part 3')

# Safe part 3
grid_df.to_pickle('../input/m5-simple-fe/grid_part_3.pkl')
print('Size:', grid_df.shape)

# We don't need calendar_df anymore
del calendar_df
del grid_df


########################### Some additional cleaning ###################################

## Part 1
# Convert 'd_(day_number)' to int
grid_df = pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')
grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

# Remove 'wm_yr_wk' as test values are not in train set
del grid_df['wm_yr_wk']
grid_df.to_pickle('grid_part_1.pkl')

del grid_df


########################### Summary ############################################
# Now we have 3 sets of features
grid_df = pd.concat([pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl'),
                     pd.read_pickle('../input/m5-simple-fe/grid_part_2.pkl').iloc[:,2:],
                     pd.read_pickle('../input/m5-simple-fe/grid_part_3.pkl').iloc[:,2:]],
                     axis=1)
                     
# Let's check again memory usage
print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
print('Size:', grid_df.shape)

# 2.5GiB + is is still too big to train our model (on kaggle with its memory limits)
# and we don't have lag features yet。 But what if we can train by state_id or shop_id?
state_id = 'CA'
grid_df = grid_df[grid_df['state_id']==state_id]
print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
#           Full Grid:   1.2GiB

store_id = 'CA_1'
grid_df = grid_df[grid_df['store_id']==store_id]
print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
#           Full Grid: 321.2MiB

# Seems its good enough now
# In other kernel we will talk about LAGS features
# Thank you.


grid_df = pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl')
cal_df = pd.read_csv('../input/calendar_disaster.csv')
grid_df = grid_df[MAIN_INDEX]

# Merge calendar partly，将和日期相关的属性，比如是否有重要活动以及政府补助出现，整合进grid_df
icols = ['date', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 
         'dis_CA', 'type_CA', 'dis_TX', 'type_TX', 'dis_WI', 'type_WI']

grid_df = grid_df.merge(cal_df[icols], on=['d'], how='left')

# Minify data 'snap_' columns we can convert to bool or int8

icols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX',
         'snap_WI', 'dis_CA', 'type_CA', 'dis_TX', 'type_TX', 'dis_WI', 'type_WI']

# 将一些列的值转换成category类型
for col in icols:
    grid_df[col] = grid_df[col].astype('category')

# Convert to DateTime
grid_df['date'] = pd.to_datetime(grid_df['date'])

# Make some features from date
# tm_w时那一年的第几周，tm_wm是那个月的第几周
grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
grid_df['tm_q'] = grid_df['date'].dt.quarter.astype(np.int8)
grid_df['tm_y'] = grid_df['date'].dt.year
grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

# 一周的第几天，是否是周末
# dt.dayofweek，周一是0，z
grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)


del grid_df['date']
grid_df.to_pickle('../input/m5-simple-fe/grid_part_3_with_q_dis.pkl')
print('Size:', grid_df.shape)


