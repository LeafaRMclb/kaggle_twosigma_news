# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
from datetime import datetime, date
import os
print(os.listdir("../input"))
import gc
gc.enable()
import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.

from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
#make sure to exclude financial crisis data/time 
(market_train_df, news_train_df) = env.get_training_data()
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
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
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
market_train_df=reduce_mem_usage(market_train_df)
test_df_columns = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
                   'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                   'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                   'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                   'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
base_df = market_train_df[market_train_df['time'] >= '2016-11-01']
base_df = base_df[test_df_columns]
base_df['id'] = -1
base_df.shape

market_train_df = market_train_df.loc[market_train_df['time'] >= '2008-10-01']
market_train_df.sort_values('time')
#market_train_df['date'] = market_train_df['time'].dt.date

# Fill nan
#market_train_fill = market_train_df
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']

market_train_df['open_close_ratio'] = np.abs(market_train_df['open']/market_train_df['close'])
market_train_df['close_open_ratio'] = np.abs(market_train_df['close']/market_train_df['open'])
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] < 1.5]
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] > 0.5]
market_train_df = market_train_df.loc[market_train_df['open_close_ratio'] < 1.5]
market_train_df = market_train_df.loc[market_train_df['open_close_ratio'] > 0.5]

market_train_df.drop(['open_close_ratio','close_open_ratio'], axis=1, inplace=True)
column_return = column_market + column_raw + ['returnsOpenNextMktres10']
orig_len = market_train_df.shape[0]
for column in column_return:
    market_train_df = market_train_df.loc[market_train_df[column]>=-1]
    market_train_df = market_train_df.loc[market_train_df[column]<=1]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)
print('Removing strange data ...')
orig_len = market_train_df.shape[0]
market_train_df = market_train_df[~market_train_df['assetCode'].isin(['PGN.N','EBRYY.OB'])]
#market_train_df = market_train_df[~market_train_df['assetName'].isin(['Unknown'])]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)

#market_train_df = market_train_df.loc[(market_train_df['time'] >= '2012-01-01') & (market_train_df['time'] < '2015-01-01')]


def prep(market_train_df):
    tmp_map_b = {}
    for i in market_train_df['assetCode'].unique():
        a,b = i.split('.')
        #tmp_map_a[i] = a
        tmp_map_b[i] = b
    #market_train_df['assetCode_asset']=market_train_df['assetCode'].map(tmp_map_a)
    market_train_df['assetCode_exchange']=market_train_df['assetCode'].map(tmp_map_b)
    market_train_df.time = pd.to_datetime(market_train_df.time)
    market_train_df['month'] = market_train_df.time.dt.month
    market_train_df['dayofweek'] = market_train_df.time.dt.dayofweek
    market_train_df['date'] = market_train_df.time.dt.date
    for i in range(len(column_raw)):
        #print(column_market[i], column_raw[i])
        market_train_df[column_market[i]] = market_train_df[column_market[i]].fillna(market_train_df[column_raw[i]])
        
        market_train_df['%s_-_%s'%(column_raw[i], column_market[i])] = (market_train_df[column_raw[i]] - market_train_df[column_market[i]])#
    return market_train_df
market_train_df = prep(market_train_df)

impcols=[f for f in market_train_df.columns if '_-_' in f]
def process_ma(mdf,columns=['open','close'], windows=[5, 20, 15, 40, 30, 50], altcols=impcols, altwindows=[5,10,20,30]):
    df = mdf[['time','assetCode','assetCode_exchange']+columns]
    idxopen = df[['open','time','assetCode','assetCode_exchange']].groupby(['assetCode_exchange','time'])['open'].transform('sum')
    idxclose = df[['close','time','assetCode','assetCode_exchange']].groupby(['assetCode_exchange','time'])['close'].transform('sum')
    idxres = idxopen - idxclose
    for col in columns:
        index_sum =  df.groupby(['assetCode_exchange','time'])[col].transform('sum')
        for window in windows:
            ma_column = f'ma_{col}_{window}'
            ma_dev_column = f'ma_dev_{col}_{window}'
            std_column = f'std_{col}_{window}'
            shifted = f'shifted_{col}_{window}'
            quasi = f'returns{col}PrevRaw{window}'
            res = f'returns{col}PrevMktres{window}'
            returnsres = f'returns{col}overidx{window}'
            #print(ma_column)
            df[shifted] = df.groupby('assetCode')[col].shift(window)
            df[quasi] = ( df[col] / df[shifted]) - 1
            
            res10 = (df[col] + df[shifted]) / index_sum
            residual = (df[col] + df[shifted]) / idxres
            
            df[returnsres] = (df[quasi] / residual)
            df[res] = (df[quasi] - res10)
            # calc moving average
            
        df.drop(col, axis=1, inplace=True)
    df.drop('assetCode_exchange', axis=1, inplace=True)
    new_df = df
    del df
    gc.collect()
    #print('processing altcols')
    df = mdf[['time','assetCode'] + altcols]
    
    for col in altcols:
        
        for sh in [1,5,10]:
            shifted = f"shift_{col}_{sh}"
            df[shifted] = df.groupby('assetCode')[col].shift(sh)
        #df['{}_shift_ng1'.format(col)] = df['{}'.format(col)].shift(-1)

        for window in altwindows:
            ma = f"ma_{col}_{window}"
            madev = f"madev_{col}_{window}"
            
            #print(ma_column)
            # calc moving average
            df[ma] = df.groupby('assetCode')[col].apply(lambda x: x.rolling(window).mean())
            # calc rate of deviation from moving average
            # calc moving std
            df[madev] = df[col] / df[ma] - 1
        
        df.drop(col, axis=1, inplace=True)
    df = pd.merge(df , new_df, how='left', on=['time','assetCode'])
    del mdf
    gc.collect()
            
    return df
import time
t=time.time()
res_df = process_ma(market_train_df)
market_train_df = pd.merge(market_train_df, res_df, how='left', on=['time','assetCode'])
print("{} elapsed".format(time.time()-t))
del res_df, news_train_df
gc.collect()
'''
impcols=[f for f in market_train_df.columns if '_-_' in f]
def process_index_res(df, columns=impcols, windows=[5, 10, 20, 30]):
    df = df[['time','assetCode']+columns]
    
    for col in columns:
        
        for sh in [1,5,10]:
            shifted = f"shift_{col}_{sh}"
            df[shifted] = df.groupby('assetCode')[col].shift(sh)
        #df['{}_shift_ng1'.format(col)] = df['{}'.format(col)].shift(-1)

        for window in windows:
            ma = f"ma_{col}_{window}"
            madev = f"madev_{col}_{window}"
            st = f"std_{col}_{window}"
            ms = f"ms_{col}_{window}"
            #print(ma_column)

            # calc moving average
            df[ma] = df.groupby('assetCode')[col].apply(lambda x: x.rolling(window).mean())
            # calc rate of deviation from moving average
            # calc moving std
            df[st] = df.groupby('assetCode')[col].apply(lambda x: x.rolling(window).std())
            
            df[ms] = (df[ma] - df[st]) / 2
            df[madev] = df[col] / df[ma] - 1
        
        df.drop(col, axis=1, inplace=True)
    
    return df
t = time.time()
res_df = process_index_res(market_train_df)

market_train_df = pd.merge(market_train_df, res_df, how='left', on=['time','assetCode'])
print("{} elapsed".format(time.time()-t))
del res_df
gc.collect()
'''


market_train_df.dropna(inplace=True)

#market_train_df = market_train_df.loc[market_train_df['time'] >= '2009-01-01']
'''
from sklearn.preprocessing import LabelEncoder
le_xch = LabelEncoder()
le_xch.fit(market_train_df['assetCode_exchange'])
market_train_df['assetCode_exchange']=le_xch.transform(market_train_df['assetCode_exchange'])

num_cols = [col for col in market_train_df.columns\
                if col not in ['time','assetCode', 'universe', 'assetName', 'returnsOpenNextMktres10']]


def code_prep(market_train):
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    #market_train = market_train.dropna(axis=0)
    return market_train
#market_train_df = code_prep(market_train_df)

num_cols = [col for col in market_train_df.columns\
                if col not in ['time','assetCode', 'universe', 'assetName', 'returnsOpenNextMktres10']]
                
'''                
num_cols = [col for col in market_train_df.columns\
                if col not in ['time','assetCode', 'universe', 'assetName', 'date', \
                               'returnsOpenNextMktres10','assetCodesMap', 'assetCode_exchange']]
market_train_df = reduce_mem_usage(market_train_df)


up = (market_train_df.returnsOpenNextMktres10 >= 0).values
r = market_train_df.returnsOpenNextMktres10.values
universe = market_train_df.universe
day = market_train_df.time.dt.date
assert market_train_df.shape[0] == up.shape[0] == r.shape[0] == universe.shape[0] == day.shape[0]

from sklearn.model_selection import train_test_split
X_train, X_test, up_train, up_test, _, r_test, _, u_test, _, d_test = \
train_test_split(market_train_df[num_cols].fillna(0), up, r, universe, day, test_size=0.25, random_state=122221)

def xgb_metric(preds, train_data):
    labels = train_data.get_label()
    preds = preds * 2 - 1
    
    # calculation of actual metric that is used to calculate final score
    #r_train = r_train.clip(-1,1) # get rid of outliers. Where do they come from??
    
    x_t_i = preds * r_test * u_test
    #x_t_i = preds * r_test * u_test
    #print(u_test)
    data = {'day' : d_test, 'x_t_i' : x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_test = mean / std
    return 'sigma_score', score_test
    
import xgboost
dtrain = xgboost.DMatrix(X_train.values, up_train)
dvalid = xgboost.DMatrix(X_test.values, up_test)

del market_train_df
gc.collect()


#[49]\teval-error:0.381658\teval-sigma_score:0.930109
#param = {'max_depth': 6, 'eta': 0.1,  'random_state':299, 'objective': 'binary:logistic', 'gamma': 1.5, 'silent':True}
param = {'max_depth': 10,  'random_state':1029, 'objective': 'binary:logistic', 'gamma': 3.993, 'silent':True, 'colsample_bytree':0.9659, 'eta':0.2,
    'colsample_bylevel':0.765,'reg_alpha':0.5159 , 'reg_lambda': 0.4685, 'subsample': 0.9308
}
watchlist = [(dvalid, 'eval')]#, (dtrain, 'train')]
num_round=150

def learning_rate_power(current_round, _):
    base_learning_rate = 0.3
    min_learning_rate = 0.1
    lr = base_learning_rate * np.power(0.995,current_round)
    return max(lr, min_learning_rate)

bst = xgboost.train(param, dtrain, num_round, watchlist , feval=xgb_metric, verbose_eval=20, early_stopping_rounds=5,
                   maximize=True) #callbacks=[xgboost.callback.reset_learning_rate(learning_rate_power)])
                      # lgb.reset_parameter(learning_rate=learning_rate_power)]))[xgb.callback.reset_learning_rate(custom_rates)]

param_2 = {'max_depth': 10,  'random_state':1029, 'objective': 'binary:logistic', 'gamma': 4.035, 'silent':True, 'colsample_bytree':0.5455, 'eta':0.2,
    'colsample_bylevel':0.6003,'reg_alpha':0.7486 , 'reg_lambda': 0.7269, 'subsample': 0.8367
}

bst_2 = xgboost.train(param_2, dtrain, num_round, watchlist , feval=xgb_metric, verbose_eval=20, early_stopping_rounds=5,
                   maximize=True)
                   
                   
param_3 = {'max_depth': 10,  'random_state':1029, 'objective': 'binary:logistic', 'gamma': 2.943, 'silent':True, 'colsample_bytree':0.9501, 'eta':0.2,
    'colsample_bylevel':0.2492,'reg_alpha':0.7199 , 'reg_lambda': 0.2972, 'subsample': 0.5845
}

m = xgboost.train(param_3, dtrain, num_round, watchlist , feval=xgb_metric, verbose_eval=20, early_stopping_rounds=5,
                   maximize=True)

train_probas = bst.predict(dtrain)
train_probas1 = bst_2.predict(dtrain)
train_probas2 = m.predict(dtrain)

trainp = (train_probas + train_probas1 + train_probas2) / 3

test_probas = bst.predict(dvalid)
test_probas1 = bst_2.predict(dvalid)
test_probas2 = m.predict(dvalid)

testp = (test_probas + test_probas1 + test_probas2) / 3
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=6)
LR.fit(trainp.reshape(-1,1), up_train)

p_calibrated = LR.predict_proba(testp.reshape(-1,1))[:,1]



#.3809 -> 1.27373
def write_submission(model, model_2, model_3, env):
    days = env.get_prediction_days()
    day_id = 0
    market_obs_df_append = None
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        print(day_id, end=" ")
        market_obs_df['id'] = day_id
        if market_obs_df_append is None:
            market_obs_df_append = base_df
            
        market_obs_df_append = pd.concat([market_obs_df_append,market_obs_df],
                                         ignore_index=True,
                                         sort=False)
        t = time.time()
        market_obs_process = prep(market_obs_df_append)
        new_df = process_ma(market_obs_process)
        market_obs_process = pd.merge(market_obs_process, new_df, how='left', on=['time', 'assetCode'])
        print("processing time {}".format(time.time()-t))
        
        market_obs_df = market_obs_process[market_obs_process['id']==day_id]
        market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
        
        #predictions
        test = xgboost.DMatrix(market_obs_df[num_cols].fillna(0).values)
        preds1 = model.predict(test, ntree_limit=model.best_ntree_limit)
        preds2 = model_2.predict(test, ntree_limit=model_2.best_ntree_limit)
        preds3 = model_3.predict(test, ntree_limit=model_3.best_ntree_limit)
        preds = (preds1 + preds2+ preds3)/ 3
        preds = LR.predict_proba(preds.reshape(-1,1))[:,1]
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        preds = preds * 2 - 1
        preds[np.isnan( preds )] = 0
        preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':preds})
        predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).\
        rename(columns={'confidence':'confidenceValue'})
        env.predict(predictions_template_df)
        if day_id == 49:
            market_obs_df_append.drop(
                market_obs_df_append.index[market_obs_df_append['id']==-1],
                inplace=True)
        elif day_id >= 50:
            market_obs_df_append.drop(
                market_obs_df_append.index[market_obs_df_append['id']==day_id-50],
                inplace=True)
        day_id += 1
        
        del market_obs_process, predictions_template_df, preds
        gc.collect()
        
    env.write_submission_file()
    print('day_count',day_id)

write_submission(bst, bst_2, m, env)