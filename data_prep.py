import os, sys, json, warnings, time
import pandas as pd
import numpy as np
import datatable as dt

from tsai.all import *
# [optional] print setup
import tsai
print(f'''
    TimeSeries AI setup:
    tsai       :{tsai.__version__}
    fastai     :{fastai.__version__}
    fastcore   :{fastcore.__version__}
    torch      :{torch.__version__}
    ''')

def df_64bits_to_32bits(df, verbose = False):
    if verbose:
        print(f'df memory use before conversion: {df.memory_usage(index=True, deep = True).sum()}')
    df = df.astype({col: np.float32 for col in df.select_dtypes('float64').columns})
    df = df.astype({col: np.int32 for col in df.select_dtypes('int64').columns})
    if verbose:
        print(f'df memory use after conversion: {df.memory_usage(index=True, deep = True).sum()}')
    return df

def reduce_memory_usage(df):
    start_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_memory} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
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
                    pass
        else:
            df[col] = df[col].astype('category')

    end_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe after reduction {end_memory} MB")
    print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df

def get_train_df(filepath, min_weight = None, min_resp = 0,
    dropna = False, fillna_method = 'ffill', reduce_mem = False):
    '''
    Args:
        fillna_method: 'ffill' means forward fill (i.e. use last observation)
    '''
    s_time = time.time()
    train_dt = dt.fread(filepath)
    #train_df = df_64bits_to_32bits(train_dt.to_pandas(), verbose= True)
    # train_df = reduce_memory_usage(train_dt.to_pandas())
    train_df = train_dt.to_pandas()
    train_df.set_index('ts_id')
    r_time = round(time.time() - s_time, 2)
    print(f'df took {r_time} seconds to load')

    if dropna:
        len_with_na = len(train_df)
        train_df = train_df.dropna(axis = 0)
        print(f'{round(1-(len(train_df)/len_with_na),2)*100}% of rows with NA values dropped.')
    elif fillna_method:
        s_time = time.time()
        train_df = train_df.fillna(method = fillna_method).fillna(0)
        r_time = round(time.time() - s_time, 2)
        print(f'{fillna_method} took {r_time} seconds to run')

    if min_weight and 'weight' in train_df.columns:
        print(f'{len(train_df)} trades in raw trainset')
        train_df = train_df[train_df['weight']> min_weight]
        print(f'{len(train_df)} non-zero weight trains')

    # Target Definition
    if 'resp' in train_df.columns:
        train_df['action'] = (train_df['resp']> min_resp).astype('int')
        print(f'actionable trades pct: {round((train_df["action"].sum()/len(train_df))*100,2)}')

    return reduce_memory_usage(train_df) if reduce_mem else train_df

def ts_df_split(df, validate_pct = 0.3, verbose = False):
    '''
    split a time-series df (must be already ordered by index)
    and return two df's
    '''
    split_point = int(len(df) * (1-validate_pct))
    if verbose:
        print(f'''
        Split ratio: {split_point}, {len(df)-split_point}
        ''')
    return df[:split_point], df[split_point:]

def get_ts_dataloader(df, window_length = 100, target_col = 'action', stride = None,
        num_workers = 2, batch_size = [64,128], valid_size = 0.3,
        tfms = [None, Categorize()], batch_tfms = [TSStandardize(by_sample = True)],
        drop_last = False, shuffle_train = False
        ):
    '''
    transform a df (2d multivariant) into a tsai dataloader
    Args:
        window_length: number of past trades per sample
        stride: If None, stride=window_len (no overlap)
        num_workers: number of CPUs to use
        batch_size: train & validation batch size
    '''
    feat_cols = [col for col in df.columns if 'feature' in col]
    X, y = SlidingWindow( window_length= window_length, stride = stride,
            get_x = feat_cols, get_y= target_col)(df)
    itemify(X,y)
    sample_count, feat_count, obs_per_sample = X.shape
    print(f'''
        Turned a DF of shape {df[feat_cols].shape} into a 3D dataset of:\n
        * {sample_count} samples \n
        * with {feat_count} features \n
        * and {obs_per_sample} observations per sample
        ''')

    splits = get_splits(y, valid_size= valid_size, stratify= True,
        random_state = 420, shuffle= False)
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid,
                bs= batch_size, batch_tfms=[TSStandardize()],
                num_workers= num_workers, drop_last = drop_last, shuffle_train = shuffle_train)

    # dls = get_ts_dls(X, y, splits = splits, tfms = tfms,
    #         drop_last = False, shuffle_train = False,
    #         batch_tfms = batch_tfms, bs = batch_size
    #         )
    return dls
