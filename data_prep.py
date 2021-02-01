import os, sys, json, warnings, time
import pandas as pd
import numpy as np
import datatable as dt

from tsai.all import *

def df_64bits_to_32bits(df, verbose = False):
    if verbose:
        print(f'df memory use before conversion: {df.memory_usage(index=True, deep = True).sum()}')
    df = df.astype({col: np.float32 for col in df.select_dtypes('float64').columns})
    df = df.astype({col: np.int32 for col in df.select_dtypes('int64').columns})
    if verbose:
        print(f'df memory use after conversion: {df.memory_usage(index=True, deep = True).sum()}')
    return df

def get_train_df(filepath, min_weight = None, min_resp = 0):
    s_time = time.time()
    train_dt = dt.fread(filepath)
    train_df = df_64bits_to_32bits(train_dt.to_pandas(), verbose= True)
    train_df.set_index('ts_id')

    if min_weight and 'weight' in train_df.columns:
        print(f'{len(train_df)} trades in raw trainset')
        train_df = train_df[train_df['weight']> min_weight]
        print(f'{len(train_df)} non-zero weight trains')

    # Target Definition
    if 'resp' in train_df.columns:
        train_df['action'] = (train_df['resp']> min_resp).astype('int')
    return train_df

def ts_df_split(df, validate_pct = 0.3):
    '''
    split a time-series df (must be already ordered by index)
    and return two df's
    '''
    split_point = int(len(df) * (1-validate_pct))
    return df[:split_point], df[split_point:]

def get_ts_dataloader(df, window_length = 100, target_col = 'action',
        num_workers = 2, batch_size = [64,128], valid_size = 0.3):
    '''
    transform a df (2d multivariant) into a tsai dataloader
    Args:
        window_length: number of past trades per sample
        num_workers: number of CPUs to use
        batch_size: train & validation batch size
    '''
    feat_cols = [col for col in train_df.columns if 'feature' in col]
    X, y = SlidingWindow( window_length= window_length,
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
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid,
                bs= batch_size, batch_tfms=[TSStandardize()],
                num_workers= num_workers)
    return dls