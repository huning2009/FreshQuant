# -*- coding: utf-8 -*-

"""
@Time    : 2018/1/18 9:41
@Author  : jessica.sun
"""

import numpy as np
import pandas as pd
from .util import data_scale
from .util import extreme_process
from .util import add_industry, single_neutral
from general.mongo_data import rpt_form_mongo_to_df

def orth(fac_ret_data,method=''):
    return

def de_extreme(fac_ret_data, num=3, method='mad'):
    """
    去极值,仅对因子列有效
    Args:
        fac_ret_data (DataFrame): multi-index.
        num (int): 超过num个标准差（或MAD）即认为是极值
        method(str): {'mad','std'}
    Returns:
        DataFrame: 去极值后的data frame
    """
    return fac_ret_data.groupby(level=0).apply(extreme_process, num=num, method=method)

def standardize(fac_ret_data, method='normal'):
    """
    标准化数据
    Args:
        fac_ret_data (DataFrame): multi-index.
        method: {'normal', 'cap'}
            'normal': data---> (data - data.mean()) / data.std()
            'cap': (data - cap_weighted_mean of data) / data.std()
    Returns:

    """
    return fac_ret_data.groupby(level=0).apply(data_scale, method=method)

def del_high_corr(fac_ret_data, corr=0.8):
    """
    剔除相关性高的因子
    Args:
        fac_ret_data (DataFrame): multi-index.
    Returns:
        fac_ret_data

    """
    fac_ret_data_copy = fac_ret_data.copy()
    fac_ret_data = fac_ret_data[[col for col in fac_ret_data.columns if col not in ['cap', 'ret']]]
    correlation = fac_ret_data.corr()
    bad_columns = []
    correlation_copy = correlation.copy()
    correlation_copy = correlation_copy.abs()  # correlation_copy 取绝对值
    correlation.values[np.triu_indices_from(correlation.values,
                                            0)] = 0.0  # 把上三角（包括对角线部分）设置为0.
    if correlation.unstack().max() > corr:
        col_idx, row_idx = correlation.unstack().argmax()  # (col_idx, row_idx)
        if correlation_copy.ix[row_idx, :].mean() > correlation_copy.ix[:,
                                                    col_idx].mean():
            bad_column = row_idx
        else:
            bad_column = col_idx
        bad_columns.append(bad_column)
        correlation_copy.drop(bad_column, axis=0, inplace=True)
        correlation_copy.drop(bad_column, axis=1, inplace=True)
    fac_ret_data_copy.drop(bad_columns, axis=1, inplace=True)
    return fac_ret_data_copy

def neutral(fac_ret_data,ind='shenwan'):
    fac_ret_cap = add_industry(fac_ret_data,ind)  # 增加并保存行业数据
    new_fac_ret_cap = fac_ret_cap.groupby(level=0).apply(lambda x: single_neutral(x))
    cols = [col for col in new_fac_ret_cap.columns if col[:1] not in ['x']]
    return new_fac_ret_cap[cols]