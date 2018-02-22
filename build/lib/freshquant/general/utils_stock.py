# -*- coding: utf-8 -*-



import concurrent
import numpy as np
import pandas as pd
import csf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from joblib import Parallel,delayed
from .util import extreme_process,data_scale,add_industry,single_neutral

def batch(iterable, n=1):
    """
    将一个长序列每次n个数据
    Args:
        iterable: 可迭代的对象
        n(int): 每批的数目

    Returns:
        长序列的子序列
    Examples:
        In [3]: for b in batch(range(10),3):
   ...:     print b
   ...:
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def parallel(n_jobs, delayed_func):
    import platform
    PoolClass = ThreadPoolExecutor if platform.system() == 'Linux' else ProcessPoolExecutor
    results = []
    with PoolClass(max_workers=n_jobs) as executor:
        jobs = {executor.submit(func, *args, **kwargs): ((func.__name__,) + args) for func, args, kwargs in
                delayed_func}
        for ret in concurrent.futures.as_completed(jobs):
            func_arg = jobs[ret]
            if ret.exception() is not None:
                print(func_arg)
            else:
                results.append(ret.result())
    return results


def get_stock_lst_date(codes):
    """
    股票首次上市日期
    Args:
        codes (list): 股票代码列表
    Returns:
        DataFrame, 两列, 一列是code, 六位股票代码, 一列是listing_date
    """
    ipo_info = Parallel(
        n_jobs=20, backend='multiprocessing', verbose=5)(
            delayed(csf.get_stock_ipo_info)(code, field=['code', 'dt'])
            for code in codes)
    ipo_info = pd.concat(ipo_info, ignore_index=True)
    ipo_info.loc[:, 'code'] = ipo_info.code.str.slice(0, 6)
    ipo_info = ipo_info.rename(columns={'dt': 'listing_date'})
    return ipo_info

def de_extreme(fac_ret_data, num=1, method='mad'):
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


def neutral(fac_ret_data):
    fac_ret_cap = add_industry(fac_ret_data)  # 增加并保存行业数据
    new_fac_ret_cap = fac_ret_cap.groupby(level=0).apply(lambda x: single_neutral(x))
    cols = [col for col in new_fac_ret_cap.columns if col[:1] not in ['x']]
    return new_fac_ret_cap[cols]


def filter_out_st(fac_ret):
    """
    过滤出ST股票
    Args:
        fac_ret (DataFrame): 一个multi-index 数据框, level0=date, level1=code.
        基本思想是和ST股票聚合, status==null说明不是ST的
    Returns:
        DataFrame, 不包含停牌股票的fac_ret
    """
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    st_stocks = [csf.get_st_stock_today(dt) for dt in dts]
    # st_stocks = Parallel(n_jobs=20, backend='multiprocessing', verbose=5)(delayed(csf.get_st_stock_today)(dt)
    #                                                                 for dt in dts)
    st_stocks = pd.concat(st_stocks, ignore_index=True)
    st_stocks.loc[:, 'code'] = st_stocks.code.str.slice(0, 6)
    st_stocks = st_stocks.set_index(['date', 'code']).sort_index()
    joined = fac_ret.join(st_stocks, how='left')
    result = joined[joined.status.isnull()][fac_ret.columns]
    return result


def filter_out_suspend(fac_ret):
    """
    过滤出停牌股票
    Args:
        fac_ret (DataFrame): multi-index, level0=date, level1=code.
    Returns:
        DataFrame: 过滤出停牌股票的DataFrame
    """
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    suspend_stocks = [csf.get_stock_sus_today(dt) for dt in dts]
    # suspend_stocks = Parallel(n_jobs=20, backend='multiprocessing', verbose=5)(delayed(csf.get_stock_sus_today)(date=dt)
    #                                                                      for dt in dts)
    for (dt, frame) in zip(dts, suspend_stocks):
        frame.loc[:, 'date'] = dt

    suspend_stocks = pd.concat(suspend_stocks, ignore_index=True)
    suspend_stocks = suspend_stocks.query('status == "T"')
    suspend_stocks = suspend_stocks.set_index(['date', 'code']).sort_index()
    joined = fac_ret.join(suspend_stocks, how='left')
    result = joined[joined.status.isnull()][fac_ret.columns]
    return result


def filter_out_recently_ipo(fac_ret, days=20):
    """
    过滤出最近上市的股票
    Args:
        fac_ret (DataFrame): multi-index, level0=date, level1=code.
        days (int): 上市天数

    Returns:
        DataFrame: 过滤出最近上市股票的DataFrame
    """
    stocks = sorted(fac_ret.index.get_level_values(1).unique())
    ipo_info = get_stock_lst_date(stocks)
    fac_ret_ = fac_ret.reset_index()
    merged = pd.merge(fac_ret_, ipo_info, on='code')

    merged.loc[:, 'days'] = (
        merged.date.map(pd.Timestamp) - merged.listing_date.map(pd.Timestamp)).dt.days

    result = (merged.query('days>{}'.format(days))
              .set_index(['date', 'code'])
              .sort_index()[fac_ret.columns])

    return result




