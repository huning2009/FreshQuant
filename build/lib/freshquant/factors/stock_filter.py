# -*- coding: utf-8 -*-

"""
@Time    : 2018/1/18 9:43
@Author  : jessica.sun
根据财务指标等过滤公司
"""

import numpy as np
import pandas as pd

def filter_out_st_from_api(fac_ret):
    """
    过滤出ST股票
    Args:
        fac_ret (DataFrame): 一个multi-index 数据框, level0=date, level1=code.
        基本思想是和ST股票聚合, status==null说明不是ST的
    Returns:
        DataFrame, 不包含停牌股票的fac_ret
    """
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    st_stocks = Parallel(n_jobs=20, backend='threading', verbose=5)(delayed(get_st_stock_today)(dt)
                                                                    for dt in dts)
    st_stocks = pd.concat(st_stocks, ignore_index=True)
    st_stocks.loc[:, 'code'] = st_stocks.code.str.slice(0, 6)
    st_stocks = st_stocks.set_index(['date', 'code']).sort_index()
    joined = fac_ret.join(st_stocks, how='left')
    result = joined[joined.status.isnull()][fac_ret.columns]
    return result

def filter_out_st_local(fac_ret):
    """
    过滤出ST股票
    Args:
        fac_ret (DataFrame): 一个multi-index 数据框, level0=date, level1=code.
        基本思想是和ST股票聚合, status==null说明不是ST的
    Returns:
        DataFrame, 不包含停牌股票的fac_ret
    """
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    st_stocks = get_st_stock_local(dts)
    st_stocks = st_stocks.set_index(['dt', 'secu']).sort_index()
    idx = set(fac_ret.index) - set(st_stocks.index)
    result = fac_ret.loc[list(idx), :].sort_index()
    return result

def filter_out_suspend_from_api(fac_ret):
    """
    过滤出停牌股票
    Args:
        fac_ret (DataFrame): multi-index, level0=date, level1=code.
    Returns:
        DataFrame: 过滤出停牌股票的DataFrame
    """
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    suspend_stocks = Parallel(n_jobs=20, backend='threading', verbose=5)(delayed(get_stock_sus_today)(date=dt)
                                                                         for dt in dts)
    for (dt, frame) in zip(dts, suspend_stocks):
        frame.loc[:, 'date'] = dt

    suspend_stocks = pd.concat(suspend_stocks, ignore_index=True)
    suspend_stocks = suspend_stocks.query('status == "T"')
    suspend_stocks = suspend_stocks.set_index(['date', 'code']).sort_index()
    joined = fac_ret.join(suspend_stocks, how='left')
    result = joined[joined.status.isnull()][fac_ret.columns]
    return result

def filter_out_suspend_local(fac_ret):
    """
    过滤出停牌股票
    Args:
        fac_ret (DataFrame): multi-index, level0=date, level1=code.
    Returns:
        DataFrame: 过滤出停牌股票的DataFrame
    """
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    sus_stocks = get_stock_sus_local(dts)
    sus_stocks = sus_stocks.set_index(['dt', 'secu']).sort_index()
    idx = set(fac_ret.index) - set(sus_stocks.index)
    result = fac_ret.loc[list(idx), :].sort_index()
    return result


def filter_out_recently_ipo_from_api(fac_ret, days=20):
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

def filter_out_recently_ipo_local(fac_ret, days=60):
    """
    过滤出最近上市的股票
    Args:
        fac_ret (DataFrame): multi-index, level0=date, level1=code.
        days (int): 上市天数
    Returns:
        DataFrame: 过滤出最近上市股票的DataFrame
    """
    stocks = sorted(fac_ret.index.get_level_values(1).unique())
    ipo_info = get_stock_lst_date_local(stocks)
    fac_ret_ = fac_ret.reset_index()
    merged = pd.merge(fac_ret_, ipo_info, on='secu')
    merged.loc[:, 'days'] = (merged.dt.map(pd.Timestamp) - merged.listing_date.map(pd.Timestamp)).dt.days
    result = (merged.query('days>{}'.format(days)).set_index(['dt', 'secu']).sort_index()[fac_ret.columns])
    return result

def filter_net_profit_ltm(fac_ret, days=20):
    """
    待完成 : 根据净利润ltm筛选股票
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

def filter_debt_ratio(fac_ret, days=20):
    """
    ** 待完成: 根据资产负债率筛选股票
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