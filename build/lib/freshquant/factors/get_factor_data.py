# -*- coding: utf-8 -*-

"""
@Time    : 2018/1/18 9:46
@Author  : jessica.sun
"""

import numpy as np
import pandas as pd

def prepare_data_from_api(factor_name, index_code, benchmark_code, start_date, end_date, freq):
    """
    获取因子数据,股票市值,股票对应的下期收益率,下期benchmark收益率
    Args:
        benchmark_code: 一个指数代码, 例如'000300'
        factor_name (str): 因子名称, 例如 'M004023'
        index_code (str): 六位指数代码, 例如 '000300'
        start_date (str): 开始日期, YYYY-mm-dd
        end_date (str): 结束日期, YYYY-mm-dd
        freq (str): 数据频率, m=month
    Returns:
        DataFrame: multi-index, level0=date, level1=code. 原始因子, 其下期收益率, 市值, benchmark下期收益率
    """
    if isinstance(factor_name, string_types):
        factor_name_ = [factor_name]
    elif isinstance(factor_name, (list, tuple)):
        factor_name_ = list(factor_name)
    if 'M004023' not in factor_name_:
        factor_name = factor_name_ + ['M004023']
    else:
        factor_name = factor_name_
    factor_name = [str(n) for n in factor_name]
    raw_fac = get_raw_factor(
        factor_name, index_code, start_date, end_date, freq)
    raw_fac = raw_fac.rename(columns={'M004023': 'cap'})
    if 'M004023' in factor_name_:
        raw_fac.loc[:, 'M004023'] = raw_fac.cap

    dts = sorted(raw_fac.index.get_level_values(0).unique())
    s, e = str(dts[0]), str(dts[-1])

    benchmark_returns = get_benchmark_return(bench_code=benchmark_code, start_date=start_date, end_date=end_date,
                                             dt_index=dts)
    stocks = sorted([str(c)
                     for c in raw_fac.index.get_level_values(1).unique()])
    returns = get_stock_returns(stocks, s, e, freq)

    # 去掉最后一期数据
    inx = raw_fac.index.get_level_values(0).unique()[:-1]
    raw_fac = raw_fac.loc[pd.IndexSlice[inx, :], :]
    fac_ret = raw_fac.join(returns)

    fac_ret = fac_ret.join(benchmark_returns)

    return fac_ret

def prepare_rpt_from_mongo(factor_name, index_code, benchmark_code, start_date, end_date, freq):
    """
    获取因子数据,股票市值,股票对应的下期收益率,下期benchmark收益率
    Args:
        benchmark_code: 一个指数代码, 例如'000300'
        factor_name (str): 因子名称, 例如 'is_tpl_1'
        index_code (str): 六位指数代码, 例如 '000300'
        start_date (str): 开始日期, YYYY-mm-dd
        end_date (str): 结束日期, YYYY-mm-dd
        freq (str): 数据频率, m=month
    Returns:
        DataFrame: multi-index, level0=date, level1=code. 原始因子, 其下期收益率, 市值, benchmark下期收益率
    """
    if isinstance(factor_name, string_types):
        factor_name_ = [factor_name]
    elif isinstance(factor_name, (list, tuple)):
        factor_name_ = list(factor_name)
    factor_name = [str(n) for n in factor_name]
    # 准备中间变量
    dt_index=form_dt_index(start_date, end_date, freq)
    multi_index = get_idx_his_stock(index_code, dt_index)
    all_stocks = list(set(multi_index.levels[1]))
    rpt_terms=generate_report_dates(dt_index)
    # 获取fin数据
    raw_fac=get_rpt_from_mongo(factor_name, rpt_terms, all_stocks,dt_index, multi_index)
    return raw_fac


def prepare_data_from_mongo(factor_name, index_code, benchmark_code, start_date, end_date, freq):
    """
    获取因子数据,股票市值,股票对应的下期收益率,下期benchmark收益率
    Args:
        benchmark_code: 一个指数代码, 例如'000300'
        factor_name (str): 因子名称, 例如 'M004023'
        index_code (str): 六位指数代码, 例如 '000300'
        start_date (str): 开始日期, YYYY-mm-dd
        end_date (str): 结束日期, YYYY-mm-dd
        freq (str): 数据频率, m=month
    Returns:
        DataFrame: multi-index, level0=date, level1=code. 原始因子, 其下期收益率, 市值, benchmark下期收益率
    """
    if isinstance(factor_name, string_types):
        factor_name_ = [factor_name]
    elif isinstance(factor_name, (list, tuple)):
        factor_name_ = list(factor_name)
    if 'M004023' not in factor_name_:
        factor_name = factor_name_ + ['M004023']
    else:
        factor_name = factor_name_
    factor_name = [str(n) for n in factor_name]

    # 准备中间变量
    dt_index=form_dt_index(start_date, end_date, freq)
    multi_index = get_idx_his_stock(index_code, dt_index)
    all_stocks = list(set(multi_index.levels[1]))
    rpt_terms=generate_report_dates(dt_index)

    # 获取股票对应的因子数据
    raw_fac = get_raw_factor_from_mongo(
        factor_name, rpt_terms,all_stocks,dt_index,multi_index)
    raw_fac = raw_fac.rename(columns={'M004023': 'cap'})
    if 'M004023' in factor_name_:
        raw_fac.loc[:, 'M004023'] = raw_fac.cap

    # 获取指数收益率数据
    benchmark_returns = get_benchmark_return_from_sql(benchmark_code,dt_index)

    # 获取股票收益率数据
    returns = get_stock_returns_from_sql(all_stocks,dt_index,multi_index)

    # 合并数据
    fac_ret = raw_fac.join(returns)
    fac_ret = fac_ret.join(benchmark_returns)

    return fac_ret