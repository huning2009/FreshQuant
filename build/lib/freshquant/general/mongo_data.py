# -*- coding: UTF-8 -*-
from pymongo import MongoClient
import numpy as np
import pandas as pd
from dateutil.parser import parse
from six import string_types
from general.util import set_mongo_cond
import os
from functools import reduce



# def get_benchmark_return_from_mongo(bench_code,dt_index):
#     """
#     BenchMark收益率
#
#     Args:
#         bench_code (str): benchMark代码，如'000300'
#         start_date (str): 开始日期
#         end_date (str): 结束日期
#         dt_index (list): a list of str, 月末/周末/季末 dt
#     Returns:
#         DataFrame
#     """
#     get_sql = GetPrice()
#     get_sql.set_isIndex(True)
#     price = get_sql.get_in_spec_dates(bench_code, dt_index)
#     ret = price.pct_change().shift(-1).dropna()
#     ret.index = [str(inx.date()) for inx in ret.index]
#     return ret
#
# def get_stock_returns_from_mongo(stocks, start_date, end_date, freq):
#     """
#     逐个调仓期,根据当期的股票代码（历史成份股记录）读取下期收益率
#     收益率 = (下期股价-当期股价)/当期股价
#     @返回：
#     ret: dict, {dt: df}
#     """
#     dt_index=[]
#     multi_index=[]
#     get_sql = GetPrice()
#     get_sql.set_isIndex(False)
#     df_pr = get_sql.get_in_spec_dates(stocks, dt_index)
#     df_ret = np.round(df_pr.pct_change().shift(-1).T, 6)
#
#     df_ret.columns = [str(x.date()) for x in df_ret.columns]
#     df_ret = df_ret.T.stack().to_frame()
#     df_ret.index.names = ['dt', 'secu']
#     df_ret.columns = ['ret']
#     ret = df_ret.ix[multi_index, :]
#     return ret



def form_mongo_to_df(tb_name, pos, filters):
    """
    因子数据
    """
    _, db_metrics = set_mongo_cond(db_name='metrics')
    tb = {
        'pr': db_metrics.comm_idx_price_his_a,
        'ltm': db_metrics.comm_idx_quant_his_a,
        'ytd': db_metrics.comm_idx_quant_ytd_his_a,
        'tech': db_metrics.comm_idx_tech_his_a
    }
    projection_dict = {}
    for k in pos:
        key = k.replace('.', '#')
        val = ''.join(["$", k])
        projection_dict[key] = val
    projection_direct = {
        'pr': dict(_id=False, dt=True, secu=True, **projection_dict),
        'ltm': dict(_id=False, y=True, secu=True, **projection_dict),
        'ytd': dict(_id=False, y=True, secu=True, **projection_dict),
        'tech': dict(_id=False, dt=True, secu=True, **projection_dict)
    }
    pipeline = [
        {"$match": filters},
        {"$project": projection_direct[tb_name]}
    ]
    all_rec = tb[tb_name].aggregate(pipeline)
    df = pd.DataFrame(list(all_rec))
    df.columns = [col.replace("#", ".") for col in df.columns]

    return df


def general_form_mongo_to_df(host, db_name, tb_name, pos, filters):
    """
    输入mongo表连接，查询字段名(全字段，如quant.f18)，过滤条件
    pos: 查找的字段名
    filters：过滤条件，字典格式
    """
    _, db, tb = set_mongo_cond(host=host, db_name=db_name, tb_name=tb_name)
    projection_dict = {}
    for k in pos:
        key = k.replace('.', '#')
        val = ''.join(["$", k])
        projection_dict[key] = val
    projection_direct = dict(_id=False, **projection_dict)
    pipeline = [
        {"$match": filters},
        {"$project": projection_direct}
    ]
    all_rec = tb.aggregate(pipeline)
    df = pd.DataFrame(list(all_rec))
    df.columns = [col.replace("#", ".") for col in df.columns]
    return df


def rpt_form_mongo_to_df(host, db_name, tb_name,
                         pos,filters):
    """
    host='192.168.250.200', db_name='fin', tb_name='fin_rpt_tpl_ltm',
    pos=['secu', 'y', 'items.ov']
    filters = {
    "y": {"$lte": end, "$gte": start},
    'secu': {"$in": stocks},
    'items.cd': {'$in': fac_lst}}
    """
    if 'items.cd' not in pos:
        pos=pos+['items.cd']
    if 'ctyp' not in filters.keys():
        filters['ctyp']=2
        print ('add to filters: ctyp=2!')
    _, db, tb = set_mongo_cond(host=host, db_name=db_name, tb_name=tb_name)
    projection_dict = {k.replace('.', '_'): '$' + k for k in pos}
    projection_direct = dict(_id=False, **projection_dict)
    pipeline = [
        {"$unwind": "$items"},
        {"$match": filters},
        {"$project": projection_direct},
    ]
    all_rec = tb.aggregate(pipeline)
    df = pd.DataFrame(list(all_rec))
    try:
        df.items_ov=df.items_ov.map(lambda x:np.nan if x == 'NA' else float(x))
    except:
        pass
    ret=pd.pivot_table(df, values='items_ov', index=['y','secu'], columns=['items_cd'])
    del ret.columns.name
    return ret


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
    raw_fac = get_raw_factor_from_mongo(
        factor_name, index_code, start_date, end_date, freq)
    raw_fac = raw_fac.rename(columns={'M004023': 'cap'})
    if 'M004023' in factor_name_:
        raw_fac.loc[:, 'M004023'] = raw_fac.cap

    dts = sorted(raw_fac.index.get_level_values(0).unique())
    s, e = str(dts[0]), str(dts[-1])

    benchmark_returns = get_benchmark_return_from_mongo(bench_code=benchmark_code,dt_index=dts)
    stocks = sorted([str(c)
                     for c in raw_fac.index.get_level_values(1).unique()])
    returns = get_stock_returns_from_mongo(stocks, s, e, freq)

    # 去掉最后一期数据
    inx = raw_fac.index.get_level_values(0).unique()[:-1]
    raw_fac = raw_fac.loc[pd.IndexSlice[inx, :], :]
    fac_ret = raw_fac.join(returns)

    fac_ret = fac_ret.join(benchmark_returns)

    return fac_ret

def get_raw_factor_from_mongo(fac_lst, index_code, start_date, end_date, freq):
    """
    FACTORS_DETAIL_PATH, fac_lst, rpt_terms, all_stocks,dt_index, multi_index
    一次性获取所有调仓期，所有股票的因子数据
    factor_name, index_code, start_date, end_date, freq
    Args:
        FACTORS_DETAIL_PATH:
        fac_lst:
        rpt_terms:
        all_stocks:
        dt_index:
        multi_index: (dt,secu)

    Returns:
        object:
    """
    fac_info = pd.read_excel('E:\SUNJINGJING\Strategy\\utils_strategy\general\quant_dict_factors_all.xlsx') # 读取因子详细信息
    df_fac = fac_info[fac_info['code'].isin(fac_lst)]
    df_fac.loc[:, 'sdt'] = df_fac.loc[:, 'sdt'].apply(lambda x: str(parse(x).date()))
    dict_tb = {'metrics.comm_idx_tech_his_a': '', 'metrics.comm_idx_quant_his_a': '_ltm',
               'metrics.comm_idx_quant_ytd_his_a': '_ytd', 'metrics.comm_idx_price_his_a': ''}
    df_fac['new_fd'] = [df_fac.loc[inx, 'fd'] + dict_tb[df_fac.loc[inx, 'tb']] for inx in df_fac.index]
    df_fac = df_fac.set_index('new_fd')
    # fin_facs = df_fac[df_fac.table == '']  # 找出在表fin中的因子
    ltm_facs = df_fac[
        df_fac.tb == 'metrics.comm_idx_quant_his_a']  # 找出在表quant中的因子
    ytd_facs = df_fac[
        df_fac.tb == 'metrics.comm_idx_quant_ytd_his_a']  # 找出在表quant_ytd中的因子
    pr_facs = df_fac[
        df_fac.tb == 'metrics.comm_idx_price_his_a']  # 找出在表pr中的因子
    tech_facs = df_fac[
        df_fac.tb == 'metrics.comm_idx_tech_his_a']  # 找出在表tech中的因子
    ltm_pos = ltm_facs.fd.tolist()
    ytd_pos = ytd_facs.fd.tolist()
    pr_pos = pr_facs.fd.tolist()
    tech_pos = tech_facs.fd.tolist()
    ltm_empty = ltm_facs.empty
    ytd_empty = ytd_facs.empty
    pr_empty = pr_facs.empty

    df_ltm = None
    df_ytd = None
    df_pr = None
    df_tech = None
    if not ltm_empty:
        filter_ltm = {
            "y": {"$in": rpt_terms.values.tolist()},
            "secu": {"$in": all_stocks}}
        df_ltm = form_mongo_to_df('ltm', ltm_pos, filter_ltm)
        df_ltm.columns = [col + '_ltm' if col not in {'y', 'secu'} else col for
                          col in df_ltm.columns]
    if not ytd_empty:
        filter_ytd = {
            "y": {"$in": rpt_terms.values.tolist()},
            "secu": {"$in": all_stocks}}
        df_ytd = form_mongo_to_df('ytd', ytd_pos, filter_ytd)
        df_ytd.columns = [col + '_ytd' if col not in {'y', 'secu'} else col for
                          col in df_ytd.columns]
    if not pr_empty:
        filter_pr = {
            "dt": {"$in": dt_index[0:-1]},
            "secu": {"$in": all_stocks}}
        df_pr = form_mongo_to_df('pr', pr_pos, filter_pr)

    if tech_pos:
        filter_tech = {
            "dt": {"$in": dt_index[0:-1]},
            "secu": {"$in": all_stocks}}
        df_tech = form_mongo_to_df('tech', tech_pos, filter_tech)

    df_rpt_terms = pd.DataFrame(
        {'dt': rpt_terms.index, 'y': rpt_terms.values})

    raw_fac = pd.DataFrame()

    long_ltm = None
    long_ytd = None
    if not ltm_empty:
        long_ltm = pd.merge(df_rpt_terms, df_ltm, how='outer',
                            on=['y'])
        long_ltm.drop(['y'], axis=1, inplace=True)
    if not ytd_empty:
        long_ytd = pd.merge(df_rpt_terms, df_ytd, how='outer',
                            on=['y'])
        long_ytd.drop(['y'], axis=1, inplace=True)

    frame_list = [long_ltm, long_ytd, df_pr, df_tech]
    frame_list = [frame for frame in frame_list if frame is not None]

    raw_fac = reduce(
        lambda left, right: pd.merge(left, right, how='outer',
                                     on=['dt', 'secu']),
        frame_list)
    for col in list(set(raw_fac.columns).difference(set(['secu', 'dt']))):
        try:
            raw_fac.loc[:, col] = raw_fac.loc[:, col].where(raw_fac.loc[:, 'dt'] >= df_fac.loc[col, 'sdt'], np.nan)
        except:
            pass
            # raw_fac.loc[:,col]=raw_fac.loc[:,col].where(raw_fac.loc[:,'dt']>=df_fac.loc[col,'sdt'],np.nan)
    raw_fac = raw_fac.set_index(['dt', 'secu']).sort_index()
    ret = raw_fac.loc[multi_index, :]
    return ret

if __name__ == "__main__":
    stocks = ['000001_SZ_EQ']
    start = "2007-05-20"
    end = "2007-10-20"
    fac_lst = ['is_tpl_22_2']
    df = rpt_form_mongo_to_df(fac_lst, start, end, stocks, host='192.168.250.200', db_name='fin',
                              tb_name='fin_rpt_tpl_ltm')
    print(df)
    print('Done!')
