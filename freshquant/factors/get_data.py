#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据模块
从csf数据接口获取分析用的数据
"""

import os
import csf
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from dateutil.parser import parse
from sqlalchemy import create_engine

from .config import FACTORS_DETAIL_PATH,WIND_A,DATA
from general.get_index_data import get_index_components
from general.get_stock_data import GetPrice,get_stock_industry_from_mongo
from general.mongo_data import form_mongo_to_df,general_form_mongo_to_df,rpt_form_mongo_to_df
from general.util import set_mongo_cond

import logging
from functools import reduce
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)

def get_fac_info():
    """
    获取所有因子的相关信息,如name\code\ascend
    """
    df = pd.read_csv(FACTORS_DETAIL_PATH)
    code = df[df['stat'] == 1][['name','ascend', 'code']]
    fac_info = code.set_index('code')
    return fac_info

def get_trade_calendar():
    """
    从trade_cal.csv文件读取交易日历数据
    """
    file_path = os.path.abspath(__file__)
    dir_name = os.path.split(file_path)[0]
    csv_file = os.path.join(dir_name, 'trade_cal.csv')
    trade_cal = pd.read_csv(
        csv_file,
        names=['date_time', 'total_day'],
        index_col=[0],
        parse_dates=True)
    return trade_cal

def get_stock_industry(codes,ind):
    """
    股票所属数库一级行业
    :param codes: list, 股票代码列表
    :return: 股票与对应行业
    """
    if ind in ['shenwan','wind','zhongxin']:
        df_ind=pd.read_csv(os.path.join(DATA,'industry.csv'),encoding='GBK')[['code',ind]]
        df_ind.code=df_ind.code.apply(str)
        df_ind.code=[x[:6] for x in df_ind.code]
        df_ind=df_ind.rename(columns={ind:'level1_name'})
        return df_ind
    elif ind == 'csf':
        codes_len = len(codes)
        fields = ['code', 'secu_name', 'level1_name', 'level1_code']
        if codes_len > 100:
            cutter = list(range(0, codes_len, 100))
            cutter.append(codes_len - 1)
            dict_cutter = list(zip(cutter[0:-1], cutter[1:]))
            df = pd.DataFrame()
            for i, j in dict_cutter:
                sub_codes = codes[i:j]
                temp = csf.get_stock_industry(sub_codes, field=fields)
                df = pd.concat([df, temp])
            return df
        else:
            return csf.get_stock_industry(codes, field=fields)

def get_benchmark_return(bench_code, start_date, end_date, dt_index):
    """
    BenchMark收益率

    Args:
        bench_code (str): benchMark代码，如'000300'
        start_date (str): 开始日期
        end_date (str): 结束日期
        dt_index (list): a list of str, 月末/周末/季末 dt
    Returns:
        DataFrame
    """
    field = ['close']
    df = csf.get_index_hist_bar(
        index_code=bench_code,
        start_date=start_date,
        end_date=end_date,
        field=field)
    if isinstance(df.index[0], pd.Timestamp):
        df.index = df.index.map(lambda dt: str(dt.date()))
        df.index.name = 'date'
    price = df[field].ix[dt_index, :].rename(
        columns={'close': 'benchmark_returns'}).sort_index()
    ret = price.pct_change().shift(-1).dropna()
    return ret

def get_benchmark_return_from_sql(idx_code, dt_index):
    """
    BenchMark收益率

    Args:
        idx_code (str): benchMark代码，如'000300'
        start_date (str): 开始日期
        end_date (str): 结束日期
        dt_index (list): a list of str, 月末/周末/季末 dt
    Returns:
        DataFrame:index.name=dt,columns=benchma
    """
    if idx_code == '881001':
        price=pd.read_csv(WIND_A,index_col=0)
        price=price.loc[dt_index,:]
        ret = price.pct_change().shift(-1).dropna()
    else:
        get_sql = GetPrice()
        get_sql.set_isIndex(True)
        price = get_sql.get_in_spec_dates(idx_code, dt_index)
        ret = price.pct_change().shift(-1).dropna()
        ret.index=[str(inx.date()) for inx in ret.index]
    ret.index.name='dt'
    ret.columns=['benchmark_returns']
    return ret

def get_raw_factor(factors, index_code, start_date, end_date, freq='M'):
    """
    原始因子值（未经处理过）
    :param factors: str or list, 因子代码"M009006"或因子代码列表["M009006", "M009007"]
    :param index_code: str, 指数代码，如"000300"
    :param start_date: str, 开始日期，如"2008-04-30"
    :param end_date: str，结束日期，如"2015-12-31"
    :param freq: str，换仓周期，周"W"、月"M"、季"Q"，每个周期的最后一个交易日
    :param filter: dict, 股票筛选
    :return: pd.DataFrame，因子值
    """
    # csf.get_stock_factor 一次最多取５个因子
    frame_list = []
    for b in batch(factors, 5):
        temp = csf.get_stock_factor_by_index(
            factors=b,
            index=index_code,
            start_date=start_date,
            end_date=end_date,
            freq=freq)
        frame_list.append(temp)
    frame = pd.concat(frame_list, ignore_index=True)
    df = pd.pivot_table(
        frame, values='value', index=['date', 'code'], columns=['cd'])
    return df

def get_raw_idx_factor(factors, start_date, end_date, freq='M'):
    return

def form_dt_index(start_date, end_date, freq):
    """
    获取每个调仓期的具体日期：
    self.freq=='M' 则返回每月最后一个交易日日期；
    self.freq=='W' 则返回每周最后一个交易日日期；
    self.freq=='Q' 则返回每季最后一个交易日日期；
    返回列表['2014-01-31']
    """
    start = start_date[0:8] + '01'
    end = end_date[0:8] + '01'
    trade_cal = get_trade_calendar()
    current_calendar = trade_cal.ix[start_date:end_date, :]
    funcs = {
        'W': lambda x: x.week,
        'M': lambda x: x.month,
        'Q': lambda x: x.quarter
    }
    ret = [str(data.index[-1].date()) for (year, func), data in
           current_calendar.groupby([current_calendar.index.year,funcs[freq](current_calendar.index)])]
    return ret

def generate_report_dates(dt_index):
    """
    生成涉及到的报告期,逻辑：
    1,2,3月底用上一年Q3的数据
    4,5,6,7月底用今年Q1的数据
    8,9月底用今年Q2的数据
    10，11，12用今年Q3的数据
    """
    date_lst = dt_index[0:-1]
    q_list = []
    for dt in date_lst:
        dt_sl = dt.split('-')
        yr = int(dt_sl[0])
        mt = int(dt_sl[1])
        if mt in [1, 2, 3]:
            q_list.append("".join([str(yr - 1), '-09-30']))
        elif mt in [4, 5, 6, 7]:
            q_list.append("".join([str(yr), '-03-31']))
        elif mt in [8, 9]:
            q_list.append("".join([str(yr), '-06-30']))
        elif mt in [10, 11, 12]:
            q_list.append("".join([str(yr), '-09-30']))
    ret = pd.Series(data=q_list, index=dt_index[0:-1])
    return ret


def get_idx_his_stock(codes,dt_index):
    """
    获取所选指数self.codes每一个调仓期的历史成份股
    调用ut模块的函数
    @返回：
    ret: dict，{dt: stock_list}
    100000:Hot;
    """
    rets = []
    if codes not in ['HOT']:
        for dt in dt_index[:-1]:
            ret = get_index_components(codes, dt)
            ret = list(zip([dt] * len(ret), ret))
            rets.extend(ret)
    else:
        client, db, tb = set_mongo_cond('192.168.250.200', 'ada', 'dict_index')
        filters = {'serie': 'HOT'}
        projection = dict(_id=False, idxcd=True)
        pipeline = [{"$match": filters}, {"$project": projection}]
        allrec = tb.aggregate(pipeline)
        ret = pd.DataFrame(list(allrec)).idxcd.tolist()
        for dt in dt_index[:-1]:
            ret = list(zip([dt] * len(ret), ret))
            rets.extend(ret)
    return pd.MultiIndex.from_tuples(rets, names=['dt', 'secu'])

def get_codes_factor_dict(factor_codes):
    fac_info = pd.read_csv(FACTORS_DETAIL_PATH,encoding='GBK')
    fac_info = fac_info[fac_info.stat == 1]
    ltm_mask = fac_info.tb == 'metrics.comm_idx_quant_his_a'
    ytd_mask = fac_info.tb == 'metrics.comm_idx_quant_ytd_his_a'
    fd = fac_info.loc[:, 'fd']
    fd[ltm_mask] += '_ltm'
    fd[ytd_mask] += '_ytd'
    code_pos_dict = dict(list(zip(fac_info.code, fd)))
    ret = {code: code_pos_dict.get(code) for code in factor_codes}
    return ret

def get_raw_factor_from_mongo(fac_lst,rpt_terms,all_stocks,
                   dt_index, multi_index):
    """
    一次性获取所有调仓期，所有股票的因子数据
    Args:
        fac_lst:
        rpt_terms:
        all_stocks:
        dt_index:
        multi_index: (dt,secu)
    Returns:
        object:
    """
    codes_factor_dict = get_codes_factor_dict(fac_lst)

    fac_info = pd.read_csv(FACTORS_DETAIL_PATH,encoding='GBK')  # 读取因子详细表
    fac_info = fac_info[fac_info.stat == 1]
    ascending = dict(fac_info[['code', 'ascend']][fac_info['code'].isin(fac_lst)].values)  #############
    if ascending is not None:
        ascending.update(ascending)
    # 防止传入的ascending的key 不在factor_codes中
    # self.ascending = {k:v for k,v in self.ascending.iteritems() if k in self.factor_codes}

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
    # rename factor name to code, i.e. p.p1 ---> MXXXX
    factor_codes_dict = dict(list(zip(list(codes_factor_dict.values()), list(codes_factor_dict.keys()))))
    columns_to_rename = {name: factor_codes_dict.get(name) for name in ret.columns}
    ret = ret.rename(columns=columns_to_rename)

    # 有一些列全是NA， 要去掉
    na_columns_mask = (ret.isnull().mean() == 1)
    if na_columns_mask.any():
        all_na_columns = ret.columns[na_columns_mask].tolist()
        logger.info('column {} are removed due to ALL NA'.format(all_na_columns))

        ret = ret.drop(all_na_columns, axis=1)
        fac_lst = sorted(set(fac_lst) - set(all_na_columns))
        for k in all_na_columns:
            ascending.pop(k)
    return ret

def get_rpt_from_mongo(fac_lst, rpt_terms, all_stocks,dt_index, multi_index):
    is_lst=[fac for fac in fac_lst if fac[:2] == 'is']
    bs_lst = [fac for fac in fac_lst if fac[:2] == 'bs']
    cf_lst = [fac for fac in fac_lst if fac[:2] == 'cf']

    host = '122.144.134.95'
    db_name = 'fin'
    pos = ['secu', 'y', 'items.ov', 'items.cd']

    filters_is = {
        "y": {"$in": rpt_terms.values.tolist()},
        "secu": {"$in": all_stocks},
        "ctyp": 2,
        "rpt":'2.1.2',
        'items.cd': {'$in': is_lst}}
    filters_bs = {
        "y": {"$in": rpt_terms.values.tolist()},
        "secu": {"$in": all_stocks},
        "ctyp": 2,
        "rpt":'2.2.2',
        'items.cd': {'$in': bs_lst}}
    filters_cf = {
        "y": {"$in": rpt_terms.values.tolist()},
        "secu": {"$in": all_stocks},
        "ctyp": 2,
        "rpt":'2.3.2',
        'items.cd': {'$in': cf_lst}}
    df_is=None
    df_bs=None
    df_cf=None
    if is_lst != []:
        df_is = rpt_form_mongo_to_df(host, db_name,tb_name = 'fin_rpt_tpl_ltm', pos=pos, filters=filters_is)
    if bs_lst != []:
        df_bs = rpt_form_mongo_to_df(host, db_name,tb_name = 'fin_rpt_tpl_ytd', pos=pos, filters=filters_bs)
    if cf_lst != []:
        df_cf = rpt_form_mongo_to_df(host, db_name,tb_name = 'fin_rpt_tpl_ltm', pos=pos, filters=filters_cf)
    lst_df=[df for df in [df_is,df_bs,df_cf] if df is not None]
    df_combined=reduce(lambda x,y:pd.merge(x.reset_index(),y.reset_index(),how='outer',on=['y','secu']),lst_df)
    df_rpt_terms = pd.DataFrame({'dt': rpt_terms.index, 'y': rpt_terms.values})
    df_combined=pd.merge(df_combined.reset_index(),df_rpt_terms,on='y',how='outer')
    df_combined.drop(['y'], axis=1, inplace=True)

    raw_fac = df_combined.set_index(['dt', 'secu']).sort_index()
    ret=raw_fac.loc[multi_index, :]
    # 有一些列全是NA， 要去掉
    na_columns_mask = (ret.isnull().mean() == 1)
    if na_columns_mask.any():
        all_na_columns = ret.columns[na_columns_mask].tolist()
        logger.info('column {} are removed due to ALL NA'.format(all_na_columns))
        ret = ret.drop(all_na_columns, axis=1)
        fac_lst = sorted(set(fac_lst) - set(all_na_columns))
        for k in all_na_columns:
            ascending.pop(k)
    return ret

def get_cap_data(index_code, start_date, end_date, freq='M'):
    """
    总市值数据
    :param index_code: str, 指数代码，如"000300"
    :param start_date: str, 开始日期，如"2008-04-30"
    :param end_date: str，结束日期，如"2015-12-31"
    :param freq: str，换仓周日，周"W"、月"M"、季"Q"
    :return: pd.DataFrame，因子值
    """
    return get_raw_factor('M004023', index_code, start_date, end_date, freq)


def get_index_component(index_code, date):
    """
    指数历史成分股
    :param index_code: str, 指数代码'000300'
    :param date: str, 日期'2015-01-10'
    :return: list, 股票代码
    """
    df = csf.get_index_component(index_code, date)
    return df.code.tolist()


def get_stock_lst_date(codes):
    """
    股票首次上市日期
    Args:
        codes (list): 股票代码列表
    Returns:
        DataFrame, 两列, 一列是code, 六位股票代码, 一列是listing_date
    """
    ipo_info = Parallel(
        n_jobs=20, backend='threading', verbose=5)(
            delayed(csf.get_stock_ipo_info)(code, field=['code', 'dt'])
            for code in codes)
    ipo_info = pd.concat(ipo_info, ignore_index=True)
    ipo_info.loc[:, 'code'] = ipo_info.code.str.slice(0, 6)
    ipo_info = ipo_info.rename(columns={'dt': 'listing_date'})
    return ipo_info

def get_stock_lst_date_local(codes):
    """
    股票首次上市日期
    Args:
        codes (list): 股票代码列表
    Returns:
        DataFrame, 两列, 一列是code, 六位股票代码, 一列是listing_date
    """
    pos = ['code','ls.dt']
    filters = {"code": {"$in": codes}}
    df=general_form_mongo_to_df(host='122.144.134.4',db_name='ada',tb_name='base_stock',pos=pos,filters=filters)
    df=df.rename(columns={'ls.dt':'listing_date','code':'secu'})
    return df

def get_csf_index_factor_data():
    """
    获取数库行业指数因子数据
    :return:
    """

    pass


def get_st_stock_today(date=None):
    """
    获取ST股票
    Args:
        date (str): 日期

    Returns:
        DataFrame: 该日期ST股票
    """
    return csf.get_st_stock_today(date)

def get_st_stock_local(dt_lst):
    '''
    获取st股票
    jyzt!="N"   交易状态 N通常状态；
    zqjb !="N"   证券级别  N 表示正常状态
    tpbz == "F"  停牌标志   T-停牌
    ## engine = create_engine('mysql://pd_team:pd_team321@!@122.144.134.21/ada-fd')
    ## sql_trade = (""" SELECT * FROM `ada-fd`.hq_stock_trade where dt = '{}' """).format(date)
    '''
    dt_tup = tuple(dt_lst*2) if len(dt_lst)==1 else tuple(dt_lst)
    engine = create_engine('mysql+mysqlconnector://pd_team:pd_team123@!@192.168.250.200/ada-fd')
    sql_trade = (""" SELECT * FROM `ada-fd`.hq_stock_trade where dt in {} """).format(dt_tup)
    trade = pd.read_sql_query(sql_trade, engine)
    st=trade.query(' zqjb == "s" or zqjb == "*" ')[['dt','tick']]
    st['dt'] = st['dt'].map(lambda x:str(x))
    st['tick']=st['tick'].apply(lambda x:str(x) + ('_SH_EQ') if x.startswith('6') else str(x) + ('_SZ_EQ') )
    st=st.rename(columns={'tick':'secu'})
    return st

def get_stock_sus_local(dt_lst):
    '''
    获取st股票
    jyzt!="N"   交易状态 N通常状态；
    zqjb !="N"   证券级别  N 表示正常状态
    tpbz == "F"  停牌标志   T-停牌
    ## engine = create_engine('mysql://pd_team:pd_team321@!@122.144.134.21/ada-fd')
    ## sql_trade = (""" SELECT * FROM `ada-fd`.hq_stock_trade where dt = '{}' """).format(date)
    '''
    dt_tup = tuple(dt_lst * 2) if len(dt_lst) == 1 else tuple(dt_lst)
    engine = create_engine('mysql+mysqlconnector://pd_team:pd_team123@!@192.168.250.200/ada-fd')
    sql_tp = (""" SELECT dt,tick FROM `ada-fd`.hq_stock_tp where dt in {} """).format(dt_tup)
    sus = pd.read_sql_query(sql_tp, engine)
    sus['dt'] = sus['dt'].map(lambda x:str(x))
    sus['tick']=sus['tick'].apply(lambda x:str(x)+('_SH_EQ') if x.startswith('6') else str(x)+('_SZ_EQ'))
    sus=sus.rename(columns={'tick':'secu'})
    sus=sus.rename(columns={'tick':'secu'})
    return sus

def get_stock_sus_today(date):
    """
    获取该日停牌股票
    Args:
        date (str): 日期

    Returns:
        DataFrame: 该日期停牌股票
    """
    return csf.get_st_stock_today(date)


def get_stock_returns(stocks, start_date, end_date, freq):
    """
    获取股票收益率
    Args:
        stocks (Iterable): 股票序列
        start_date (str): 开始日期
        end_date (str): 结束日期
        freq (str): 频度, {'W','M','Q'}
                W: 每周
                M: 每月
                Q: 每季度

    Returns:
        DataFrame: multi-index, level0=date, level1=code. 收益率, 仅有一列,列名称为'ret'
    """
    close_price = Parallel(
        n_jobs=10, backend='threading', verbose=5)(
            delayed(csf.get_stock_hist_bar)(code,
                                            freq,
                                            start_date=start_date,
                                            end_date=end_date,
                                            field=['date', 'close'])
            for code in stocks)
    for start_date, p in zip(stocks, close_price):
        p['tick'] = start_date
    close_price = pd.concat(close_price)
    close_price = close_price.dropna()
    # index.name原来为空
    close_price.index.name = 'dt'
    # 转成一个frame, index:dt, columns:tick
    close_price = (close_price.set_index(
        'tick', append=True).to_panel()['close'].sort_index()
                   .fillna(method='ffill'))
    # 取每个周期末
    group_key = {'M': [close_price.index.year, close_price.index.month],
                 'W': [close_price.index.year, close_price.index.week],
                 'Q': [close_price.index.year, close_price.index.quarter]}
    close_price = close_price.groupby(group_key[freq]).tail(1)
    returns = close_price.pct_change().shift(-1).dropna(axis=1, how='all')
    returns.index = returns.index.map(lambda dt: str(dt.date()))
    returns.index.name = 'date'
    returns = returns.unstack().to_frame()
    returns.columns = ['ret']
    returns = returns.swaplevel(0, 1).sort_index()
    returns.index.names = ['date', 'code']
    return returns

def get_stock_returns_from_sql(stocks, dt_index, multi_index):
    """
    逐个调仓期,根据当期的股票代码（历史成份股记录）读取下期收益率
    收益率 = (下期股价-当期股价)/当期股价
    @返回：
    ret: dict, {dt: df}
    """
    get_sql = GetPrice()
    get_sql.set_isIndex(False)
    df_pr = get_sql.get_in_spec_dates(stocks, dt_index)
    df_ret = np.round(df_pr.pct_change().shift(-1).T, 6)

    df_ret.columns = [str(x.date()) for x in df_ret.columns]
    df_ret = df_ret.T.stack().to_frame()
    df_ret.index.names = ['dt', 'secu']
    df_ret.columns = ['ret']
    ret = df_ret.ix[multi_index, :]
    return ret

def get_idx_returns(stocks, start_date, end_date, freq):
    """
    获取股票收益率
    Args:
        stocks (Iterable): 股票序列
        start_date (str): 开始日期
        end_date (str): 结束日期
        freq (str): 频度, {'W','M','Q'}
                W: 每周
                M: 每月
                Q: 每季度

    Returns:
        DataFrame: multi-index, level0=date, level1=code. 收益率, 仅有一列,列名称为'ret'
    """
    close_price = Parallel(
        n_jobs=10, backend='threading', verbose=5)(
            delayed(csf.get_stock_hist_bar)(code,
                                            freq,
                                            start_date=start_date,
                                            end_date=end_date,
                                            field=['date', 'close'])
            for code in stocks)
    for start_date, p in zip(stocks, close_price):
        p['tick'] = start_date
    close_price = pd.concat(close_price)
    close_price = close_price.dropna()
    # index.name原来为空
    close_price.index.name = 'dt'
    # 转成一个frame, index:dt, columns:tick
    close_price = (close_price.set_index(
        'tick', append=True).to_panel()['close'].sort_index()
                   .fillna(method='ffill'))
    # 取每个周期末
    group_key = {'M': [close_price.index.year, close_price.index.month],
                 'W': [close_price.index.year, close_price.index.week],
                 'Q': [close_price.index.year, close_price.index.quarter]}
    close_price = close_price.groupby(group_key[freq]).tail(1)
    returns = close_price.pct_change().shift(-1).dropna(axis=1, how='all')
    returns.index = returns.index.map(lambda dt: str(dt.date()))
    returns.index.name = 'date'
    returns = returns.unstack().to_frame()
    returns.columns = ['ret']
    returns = returns.swaplevel(0, 1).sort_index()
    returns.index.names = ['date', 'code']
    return returns


def get_industries(stocks):
    """
    获取股票对应的行业代码
    Args:
        stocks(list) : 股票列表

    Returns:
        DataFrame: 股票及其对应行业代码
    """
    return [csf.get_stock_csf_industry(
        codes, field=['code', 'level2_name']) for codes in batch(
            stocks, n=90)]

def get_industries_local(codes):
    """
    获取股票对应的行业代码
    Args:
        codes(list) : 股票列表

    Returns:
        DataFrame: 股票及其对应行业代码
    """
    dic_ind=get_stock_industry_from_mongo(codes,ind='csf')
    return dic_ind

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
