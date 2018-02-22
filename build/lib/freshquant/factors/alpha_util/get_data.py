#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import pandas as pd
import numpy as np
import time
import datetime

from alpha_util.util import form_mongo_to_df, logger, set_mongo_cond, timethis
from .util import GetSql
from dateutil.parser import parse
from alpha_util.alpha_config import WIND_A
import os
from functools import reduce


@timethis
def get_benchmark_return(bench_code, dt_index):
    idx_code = bench_code
    if idx_code == '881001':
        price=pd.read_csv(WIND_A,index_col=0)
        price=price.loc[dt_index,:]
        ret = price.pct_change().shift(-1).dropna()
    else:
        get_sql = GetSql()
        get_sql.set_isIndex(True)
        price = get_sql.get_in_spec_dates(idx_code, dt_index)
        ret = price.pct_change().shift(-1).dropna()
        ret.index=[str(inx.date()) for inx in ret.index]
    return ret


@timethis
def get_raw_factor(FACTORS_DETAIL_PATH, fac_lst, rpt_terms, all_stocks,
                   dt_index, multi_index):
    """
    一次性获取所有调仓期，所有股票的因子数据

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
    fac_info = pd.read_csv(FACTORS_DETAIL_PATH)  # 读取常规因子详细表
    # fac_info = fac_info[fac_info.stat == 1]
    # 对列表中因子进行区分，常规因子(fac_lst1)即factors上提供的部分，新增因子(fac_lst2)为其他因子。
    fac_lst1 = [fac for fac in fac_lst if fac in fac_info['code'].tolist()]
    fac_lst2 = [fac for fac in fac_lst if fac not in fac_info['code'].tolist()]
    if fac_lst1 != []:
        df_fac = fac_info[fac_info['code'].isin(fac_lst1)]
        df_fac.loc[:,'sdt']=df_fac.loc[:,'sdt'].apply(lambda x:str(parse(x).date()))
        dict_tb={'metrics.comm_idx_tech_his_a':'','metrics.comm_idx_quant_his_a':'_ltm','metrics.comm_idx_quant_ytd_his_a':'_ytd','metrics.comm_idx_price_his_a':''}
        df_fac['new_fd']=[df_fac.loc[inx,'fd']+dict_tb[df_fac.loc[inx,'tb']] for inx in df_fac.index]
        df_fac=df_fac.set_index('new_fd')
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
                "y":{"$in": rpt_terms.values.tolist()},
                "secu":{"$in": all_stocks}}
            df_ltm = form_mongo_to_df('ltm', ltm_pos, filter_ltm)
            df_ltm.columns = [col + '_ltm' if col not in {'y', 'secu'} else col for
                              col in df_ltm.columns]
        if not ytd_empty:
            filter_ytd = {
                "y"   : {"$in": rpt_terms.values.tolist()},
                "secu": {"$in": all_stocks}}
            df_ytd = form_mongo_to_df('ytd', ytd_pos, filter_ytd)
            df_ytd.columns = [col + '_ytd' if col not in {'y', 'secu'} else col for
                              col in df_ytd.columns]
        if not pr_empty:
            filter_pr = {
                "dt"  : {"$in": dt_index[0:-1]},
                "secu": {"$in": all_stocks}}
            df_pr = form_mongo_to_df('pr', pr_pos, filter_pr)

        if tech_pos:
            filter_tech = {
                "dt"  : {"$in": dt_index[0:-1]},
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
        for col in list(set(raw_fac.columns).difference(set(['secu','dt']))):
            try:
                raw_fac.loc[:,col]=raw_fac.loc[:,col].where(raw_fac.loc[:,'dt']>=df_fac.loc[col,'sdt'],np.nan)
            except:
                pass
            # raw_fac.loc[:,col]=raw_fac.loc[:,col].where(raw_fac.loc[:,'dt']>=df_fac.loc[col,'sdt'],np.nan)
        raw_fac = raw_fac.set_index(['dt', 'secu']).sort_index()
        ret = raw_fac.loc[multi_index, :]
    if fac_lst2 != []:
        df_fac = fac_info[fac_info['code'].isin(fac_lst1)]
        df_fac.loc[:,'sdt']=df_fac.loc[:,'sdt'].apply(lambda x:str(parse(x).date()))
        pth = os.path.join('E:/SUNJINGJING/Strategy/ALpha/data/origin','N001001.csv')
        temp = pd.read_csv(pth)
        # temp['dt'] = temp['dt'].apply(lambda x: datetime.datetime.strptime(x,"%Y/%m/%d").strftime('%Y-%m-%d'))
        temp.loc[:, 'dt'] = pd.DatetimeIndex(temp.loc[:, 'dt']).to_series().apply(lambda dt: str(dt.date()))
        temp['secu'] = temp['secu'].apply(lambda x: x[2:]+'_SH_EQ' if x[2:][0] in ['6'] else x[2:]+'_SZ_EQ')
        df = temp.set_index(['dt','secu']).sort_index()
        ret2=df.loc[multi_index,:]
    # 合并数据
    if fac_lst1 == []:
        return ret2
    elif fac_lst2 == []:
        return ret
    else:
        return pd.concat([ret,ret2],axis=1)


@timethis
def get_term_return(all_stocks, dt_index, multi_index):
    """
    逐个调仓期,根据当期的股票代码（历史成份股记录）读取下期收益率
    收益率 = (下期股价-当期股价)/当期股价
    @返回：
    ret: dict, {dt: df}
    """
    get_sql = GetSql()
    get_sql.set_isIndex(False)
    df_pr = get_sql.get_in_spec_dates(all_stocks, dt_index)
    df_ret = np.round(df_pr.pct_change().shift(-1).T, 6)

    df_ret.columns = [str(x.date()) for x in df_ret.columns]
    df_ret = df_ret.T.stack().to_frame()
    df_ret.index.names = ['dt', 'secu']
    df_ret.columns = ['ret']
    ret = df_ret.ix[multi_index, :]
    return ret




@timethis
def get_cap_data(dt_index, multi_index, all_stocks):
    """

    """
    _, _, tb = set_mongo_cond(
        db_name='metrics', tb_name='comm_idx_price_his_a')

    t0 = time.time()
    filters = {
        "dt"  : {"$in": dt_index[:-1]},
        "secu": {"$in": all_stocks}
    }
    projections = {"_id": 0, "cap": "$tfc", "secu": 1, "dt": 1}
    pipeline = [{"$match": filters}, {"$project": projections}]
    all_rec = tb.aggregate(pipeline)
    caps = pd.DataFrame(list(all_rec)).set_index(['dt', 'secu'])
    caps = caps.loc[multi_index, :]
    t1 = time.time()
    logger.debug('find once cost {}'.format(t1 - t0))
    return caps

def get_report_date():
    """
    获取不在因子库中的其他财务指标
    """
    return
