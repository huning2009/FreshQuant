# -*- coding: utf-8 -*-
__author__ = 'jessica.sun'

import numpy as np
import pandas as pd
import datetime
from csf_utils import get_mongo_connection
from utils_strategy.general.trade_calendar import get_trade_calendar
from utils_strategy.general.get_stock_data import general_form_mongo_to_df
from sqlalchemy import create_engine


def summary(self):
    "统计各个月份的事件数量"
    df = self.data
    df.dt = df.dt.apply(pd.Timestamp)
    df = df.drop_duplicates(['dt', 'secu'])
    df = df.set_index('dt')
    ret = {str(y) + str(m).zfill(2): [data.count()] for (y, m), data in
           df.secu.groupby([df.index.year, df.index.month])}
    ret = pd.DataFrame(ret).T
    ret.columns = ['num']
    return ret

def form_df_to_selected(df):
    """
    df:dt和secu两列
    """
    # 去退市
    all_stocks=list(df.secu.unique())
    df_edt=general_form_mongo_to_df(host='122.144.134.95',db_name='ada',tb_name='base_stock',
                                    pos=['code','ls.edt'],filters={"code":{"$in":all_stocks}}).dropna()
    df=df[~ df.secu.isin(df_edt.code.tolist())]

    df.secu = df.secu.map(lambda x: x if isinstance(x, list) else [x])
    df_result = df[['dt', 'secu']].set_index('dt').sort_index()
    df_result = df_result.groupby(level=0).sum()
    df_result = df_result.applymap(
        lambda x: [secu for secu in x if secu[-6:] in ['_SH_EQ', '_SZ_EQ'] and secu[0] in ['0', '3', '6']])
    stocks_buy = df_result.applymap(lambda x: tuple(
        [secu[:6] + '.XSHG' if secu.startswith('6') else secu[:6] + '.XSHE' for secu in x]))  # 股票code改为6位
    return stocks_buy

def align_single_to_trading(dt):
    """
    把单个事件日期映射到其最近的一个交易日
    dt ---> 离dt最近的下一个交易日
    """
    trade_cal = get_trade_calendar()
    return str(trade_cal.index[trade_cal.index.searchsorted(str(dt))].date())

def align_to_trading_days(df):
    """
    无效：把一列事件日期分别映射到其最近的一个交易日
    """
    trade_cal = get_trade_calendar()
    return df.applymap(lambda dt:str(trade_cal.index[trade_cal.index.searchsorted(str(dt))].date()))

def filter_df(row):
    """
    标记一字涨停板
    """
    if (np.isclose(row.open, row.high) and np.isclose(row.open, row.low) and np.isclose(row.open, row.close) and row.inc>0):
        return True
    else:
        return False

def get_event_data(event_id, begt, endt):
    """
    获取公告事件，event_id是公告事件代码,begt是起始日期，endt是截止日期
    获取指定日期范围内的某些事件
    """
    _, db, tb = get_mongo_connection(host='122.144.134.95', db_name='news', tb_name='announcement')
    filters = {'typ': event_id, "pdt": {"$gte": begt, "$lte": endt}}
    projections = {"pdt": "$pdt", 'stock': "$secu.cd"}
    pipeline = [{"$match": filters}, {"$project": projections}, {'$unwind': "$stock"}]  # , { '$unwind' : "$stock" }
    all_rec = tb.aggregate(pipeline)
    rec_lst = list(all_rec)
    df = pd.DataFrame(rec_lst)

    def transform_date(time_stamp):
        if time_stamp.time() < datetime.time(9, 30, 0):
            return time_stamp.date() - datetime.timedelta(1)
        else:
            return time_stamp.date()

    if len(df) > 0:
        df = df[df.stock.apply(len) == 12]  # 仅仅取得沪深股市的股票，不包括港股
        df.loc[:, 'pdt'] = df.pdt.apply(transform_date)
        df = df.sort_values(by=['pdt'])
        df = df.reset_index(drop=True)
        df = df[df.stock.apply(lambda s: s[7:12] in ['SZ_EQ', 'SH_EQ'])]
        df.stock = df.stock.apply(lambda s: s[:6])
        df = df[df.stock.apply(lambda s: s[0] in ['0', '3', '6'])]
        del df['_id']
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
    return df

def get_before_and_after_n_days(row, N, pr,index=False):
    """
    取得单个事件发生前后n天的绝对收益率
    """
    dt = row['dt']
    stock = row['secu'] if index is False else pr.tick[0]
    trade_cal = get_trade_calendar()
    idx_dt = trade_cal.index.searchsorted(dt)
    before_N = str(trade_cal.index[idx_dt - N].date())
    after_N = str(trade_cal.index[idx_dt + N].date())
    # 防止个别日期取不到，导致所得到的行情少于2N+1
    dt_inx=trade_cal.iloc[idx_dt-N:idx_dt+N+1].index
    df_inx=pd.DataFrame([str(dt.date()) for dt in dt_inx],columns=['dt'])
    query_string = "tick == '{tick}' & dt>='{before}' & dt <='{after}'".format(tick=stock, before=before_N,after=after_N)
    temp=df_inx.merge(pr.query(query_string).sort_values(by='dt'),on='dt',how='outer')
    incs = temp.inc.to_frame().T.reset_index(drop=True)
    incs.columns=[str(i) for i in range(-N, N + 1)]
    incs.index=[dt]
    return incs

def get_after_n_days(row, N,pr,index=False):
    """
    取得单个事件发生后n天的绝对收益率
    """
    dt = row['dt']
    stock = row['secu'] if index is False else pr.tick[0]
    e=row['typ']
    print (dt,stock)
    trade_cal = get_trade_calendar()
    idx_dt = trade_cal.index.searchsorted(dt)
    before_N = dt
    after_N = str(trade_cal.index[idx_dt + N].date())
    # 防止个别日期取不到，导致所得到的行情少于N+1
    dt_inx=trade_cal.iloc[idx_dt:idx_dt+N].index.map(lambda x:str(x.date()))
    # df_inx=pd.DataFrame([str(dt.date()) for dt in dt_inx],columns=['dt'])
    incs=pr.loc[dt_inx.tolist(),['inc']].T
    incs.columns=[str(i) for i in range(1, N + 1)]
    incs.index=[e]
    # incs.index=pd.MultiIndex.from_tuples([(dt,stock)],names=('pdt','stock'))
    return incs

