# -*- coding: UTF-8 -*-

from alpha_util.alpha_config import *
import pandas as pd
import csf
import os

# def get_fac_ret_cap(fac_codes=None,start_date=None):
#     fac_ret_cap=pd.read_hdf(FAC_RET_CAP)
#     if fac_codes is not None:
#         fac_ret_cap=fac_ret_cap[fac_codes]
#     if start_date is not None:
#         fac_ret_cap=fac_ret_cap.loc[start_date:]
#     return fac_ret_cap
#
def get_fac_info():
    df = pd.read_csv(FACTORS_DETAIL_PATH)
    code = df[df['stat'] == 1][['name','ascend', 'code']]
    fac_info = code.set_index('code')
    return fac_info

def get_fac_ascend():
    df = pd.read_excel(FACTORS_DETAIL_PATH)
    code = df[df['stat'] == 1][['name','ascend', 'code']]
    fac_info = code.set_index('code')
    return dict(fac_info['ascend'])

def get_benchmark(code='000905',start_date=None,end_date=None):
    if code == '881001':
        if os.path.exists(WIND_A):
            ret=pd.read_hdf(WIND_A)
    else:
        close=csf.get_index_hist_bar(code,start_date=start_date,end_date=end_date,field=['close'])
        ret=close.pct_change().dropna()
        # 改dt为字符串
        ret.index=[str(dt)[:10] for dt in ret.index]
    return ret

def get_trade_calendar():
    trade_cal = pd.read_csv(TRADE_CAL_PATH, names=['date_time', 'total_day'],
                            index_col=[0], parse_dates=True)
    return trade_cal

def form_dt(start_date, end_date, freq):
    """
    获取每个调仓期的具体日期：
    self.freq=='m' 则返回每月最后一个交易日日期；
    self.freq=='w' 则返回每周最后一个交易日日期；
    self.freq=='q' 则返回每季最后一个交易日日期；
    返回列表['2014-01-31']
    """
    start = start_date[0:8] + '01'
    end = end_date[0:8] + '01'
    trade_cal = get_trade_calendar()
    current_calendar = trade_cal.ix[start_date:end_date, :]
    funcs = {
        'w': lambda x: x.week,
        'm': lambda x: x.month,
        'q': lambda x: x.quarter
    }
    ret = [str(data.index[-1].date()) for (year, func), data in
           current_calendar.groupby([current_calendar.index.year,funcs[freq](current_calendar.index)])]
    return ret


if __name__=='__main__':
    from .csf_alpha import *
    fac_codes=get_fac_info().index.tolist()
    start_date='2007-01-01'
    end_date='2016-07-31'
    index='881001'
    ins=CSFAlpha(index,fac_codes,start_date,end_date,'881001',freq='m',isIndex=True,
                 turnover_method='count',ic_method='normal',return_mean_method='eqwt',
                 num_group=30,fp_month=12,g_buy='Q1',g_sell='Q30',sam_level=1,
                 remove_extreme_value_method='mad',scale_method='normal',ascending=None)
    ins.all_stocks.to_csv('/media/jessica/00001D050000386C/SUNJINGJING/Strategy/ALpha/data/all_stocks.csv')
    # ins.get_all_data()
    # all_data=ins.all_data
    # all_data['fac_ret_cap'].to_hdf(('/media/jessica/00001D050000386C/SUNJINGJING/'
    #                                 'Strategy/ALpha/data/origin/fac_ret_cap_{}.hd5').format(index), 'df')
    # all_data['df_raw_factor'].to_hdf(('/media/jessica/00001D050000386C/SUNJINGJING/'
    #                                   'Strategy/ALpha/data/origin/df_raw_factor_{}.hd5').format(index), 'df')
    print('Done!')