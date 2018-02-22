# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from .config import TRADE_CAL_PATH


def get_trade_calendar():
    trade_cal = pd.read_csv(TRADE_CAL_PATH, names=['date_time', 'total_day'],
                            index_col=[0], parse_dates=True)
    return trade_cal

def get_nearest_trade_day(dt):
    trade_cal=get_trade_calendar()
    if isinstance(dt,list):
        return [str(trade_cal.index[trade_cal.index.searchsorted(str(dt))].date()) for i in dt]
    else:
        return str(trade_cal.index[trade_cal.index.searchsorted(str(dt))].date())

def get_next_n_trade_day(dt,n):
    trade_cal=get_trade_calendar()
    if isinstance(dt,list):
        return [str(trade_cal.index[trade_cal.index.searchsorted(str(dt))+n].date()) for i in dt]
    else:
        return str(trade_cal.index[trade_cal.index.searchsorted(str(dt))+n].date())

def get_before_n_trade_day(dt,n):
    trade_cal=get_trade_calendar()
    if isinstance(dt,list):
        return [str(trade_cal.index[trade_cal.index.searchsorted(str(dt))-n].date()) for i in dt]
    else:
        return str(trade_cal.index[trade_cal.index.searchsorted(str(dt))-n].date())

def form_dt_index(start_date, end_date, freq):
    """
    获取每个调仓期的具体日期：
    self.freq=='m' 则返回每月最后一个交易日日期；
    self.freq=='w' 则返回每周最后一个交易日日期；
    self.freq=='q' 则返回每季最后一个交易日日期；
    self.freq=='d' 则返回最一个交易日日期；
    返回列表['2014-01-31']
    """
    start = start_date[0:8] + '01'
    end = end_date[0:8] + '01'
    trade_cal = get_trade_calendar()
    if freq == 'd' :
        lst_trade = [str(dt)[:10] for dt in trade_cal.index.tolist()]
        start = np.searchsorted(lst_trade, start_date)
        end = np.searchsorted(lst_trade, end_date) + 1  # 取闭集
        ret = lst_trade[start:end]
    else:
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
    print('Done!')

