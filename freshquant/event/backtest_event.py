# -*- coding: utf-8 -*-
__author__ = 'jessica.sun'
"""
获取xx指数成分股的xx事件数据（如盈利预增）
每个交易日，将昨天发布盈利预增事件公司加入买入列表
根据调仓限制和买入列表进行调仓
调仓限制
(1) 当持有股票数量n低于 N 时，单只股票建仓仓位为 1/N；当持有股票数量 n 超过 N 只时， 单只股票建仓仓位为 1/n
(2) 一旦买入，持有y个交易日
(3) 仅当持有数量低于x只时才买入股票，补满x只
"""
from freshquant.event.event import Jessica_EVENT
from rqalpha.api import *
import pandas as pd

pd.set_option("display.max_rows",10)

def get_selected_stocks(df):
    df_stocks= df.groupby(level=0).apply(lambda x: x.reset_index(level=0, drop=True).index.tolist()).to_frame()
    stocks_top = df_stocks.applymap(lambda x: tuple([secu[:6] + '.XSHG' if secu.startswith('6') else secu[:6] + '.XSHE' for secu in x]))
    stocks_top.columns=['stocks']
    return stocks_top


def init(context):
    context.max_n = 20                  # 股票持有不超过x只
    context.max_t = 30                # 一旦买入，持有y个交易日
    context.selected_stocks = get_selected_stocks(Jessica_EVENT(stg=['performance']).ret)
    context.hold_period = {}
    scheduler.run_weekly(log_cash,tradingday=-1)

def log_cash(context,dict):
    logger.info('Positions:%r'% (round(100*context.portfolio.market_value/(context.portfolio.market_value+context.portfolio.cash),3)))

def before_trading(context):
    """每个交易日前更新股票的持有日"""
    if len(context.portfolio.positions) >=1:
        for stock in context.portfolio.positions.keys():
            if stock not in context.hold_period.keys():
                context.hold_period[stock]=1
            else:
                context.hold_period[stock]+=1

def handle_bar(context,bar_dict):
    dt = str(context.now.date())
    if dt in context.selected_stocks.index:
        buylist = list(context.selected_stocks.loc[dt,:].values[0])
    else:
        buylist =[]
    rebalance(context, buylist)


def rebalance(context, buylist):
    dic=context.hold_period.copy()
    if len(dic)>=1:
        for stock, t in context.hold_period.items():
            if t == context.max_t:
                order_target_percent(stock, 0)
                del dic[stock]
    context.hold_period=dic
    num_stocks=len(context.portfolio.positions)
    if num_stocks == context.max_n or buylist == []:
        return
    b = context.max_n - num_stocks
    buylist = [s for s in buylist if s not in context.hold_period]
    for stock in buylist[:b]:
        order_value(stock, context.portfolio.portfolio_value/b)