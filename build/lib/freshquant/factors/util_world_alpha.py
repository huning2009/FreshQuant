# -*- coding: UTF-8 -*-
import numpy as np
import math
import pandas as pd
import statsmodels.formula.api as smf

class FactorData(object):
    def __init__(self, name, stocks, mean_ret, cum_ret):
        self.name = name
        self.stocks=stocks
        self.mean_ret=mean_ret
        self.cum_ret=cum_ret

def single_analyze(name,all_data,alpha,top):
    df = alpha[name]

    stocks = df.groupby(level=0).apply(
        lambda x: x.T.nlargest(top, x.T.columns[0]) / x.T.nlargest(top, x.T.columns[0]).sum())
    stocks.columns = ['weights']
    stocks.index.names=['dt','tick']

    inc = all_data['period_inc'].shift(-1)
    if isinstance(inc,pd.DataFrame):
        inc=inc['period_inc']
    inc.name = 'period_inc'
    combine = stocks.join(inc)
    mean_ret = combine.groupby(level=0).apply(lambda x: (x['weights'] * x['period_inc']).sum())
    cum_ret = (mean_ret + 1).cumprod() - 1

    factor_data= FactorData(name=name,
                            stocks=stocks,
                            mean_ret=mean_ret,
                            cum_ret=cum_ret)

    return factor_data

def risk_rejust(df,fac):
    """风格因子：个股Beta、流通市值、BP；
    行业因子：中信一级行业哑变量"""
    df.mkt=np.log10((df.mkt+0.000000000001).values.tolist())
    cols=[col for col in df.columns if col not in [fac]]
    formula='Y~'+'+'.join(cols)
    ret=pd.DataFrame(smf.ols(formula.replace('Y', fac), data=df.loc[df[fac].dropna().index, :].fillna(0)).fit().resid,
                 index=df[fac].dropna().index, columns=[fac]).loc[df.index, :]
    return ret