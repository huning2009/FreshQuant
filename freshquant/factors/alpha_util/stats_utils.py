# -*- coding:utf-8 -*-

import math
import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, pearsonr

'''
alpha回测逻辑：
1. 单因子回测
    获取所有因子数据
    获取所有的超配股票
    对于每个因子：
        分别回测：获取analyze中每月持仓股票
        然后计算IC、收益率、换手率、以及相关的收益率指标。
2. 筛选有效单因子
3. 构建多因子策略
    因子加权
    组合加权
    策略回测：
        1） 获取多因子数据，获取多因子的超配股票
        2） 回测，获取analyze中每月持仓股票
        3） 计算策略相关指标
'''
def max_drawdown(arr):
    r_max = arr[0]
    d_max = 0
    for i in arr[1:]:
        r_max = max(i, r_max)
        d = (r_max - i) / (r_max)
        d_max = max(d, d_max)
    return d_max

def down_side_stdev(arr):
    mu = np.mean(arr)
    d_arr = [(i - mu) ** 2 for i in arr if i < mu]
    ret = np.sqrt(np.sum(d_arr) / len(arr))
    return ret

def cal_stats(ret_mean,benchmark_term_return,freq='m',name=None):
    ret_mean=pd.DataFrame(ret_mean)
    ret_mean.columns=['algo_mean']
    benchmark_term_return=benchmark_term_return.loc[ret_mean.index]
    benchmark_term_return=pd.DataFrame(benchmark_term_return)
    benchmark_term_return.columns=['benchmark_term_return']
    FREQ_NUM = {'w': 52, 'm': 12, 'q': 4,'d':250}
    df_stats = pd.DataFrame()
    df_stats['algo_mean'] = ret_mean.mean()
    g_ret_cum = (ret_mean + 1).cumprod() - 1
    bench_ret_cum = (benchmark_term_return + 1).cumprod() - 1
    df_stats['algo_cum'] = g_ret_cum.ix[-1, :]
    df_stats['bench_cum'] = bench_ret_cum.ix[-1, :]
    y_dic = FREQ_NUM
    y_num = y_dic[freq]
    y_factor = np.float(y_num)/len(ret_mean.index)
    # 累计收益率(年化)
    df_stats['algo_cum_y'] = (1 + df_stats['algo_cum']) **(y_factor) - 1
    df_stats['bench_cum_y'] = (1 + df_stats['bench_cum']) ** (y_factor) - 1
    # 回测区间最大收益率
    df_stats['min'] = ret_mean.min()
    # 回测区间最小收益率
    df_stats['max'] = ret_mean.max()
    # 标准差
    sigma = ret_mean.std()
    # 标准差（年化）
    df_stats['sigma_y'] = sigma * np.sqrt(y_num)
    # 下行标准差（年化）
    df_stats['d_sigma'] = ret_mean.apply(lambda x: down_side_stdev(x) * np.sqrt(y_num))
    # 最大回撤
    df_stats['max_drawdown'] = g_ret_cum.apply(
        lambda x: max_drawdown(x + 1))
    # 夏普比率
    df_stats['sharp_ratio'] = df_stats['algo_cum_y'] / df_stats['sigma_y']
    # +++++分组收益率与bench_mark作比较分析+++++
    # bench_mark 收益率
    # benchmark_term_return = self.all_data['benchmark_term_return']
    # bench_mark 累计收益率
    bench_r_cum = (benchmark_term_return + 1).cumprod() - 1
    # bench_mark 累计收益率(年化)
    bench_r_cum_y = (1 + bench_r_cum.iloc[-1]) ** y_factor - 1

    # # 分组个股收益率
    # df_stats['ex_num_ratio'] = grouped.apply(ex_num_ratio).unstack().mean()
    # 每期超额收益率
    g_ex_ret = ret_mean.sub(benchmark_term_return.benchmark_term_return, axis=0)

    # 最大超额收益率
    df_stats['ex_r_max'] = g_ex_ret.max()
    # 最小超额收益率
    df_stats['ex_r_min'] = g_ex_ret.min()
    # 跟踪误差（年化）
    df_stats['track_error'] = g_ex_ret.std() * np.sqrt(y_num)
    # 超额收益率(年化)
    df_stats['ex_r_cum_y'] = df_stats['algo_cum_y'] - bench_r_cum_y.values
    # 信息比率
    df_stats['info_ratio'] = df_stats['ex_r_cum_y'] / df_stats['track_error']
    # 胜率
    df_stats['win_ratio'] = g_ex_ret.apply(
        lambda x: sum(x > 0) / float(len(x)))
    # alpha & beta
    x = sm.add_constant(benchmark_term_return)
    try:
        df_stats['alpha'] = [
            sm.OLS(ret_mean[col].dropna().values, x.loc[ret_mean[col].dropna().index, :].values).fit().params[0] for col in
            ret_mean.columns]
        df_stats['beta'] = [
            sm.OLS(ret_mean[col].dropna().values, x.loc[ret_mean[col].dropna().index, :].values).fit().params[1] for col
            in ret_mean.columns]
    except:
        df_stats['alpha']=np.nan
        df_stats['beta']=np.nan
    columns = sorted(df_stats.columns)
    df_stats = df_stats.loc[:, columns]
    if name is not None:
        df_stats.index=[name]
    return df_stats

if __name__=='__main__':
    ret_mean=pd.DataFrame(np.random.randn(100,1),index=pd.date_range('2016-01-01',periods=100),columns=['A'])
    benchmark_term_return = pd.DataFrame(np.random.randn(100, 1), index=pd.date_range('2016-01-01', periods=100),
                            columns=['benchmark_term_return'])
    df_stats=cal_stats(ret_mean,benchmark_term_return,freq='m')
    print (df_stats)
    print ('Done')

