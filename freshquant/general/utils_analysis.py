# -*- coding: utf-8 -*-
__author__ = 'jessica.sun'

import numpy as np
import pandas as pd

def get_factor_name(fac_ret):
    """
    在单因子分析中,有因子,因子对应的下期收益率, benchmark下期收益率, 市值, 分组列,本函数取得因子列的列名称.

    Args:
        fac_ret: 一个数据框

    Returns:
        str: 因子名称
    """
    keep_columns = ['cap', 'benchmark_returns', 'ret', 'group']
    factor_name = set(fac_ret.columns) - set(keep_columns)
    # assert len(factor_name) == 1, "there should be only one factor, got {}".format(factor_name)
    factor_name = factor_name.pop()
    return factor_name

def return_analysis(fac_ret_data_, stock_weight_method='eqwt', plot=False):
    """
    收益率分析
    Args:
        fac_ret_data (DataFrame): 含有因子,市值,收益率,分组的数据框.且分组的列名称为'group'
        stock_weight_method : 股票等权'eqwt',股票市值加权'capwt'
        plot (bool): 是否画图
    Returns:
        ReturnAnalysis
    Raises:
        ValueError, 当bench_returns index 不能包含(覆盖)fac_ret_returns
    """
    fac_ret_data = fac_ret_data_[['score', 'cap', 'ret', 'benchmark_returns', 'group']]
    benchmark_returns = fac_ret_data.groupby(
        level=0)['benchmark_returns'].head(1).reset_index(level=1, drop=True)
    grouped = fac_ret_data.groupby([fac_ret_data.index.get_level_values(0), fac_ret_data['group']])
    if stock_weight_method == 'eqwt':
        group_mean = grouped.apply(lambda x: x[['ret']].mean())
    elif stock_weight_method == 'capwt':
        group_mean = grouped.apply(lambda x: (x.ret * x.cap).sum() / x.cap.sum())
        group_mean = pd.DataFrame(group_mean, columns=['ret'])
    group_mean = group_mean.to_panel()['ret']
    group_mean['Q_LS'] = group_mean.ix[:, 0] - group_mean.ix[:, -1]
    return_stats = pd.DataFrame()
    for col in group_mean.columns:
        return_stats[col] = return_perf_metrics(
            group_mean[col], benchmark_returns)

    ret = ReturnAnalysis()
    ret.benchmark_return = benchmark_returns
    ret.return_stats = return_stats
    ret.group_mean_return = group_mean
    ret.group_cum_return = (group_mean + 1).cumprod() - 1

    if plot:
        plot_ret(ret)
    return ret

def information_coefficient_analysis(fac_ret_data_, ic_method='normal', plot=False):
    """
    信息系数（IC）分析

    Args:
        fac_ret_data (DataFrame): 含有因子,市值,收益率,分组的数据框.且分组的列名称为'group'
        ic_method (str): ic计算方法, 有normal, rank, rank_adj
        plot (bool): 是否画图
    Returns:
        ICAnalysis
    """
    fac_ret_data = fac_ret_data_[['score', 'cap', 'ret', 'benchmark_returns', 'group']]
    factor_name = get_factor_name(fac_ret_data)
    ic_series = fac_ret_data.groupby(level=0).apply(
        lambda frame: information_coefficient(frame[factor_name], frame['ret'], ic_method))
    ic = ic_series.map(lambda e: e[0])
    p_value = ic_series.map(lambda e: e[1])
    ic_series = pd.DataFrame({'ic': ic, 'p_value': p_value})
    ic_decay = IC_decay(fac_ret_data)

    group_ic = fac_ret_data.groupby(level=0).apply(lambda frame: frame.groupby('group').apply(
        lambda df: information_coefficient(df[factor_name], df['ret'], ic_method)))
    group_ic_ic = group_ic.applymap(lambda e: e[0])
    group_ic_p_value = group_ic.applymap(lambda e: e[1])
    group_ic = pd.Panel({'ic': group_ic_ic, 'p_value': group_ic_p_value})

    ic_statistics = pd.Series({'IC_mean': ic.mean(), 'p_mean': p_value.mean(),
                               'IC_Stdev': ic.std(),
                               'IC_IR': ic.mean() / ic.std()})

    ret = ICAnalysis()
    ret.IC_series = ic_series
    ret.IC_decay = ic_decay
    ret.IC_statistics = ic_statistics
    ret.groupIC = group_ic

    if plot:
        plot_ic(ret)
    return ret

def IC_decay(fac_ret_cap_):
    """
    信息系数衰减, 不分组
    Args:
        fac_ret_cap (DataFrame): 一个Multiindex数据框
    Returns:
        DataFrame: ic 衰退
    """
    fac_ret_cap = fac_ret_cap_[['score', 'cap', 'ret', 'benchmark_returns', 'group']]
    grouped = fac_ret_cap.groupby(level=0)
    n = len(grouped)
    lag = min(n, 12)

    factor_name = get_factor_name(fac_ret_cap)

    rets = []
    dts = [dt for dt, _ in grouped]
    frames = (frame.reset_index(level=0, drop=True) for _, frame in grouped)
    for piece_data in window(frames, lag, longest=True):
        ret = [information_coefficient(df_fac.loc[:, factor_name], df_ret.loc[:, 'ret'])[0]
               if df_ret is not None else np.nan
               for df_fac, df_ret in zip([piece_data[0]] * lag, piece_data)]
        rets.append(ret)

    columns = [''.join(['lag', str(i)]) for i in range(lag)]
    df = pd.DataFrame(rets, index=dts[:len(rets)], columns=columns)
    decay = df.mean().to_frame()
    decay.columns = ['decay']
    return decay

def turnover_analysis(fac_ret_data_, turnover_method='count', fp_month=5, plot=False):
    """
    换手率分析
    Args:
        fac_ret_data (DataFrame): 一个Multi index 数据框, 含有factor, ret, cap, group列
        turnover_method (str): count or cap_weighted
        fp_month: 计算信号衰退与翻转的最后fp_month个月
        plot (bool): 是否画图
    Returns:
        TurnoverAnalysis
    """
    fac_ret_data = fac_ret_data_[['score', 'cap', 'ret', 'benchmark_returns', 'group']]
    ret = TurnOverAnalysis()

    # code_and_cap: index:dts, columns:groups, elements:dict, keys-->tick,
    # values-->cap
    code_and_cap = (fac_ret_data.groupby([fac_ret_data.index.get_level_values(0), fac_ret_data.group])
                    .apply(lambda frame: dict(list(zip(frame.index.get_level_values(1), frame['cap']))))
                    .unstack()
                    )

    def __count_turnover(current_dict, next_dict):
        current_codes = set(current_dict.keys())
        next_codes = set(next_dict.keys())
        try:
            ret = len(next_codes - current_codes) * 1.0 / len(current_codes)
        except ZeroDivisionError:
            ret = np.inf
        return ret

    def __capwt_turnover(current_dict, next_dict):
        current_df = pd.Series(current_dict, name='cap').to_frame()
        current_weights = current_df.cap / current_df.cap.sum()
        next_df = pd.Series(next_dict, name='cap').to_frame()
        next_weights = next_df.cap / next_df.cap.sum()

        cur, nxt = current_weights.align(
            next_weights, join='outer', fill_value=0)
        ret = (cur - nxt).abs().sum() / 2
        return ret

    def auto_correlation(fac_ret_data_):

        factor_name = get_factor_name(fac_ret_data_)

        grouped = fac_ret_data_.groupby(level=0)
        n = len(grouped)
        lag = min(n, 12)
        dts = sorted(fac_ret_data.index.get_level_values(0).unique())
        group_names = sorted(grouped.groups.keys())
        table = []
        for idx in range(0, n - lag):
            rows = []
            for l in range(idx + 1, idx + 1 + lag):
                current_frame = (grouped.get_group(group_names[idx])
                                 .reset_index()
                                 .set_index('code')[factor_name].dropna())
                next_frame = (grouped.get_group(group_names[l])
                              .reset_index()
                              .set_index('code')[factor_name].dropna())
                x, y = current_frame.align(next_frame, join='inner')
                rows.append(pearsonr(x.values, y.values)[0])
            table.append(rows)
        auto_corr_ = pd.DataFrame(
            table, index=dts[:(n - lag)], columns=list(range(1, lag + 1)))
        return auto_corr_

    def signal_decay_and_reversal():
        data = (fac_ret_data.groupby([fac_ret_data.index.get_level_values(0), fac_ret_data.group])
                .apply(lambda frame: frozenset(frame.index.get_level_values(1)))
                .unstack()
                )
        group_buy = data.iloc[:, 0]
        group_sell = data.iloc[:, -1]
        n = len(data)
        lag = min(n, fp_month)

        decay = list(map(lambda x, y=group_buy[-lag]: len(x.intersection(y)) / len(x), group_buy[-lag:]))
        reversal = list(map(lambda x, y=group_sell[-lag]: len(x.intersection(y)) / len(x), group_sell[-lag:]))

        return pd.DataFrame({'decay': decay, 'reversal': reversal}, index=data.index[-lag:])

    method = __count_turnover if turnover_method == 'count' else __capwt_turnover

    dts = fac_ret_data.index.get_level_values(0).unique()[:-1]
    results = {}
    for group in code_and_cap.columns:
        group_ret = []
        for idx, dic in enumerate(code_and_cap.ix[:-1, group]):
            current_dic = dic
            next_dic = code_and_cap.ix[idx + 1, group]
            group_ret.append(method(current_dic, next_dic))
        results[group] = group_ret

    turnov = pd.DataFrame(results, index=dts)

    auto_corr = auto_correlation(fac_ret_data)
    ret.auto_correlation = auto_corr
    ret.turnover = turnov
    ret.buy_signal = signal_decay_and_reversal()
    if plot:
        plot_turnover(ret)
    return ret

def code_analysis(fac_ret_data_, plot=False):
    """
    选股结果分析
    含有因子,市值,收益率,分组的数据框.且分组的列名称为'group'

    Args:
        fac_ret_data (DataFrame):  一个Multi index 数据框, 含有factor, ret, cap, group列
        plot (bool): 是否画图
    Returns:
        CodeAnalysis result
    """
    fac_ret_data = fac_ret_data_[['score', 'cap', 'ret', 'benchmark_returns', 'group']]
    ret = CodeAnalysis()

    grouped = fac_ret_data.groupby(
        [fac_ret_data.index.get_level_values(0), fac_ret_data.group])

    # index:dt, columns:group
    stocks_per_dt_group = grouped.apply(
        lambda frame_: tuple(frame_.index.get_level_values(1))).unstack()

    mean_cap_per_dt_group = grouped.apply(
        lambda frame_: frame_['cap'].mean()).unstack()  # index:dt, columns:group

    mean_cap_per_group = mean_cap_per_dt_group.mean()

    stocks = sorted(fac_ret_data.index.get_level_values(1).unique())

    industries = get_industries(stocks)

    industries = pd.concat(industries)
    industries.loc[:, 'code'] = industries.code.str.slice(0, 6)
    industries_dict = dict(list(zip(industries.code, industries.level2_name)))

    # code ---> industry
    industries_per_dt_group = stocks_per_dt_group.applymap(
        lambda tup: tuple(industries_dict[t] for t in tup))

    # industry tuple ---> Counter
    counter = industries_per_dt_group.applymap(lambda tup: Counter(tup))

    # counter ----> percent
    counter_percent = counter.applymap(
        lambda dic: {k: v * 1.0 / sum(dic.values()) for k, v in dic.items()})

    dic_frame = {}
    for col in counter_percent.columns:
        frame = pd.DataFrame(
            counter_percent[col].tolist(), index=counter_percent.index).fillna(0)
        frame = frame[
            list(frame.iloc[0, :].sort_values(ascending=False).index)]
        dic_frame[col] = frame

    # 行业平均占比: 所有分组, 所有dt合并到一起
    industries_total = Counter(industries_per_dt_group.sum().sum())
    industries_total = {str(k): v for k, v in industries_total.items()}
    industries_total = pd.Series(industries_total).sort_values(ascending=False)

    ret.cap_analysis = mean_cap_per_dt_group
    ret.industry_analysis = IndustryAnalysis(
        gp_mean_per=industries_total, gp_industry_percent=dic_frame)
    ret.stock_list = stocks_per_dt_group
    if plot:
        plot_code_result(ret)
    return ret

def risk_analysis(fac_ret_data_):
    import statsmodels.formula.api as smf
    fac_ret_data = fac_ret_data_.query('group == "Q01"')
    fac_ret_data = fac_ret_data.drop(['benchmark_returns', 'group', 'score', 'cap'], axis=1)
    cols = fac_ret_data.drop('ret', axis=1).columns
    formula = 'ret~' + '+'.join(cols)
    risk = fac_ret_data.groupby(level=0).apply(lambda x: smf.ols(formula, x.reset_index(level=0, drop=True)).fit())
    params = risk.apply(lambda x: x.params)
    resid = risk.apply(lambda x: x.resid).stack()
    return risk