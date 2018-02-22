# coding: utf8

import csf_utils
from alpha_util.csf_alpha import *
from alpha_util.alpha_functions import phil_single_factor_analysis

def sort_and_na(df):
    df=df.rank(method='first', na_option='keep')
    df2=df+df.isnull().sum()
    df3=df2.fillna(0.0)
    return df3

def __frame_to_rank(biggest_best, standardized_factor):
    standardized_factor_copy = standardized_factor.copy()
    # transfer True to -1, False to 1
    asc = {}
    for k, v in biggest_best.items():
        new_value = -1 if v else 1
        asc[k] = new_value
    standardized_factor_copy *= pd.Series(asc)
    return standardized_factor_copy.sort_index(ascending=False)

def equal_weighted_scoring(standardized_factor,biggest_best=None):
    dts = standardized_factor.index.levels[0]
    columns = standardized_factor.columns
    weights = pd.DataFrame(1, index=dts, columns=columns)

    standardized_factor_copy = __frame_to_rank(biggest_best, standardized_factor)
    # rank, the smallest get 1, i.e. ascending = True
    standardized_factor_copy=standardized_factor_copy.sort_index()     #先按照股票代码排序
    __sort=lambda x : sort_and_na(x)  # Na 的处理办法
    ret1=standardized_factor_copy.groupby(level=0).apply(__sort)
    ret = (ret1* weights).mean(axis=1)
    return ret


def multi_factor_analysis(all_data,all_stocks,single_factor_analysis_results,freq,factor_names,biggest_best,
                          num_group=5, comb_name=None, score_method='eqwt', score_window=12,
                          fac_ret_gp="Q_LS",g_sell='Q_LS'):
    """
    fac_names: 多因子组合的因子代码列表
    comb_name: 自定义组合名称

    Args:
    biggest_best:
    """
    df_com_score = factor_scoring(factor_names, all_data['standardized_factor'], single_factor_analysis_results,
                                  score_method=score_method,score_window =score_window, fac_ret_gp=fac_ret_gp,
                                  biggest_best=biggest_best)

    if comb_name is not None:
        df_com_score.name = comb_name

    name = df_com_score.name

    df_com_score = pd.concat([df_com_score,
                              all_data['fac_ret_cap'].loc[:,
                              ['ret', 'cap']]], axis=1)

    ans = phil_single_factor_analysis(factor_name=name,
                                 fac_ret_cap=df_com_score,
                                 freq=freq,
                                 g_sell=g_sell,
                                 benchmark_term_return=all_data[
                                     'benchmark_term_return'],
                                 num_group=num_group,
                                 all_stocks=all_stocks, ascending=False)
    return ans

def score_analysis(df_com_score,fac_ret_cap,benchmark_term_return,all_stocks, freq, factor_names,
                              num_group=5, comb_name=None, g_sell='Q_LS'):
    """
    df_com_score:打分结果
    comb_name: 自定义组合名称
    """
    if comb_name is not None:
        df_com_score.columns=[comb_name]

    name = comb_name

    df_com_score = pd.concat([df_com_score,fac_ret_cap.loc[:,['ret', 'cap']]],axis=1)

    ans = phil_single_factor_analysis(factor_name=name,
                                      fac_ret_cap=df_com_score,
                                      freq=freq,
                                      g_sell=g_sell,
                                      benchmark_term_return=benchmark_term_return,
                                      num_group=num_group,
                                      all_stocks=all_stocks, ascending=False)
    return ans



def parallel_run_single_factor_analysis(factor_codes,all_data,freq,g_sell,turnover_method,ic_method,
                                        return_mean_method,num_group,fp_month,g_buy,sam_level,
                                        all_stocks,ascending,n_jobs=1):
    ret = Parallel(n_jobs=n_jobs)(
        delayed(single_factor_analysis)(
            factor_name,
            fac_ret_cap=all_data['fac_ret_cap'],
            freq=freq,
            g_sell=g_sell,
            benchmark_term_return=all_data['benchmark_term_return'],
            turnover_method=turnover_method,
            ic_method=ic_method,
            return_mean_method=return_mean_method,
            num_group=num_group,
            fp_month=fp_month,
            g_buy=g_buy,
            sam_level=sam_level,
            all_stocks=all_stocks,
            ascending=ascending[factor_name]
        ) for factor_name in factor_codes)

    ret = dict(list(zip(factor_codes, ret)))
    return ret

if __name__=='__main__':
    ##组合分析####
    # from alpha_util.function import score_analysis
    # comb_name = 'Nonlinear'
    # benchmark_term_return = ins.all_data['benchmark_term_return']
    # score_analysis = score_analysis(df_score, ins.all_data['fac_ret_cap'], benchmark_term_return, all_stocks, freq,
    #                                 fac_codes1, num_group=num_group, comb_name=comb_name, g_sell=g_sell)
    # score_analysis.IC_analysis.IC_series.to_hdf('IC_series.hd5', 'df')
    # score_analysis.return_analysis.group_return_cumulative.to_hdf('group_return_cumulative.hd5', 'df')
    print('Done!')