# coding: utf8
import os
import pickle

from .data_type import FactorData
from .util import get_factor_name
from joblib import Parallel, delayed
from .config import ANALYSIS


def single_factor_analysis(data, pipeline, params):
    """
    单个因子分析
        data (DataFrame): 一个multi_index 数据框, 有因子, 对应下期收益率, 市值列. multi_index-->level0=dt, level1=code
        pipeline(List): 前面N-1个为对数据进行处理, 最后一个元素为一个元组,元组的元素为xxx_analysis
    Examples:result = single_factor_analysis(data,pipeline,params)
    """
    X = data.copy()
    factor_names = [fac for fac in X.columns if fac not in ['ret', 'cap', 'benchmark_returns', 'group']]
    if len(factor_names) > 1:
        X = X[[col for col in X.columns if col not in factor_names[1:]]]
        print('more than one factor have been given,but only the first factor will be calculated!')

    for func in pipeline[:-1]:
        X = func(X, **(params.get(func.__name__, {})))

    result_dict = {}
    for func in pipeline[-1]:
        result_dict[func.__name__] = func(X, **(params.get(func.__name__, {})))

    factor_name = get_factor_name(X)
    factor_result = FactorData(name=factor_name, **result_dict)
    return factor_result


def parallel_single_factor(data, pipeline, params, njobs=3):
    """
    多个因子分别做单因子分析(多线程)
    data=pd.read_hdf('E:\SUNJINGJING\Strategy\ALpha\data\origin\\after_neutral_881001.hd5')
    pipeline = [score,add_group,
                (information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis)]
    params={'add_group':{'num_group':10}}
    ret=parallel_single_factor(data,pipeline,params,njobs)
    """
    X = data.copy()
    factor_names = [fac for fac in X.columns if fac not in ['ret', 'cap', 'benchmark_returns', 'group']]
    # 全部因子数据去极值、标准化、中性化、打分、分组
    for func in pipeline[:-1]:
        X = func(X, **(params.get(func.__name__, {})))
    # 多线程做单因子分析（IC、Return、Turnover）
    frame = X[['ret', 'cap', 'benchmark_returns', 'score', 'group']]
    ret = Parallel(n_jobs=njobs, backend='threading')(
        delayed(single_factor_analysis)(frame.join(data[fac]), pipeline, params)
        for fac in factor_names)
    single_factor_analysis_results = dict(list(zip(factor_names, ret)))
    return single_factor_analysis_results


def one_by_one_single_factor(data, pipeline, params):
    """
    多个因子分别做单因子分析(one by one)
    data=pd.read_hdf('E:\SUNJINGJING\Strategy\ALpha\data\origin\\after_neutral_881001.hd5')
    pipeline = [score,add_group,
                (information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis)]
    params={'add_group':{'num_group':10}}
    ret=one_by_one_single_factor(data,pipeline,params)
    """
    X = data.copy()
    factor_names = [fac for fac in X.columns if fac not in ['ret', 'cap', 'benchmark_returns', 'group']]
    # 逐个因子打分+分组+分析
    frame = X[['ret', 'cap', 'benchmark_returns']]
    for factor_name in factor_names:
        print(factor_name)
        Y= frame.join(X[factor_name])
        for func in pipeline[:-1]:
            Y = func(Y, **(params.get(func.__name__, {})))
        result_dict = {}
        for func in pipeline[-1]:
            result_dict[func.__name__] = func(Y, **(params.get(func.__name__, {})))
        factor_result = FactorData(name=factor_name, **result_dict)
        output = open(os.path.join(ANALYSIS, 'single', str(factor_name) + '_881001.pkl'), 'wb')
        pickle.dump(factor_result, output)













            # def parallel_single_factor(data,pipeline, params, njobs=3):
            #     """
            #     多个因子分别做单因子分析
            #     Examples:
            #         parallel_single_factor(data,pipeline,params,njobs=3)
            #     """
            #     frame = data[['ret', 'cap', 'benchmark_returns']]
            #     factor_name=[col for col in data.columns if col not in ['ret','cap','benchmark_returns']]
            #     ret = Parallel(n_jobs=njobs,backend='threading')(
            #         delayed(single_factor_analysis)(frame.join(data[fac]), pipeline, params)
            #         for fac in factor_name)
            #     single_factor_analysis_results = dict(list(zip(factor_name, ret)))
            #     return single_factor_analysis_results
