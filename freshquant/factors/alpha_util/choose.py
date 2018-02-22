# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from alpha_util.csf_alpha import *
import itertools
from alpha_util.get_alpha_data import *
from alpha_util.stats_utils import cal_stats
from alpha_util.alpha_config import analysis_pwd

def get_all_single():
    '''
    获取所有单因子结果
    '''
    ic = pd.read_hdf(os.path.join(analysis_pwd,'ic_series.hd5'))
    cum = pd.read_hdf(os.path.join(analysis_pwd,'ret_group_cum.hd5'))
    mean = pd.read_hdf(os.path.join(analysis_pwd,'ret_group_mean.hd5'))
    stats = pd.read_hdf(os.path.join(analysis_pwd,'ret_stats.hd5'))
    return ic,cum,mean,stats

def choose_static(ic,cum,mean,stats,i=1,good_facs=True):
    ic=ic.loc[:, :, 'ic']
    code=get_fac_info()
    code_ascend=dict(code['ascend'])
    code_name= dict(code['name'])
    # 计算指标
    ic_percent=pd.DataFrame({col:[(ic[col]<=-0.03).sum()/np.float(ic[col].notnull().sum()) if code_ascend[col] == -1.0
       else (ic[col]>=0.03).sum()/np.float(ic[col].notnull().sum())] for col in ic.columns}).T
    ic_percent.columns=['ic_percent']
    cum=cum.loc[:,'Q1',:].tail(1).T
    cum.columns=['cum']
    cum=cum.sort_values(by='cum',ascending=False)
    stats=stats.loc[:, 'Q1', :].T
    combined_stas=pd.concat([ic_percent,cum,stats],axis=1)

    # 获取因子列表
    if good_facs is True:
        ic_lst = ic_percent[ic_percent['ic_percent'] > 0.5].index.tolist()
        cum_lst = cum[cum.values >= 3.0].index.tolist()
        win_lst = stats[stats['win_ratio'] >= 0.55].index.tolist()
        sharp_lst=stats[stats['sharp_ratio']>=0.35].index.tolist()
        s=set()
        if i == 1:
            for (l1) in itertools.combinations([ic_lst, cum_lst, win_lst, sharp_lst], 1):
                s = set(l1[0])| s
        if i == 2:
            for (l1,l2) in itertools.combinations([ic_lst, cum_lst, win_lst, sharp_lst],2):
                s=set(l1)&set(l2)|s
        if i == 3:
            for (l1, l2,l3) in itertools.combinations([ic_lst, cum_lst, win_lst, sharp_lst], 3):
                s = set(l1) & set(l2) & set(l3) | s
        combined_stas=combined_stas.loc[list(s)]
    return combined_stas

def choose_dynamic(benchmark,ic,ret_mean,dt,i,period=12):
    code = get_fac_info()
    dic_code = dict(code['ascend'])
    dic_code = dict(code['name'])
    ret_mean=ret_mean.loc[:,'Q1',:dt]
    benchmark_term_return = get_benchmark(benchmark).loc[ret_mean.index]
    stats=pd.concat([cal_stats(ret_mean[col],benchmark_term_return,freq='m',name=col) for col in ret_mean.columns])

    win_lst = stats[stats['win_ratio'] >= 0.55].index.tolist()
    sharp_lst = stats[stats['sharp_ratio'] >= 0.35].index.tolist()
    cum_y_lst = stats[stats['algo_cum_y'] >= 0.15].index.tolist()

    ic=ic.loc[:,:dt,'ic']
    ic_percent=pd.DataFrame({col:[(ic[col]<=-0.03).sum()/np.float(ic[col].notnull().sum()) if dic_code[col] == -1.0
       else (ic[col]>=0.03).sum()/np.float(ic[col].notnull().sum())] for col in ic.columns}).T
    ic_percent.columns=['ic_percent']
    ic_lst=ic_percent[ic_percent['ic_percent']>0.5].index.tolist()

    s=set()
    if i==1:
        for l1 in itertools.combinations([ic_lst, cum_y_lst, win_lst, sharp_lst],i):
            s=set(l1[0])|s
    if i==2:
        if i == 2:
            for (l1, l2) in itertools.combinations([ic_lst, cum_y_lst, win_lst, sharp_lst], i):
                s = set(l1) & set(l2) | s
    dic_factors={}
    for fac in list(s):
        dic_factors[fac]=dic_code[fac]
    return dic_factors




def comb_analysis(ins, factors, fac_sort):
    '''
    因子组合分析：
    '''
    mret = ins.multi_factor_analysis(fac_names=factors, comb_name='xnxn', biggest_best=fac_sort)
    mret.return_analysis.group_return_cumulative.T['Q1'].plot()
    plt.show()


if __name__ == '__main__':
    #
    df = pd.read_excel('/media/jessica/00001D050000386C/SUNJINGJING/Strategy/ALpha/data/origin/quant_dict_factors_all.xls')
    code = df[df['stat'] == 1][['name', 'ascend', 'code']]
    code = code.set_index('code')
    ic,cum,mean,stats=get_all_single()
    good_facs= choose_static(ic,cum,mean,stats,i=3,good_facs=True)
    good_facs.to_hdf(GOOD_FACS,'df')
    print('Done!')
    print('Done!')
