# -*- coding:UTF-8 -*-
import csf_utils
from alpha_util.csf_alpha import *
import pandas as pd
import os
from alpha_util.alpha_config import data_pwd

def test_single_factor(ins, ascend, n_jobs=1):
    '''
    计算单因子，验证算法
    '''
    ins.parallel_run_single_factor_analysis(n_jobs)
    single_factor_analysis_results = ins.single_factor_analysis_results[ins.factor_codes[0]]
    ret_stat = pd.DataFrame(single_factor_analysis_results.return_analysis.return_statistics)
    ret_cum = pd.DataFrame(single_factor_analysis_results.return_analysis.group_return_cumulative.loc['Q1', :])
    ret_mean = pd.DataFrame(single_factor_analysis_results.return_analysis.group_return_mean.loc['Q1', :])
    ic_sery = pd.DataFrame(single_factor_analysis_results.IC_analysis.IC_series)
    # ret_cum,ret_mean=update_ret_to_now(ret_cum,ret_mean,stocks)
    xls_name = ('single_fac_{}_{}.xls').format(ins.factor_codes, ascend)
    writer = pd.ExcelWriter(xls_name, engine='xlsxwriter', options={'encoding': 'gbk'})
    ret_stat.to_excel(writer, sheet_name='ret_stat')
    ret_cum.to_excel(writer, sheet_name='ret_cum')
    ret_mean.to_excel(writer, sheet_name='ret_mean')
    ic_sery.to_excel(writer, sheet_name='ic_sery')
    writer.save()


def cal_all_single_factor(ins, n_jobs=1):
    '''
    计算所有单因子，收益率结果汇总
    '''
    fac_lst = ins.factor_codes
    q = ins.g_sell
    ins.parallel_run_single_factor_analysis(n_jobs)
    single_factor_analysis_results = ins.single_factor_analysis_results
    dic_mean = {}
    dic_cum = {}
    dic_ic_series = {}
    dic_ic_decay = {}
    dic_ic_stats = {}
    dic_ret_stats = {}
    dict_buy_signal = {}
    for factor_name in fac_lst:
        dic_mean[factor_name] = single_factor_analysis_results[factor_name].return_analysis.group_return_mean
        dic_cum[factor_name] = single_factor_analysis_results[factor_name].return_analysis.group_return_cumulative
        dic_ic_series[factor_name] = single_factor_analysis_results[factor_name].IC_analysis.IC_series
        dic_ic_decay[factor_name] = single_factor_analysis_results[factor_name].IC_analysis.IC_decay
        dic_ic_stats[factor_name] = single_factor_analysis_results[factor_name].IC_analysis.IC_statistics
        dic_ret_stats[factor_name] = single_factor_analysis_results[factor_name].return_analysis.return_statistics
        # dict_buy_signal[factor_name] = single_factor_analysis_results[factor_name].turnover_analysis.buy_signal
    panel_mean = pd.Panel(dic_mean)
    panel_cum = pd.Panel(dic_cum)
    panel_ic_series = pd.Panel(dic_ic_series)
    panel_ic_decay = pd.Panel(dic_ic_decay)
    df_ic_stats = pd.DataFrame(dic_ic_stats)
    panel_ret_stats = pd.Panel(dic_ret_stats)
    # panel_buy_signal = pd.Panel(dict_buy_signal)
    panel_mean.to_hdf(os.path.join(analysis_pwd,'ret_group_mean.hd5'), 'df')
    panel_cum.to_hdf(os.path.join(analysis_pwd,'ret_group_cum.hd5'),'df')
    panel_ic_series.to_hdf(os.path.join(analysis_pwd,'ic_series.hd5'),'df')
    panel_ic_decay.to_hdf(os.path.join(analysis_pwd,'ic_decay.hd5'),'df')
    df_ic_stats.to_hdf(os.path.join(analysis_pwd,'ic_stats.hd5'),'df')
    panel_ret_stats.to_hdf(os.path.join(analysis_pwd,'ret_stats.hd5'),'df')
    # panel_buy_signal.to_hdf('/media/jessica/00001D050000386C/SUNJINGJING/Strategy/ALpha/data/buy_signal.hd5', 'df')

if __name__ == '__main__':
    fac_info = pd.read_csv(FACTORS_DETAIL_PATH)
    fac_info = fac_info[fac_info['stat'] == 1]
    fac_codes = fac_info['code'].tolist() # 因子列表
    print(fac_codes)
    start_date = '2007-04-01'  # 开始日期
    end_date = '2016-07-30'  # 结束日期
    codes = '881001'  # codes为股票池，bench_code为基准指数
    neutral_fac_ret_cap=pd.read_hdf(NEUTRAL_FAC_RET_CAP_881001) ## 此处传入的是中性化后的数据
    ins = CSFAlpha(codes, fac_codes, start_date, end_date, fac_ret_cap=neutral_fac_ret_cap,
                   bench_code='881001', freq='m', ic_method='normal',
                   isIndex=True, num_group=30, g_sell='Q30')
    cal_all_single_factor(ins, n_jobs=1)
    print('Done!')
