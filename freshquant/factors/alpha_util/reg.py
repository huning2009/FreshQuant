# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from alpha_util.function import score_analysis
'''
多因子--回归法
1. 选定因子范围
2. 分别进行单因子回归
3. 分别计算单因子指标
4. 筛选出因子
5. 因子之间的线性关系研究,剔除冗余因子
5. 收益率对各个因子回归,得到的预测值排序
'''
def reg_stats(fac_ret_cap):
    return fac_ret_cap

def alpha_reg(fac_ret_cap):
    return score

if __name__=='__main__':
    fac_info = pd.read_csv(FACTORS_DETAIL_PATH)

    start_date = '2008-04-01'  # 开始日期
    end_date = '2016-06-30'  # 结束日期
    bench_code = '000905'  # benchmark代码
    ins = CSFAlpha(bench_code, fac_codes, start_date, end_date, bench_code='000905', freq='m', ic_method='normal',
                   isIndex=True, num_group=30, g_sell='Q30')
    ins.get_all_data()
    all_stocks=ins.all_stocks
    all_data=ins.all_data
    freq=ins.freq
    fac_ret_cap=all_data['fac_ret_cap']
    benchmark_term_return=all_data['benchmark_term_return']
    fac_ret_cap=reg_stats(fac_ret_cap)
    df_score=alpha_reg(fac_ret_cap)
    score_analysis(df_score, fac_ret_cap, benchmark_term_return, all_stocks, freq,
                   fac_codes, num_group=num_group, comb_name=comb_name, g_sell=g_sell)

