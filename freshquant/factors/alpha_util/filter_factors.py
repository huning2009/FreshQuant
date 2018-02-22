# coding: utf8
"""
1. 剔除共线性因子
2. 剔除特征归因失效的因子
"""

import numpy as np
from alpha_util.csf_alpha import CSFAlpha

def del_high_corr(df,corr=0.8):
    """
    本函数找出相关性大于阈值的因子,并剔除其中一个
    """
    df=df[[col for col in df.columns if col not in ['cap', 'ret']]]
    correlation=df.corr()
    bad_columns=[]
    correlation_copy = correlation.copy()
    correlation_copy = correlation_copy.abs()  # correlation_copy 取绝对值
    correlation.values[np.triu_indices_from(correlation.values,
                                            0)] = 0.0  # 把上三角（包括对角线部分）设置为0.
    if correlation.unstack().max()>corr:
        col_idx, row_idx = correlation.unstack().argmax()  # (col_idx, row_idx)
        if correlation_copy.ix[row_idx, :].mean() > correlation_copy.ix[:,
                                                    col_idx].mean():
            bad_column = row_idx
        else:
            bad_column = col_idx
        # 把该列名称从相关系数矩阵的行/列里去掉
        bad_columns.append(bad_column)
        correlation_copy.drop(bad_column, axis=0, inplace=True)
        correlation_copy.drop(bad_column, axis=1, inplace=True)
    df.drop(bad_columns,axis=1,inplace=True)
    return df

def del_attrbution_inefficient(df):
    return df

if __name__=='__main__':
    # fac_codes = ['M001007','M001001','M004023']
    fac_codes = ['M002013L','M008001','M004006','M010007','M004023','M010006']
    start_date = '2015-04-01'  # 开始日期
    end_date = '2016-06-30'  # 结束日期
    bench_code = '000300'  # benchmark代码
    ins = CSFAlpha(bench_code, fac_codes, start_date, end_date, bench_code='000300', freq='m', ic_method='normal',
                   isIndex=True, num_group=30, g_sell='Q30')
    ins.get_all_data()
    data=ins.all_data['fac_ret_cap']
    ret=del_attrbution_inefficient(data)
    print(ret)
    # ret=ins.multi_factor_analysis(fac_names=fac_codes,num_group=30,comb_name='comb',score_method='eqwt',score_window=None,biggest_best=None)