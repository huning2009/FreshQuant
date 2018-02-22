# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel,delayed
from pymongo import MongoClient
from general.get_stock_data import get_stock_industry_from_mongo
from alpha_util.alpha_config import *
# import csf
# AccessKey = '382035e5421135c15753aeb901b14454'
# SecretKey = 'qrPqt/0rAYjCt8+0XDuAlSs901s='
# csf.config.set_token(AccessKey, SecretKey)

def add_industry(df0,ind):
    '''
    输入fac_ret_cap,加上n列行业数据
    返回new_fac_ret_cap
    '''
    all_stocks=df0.index.get_level_values(1).unique().tolist()
    all_ind=get_stock_industry(all_stocks, ind)
    ind_sery=df0.index.get_level_values(1).to_series().map(all_ind)
    ind_dummies=pd.get_dummies(ind_sery)
    for col in ind_dummies.columns:
        df0[col]=ind_dummies[col].values
    print('industry has been added')
    return df0

def single_cap_neutral(col,x,df):
    print(col)
    if col not in ['ret','cap', 'M004023']:
        try:
            if df[col].isnull().sum() != len(df):
                return pd.DataFrame(sm.OLS(df[col].dropna().values, x.loc[df[col].dropna().index, :].values).fit().resid,
                                       index=df[col].dropna().index, columns=[col]).loc[df.index,:]
        except:
            return pd.DataFrame(df[col],columns=[col])
    else:
        return pd.DataFrame(df[col],columns=[col])

def cap_neutral(df):
    '风格中性'
    df=df.reset_index(level=0,drop=True)
    # df=df.dropna(how='all',axis=1)
    x = sm.add_constant(df['cap'].map(np.log))
    ret=Parallel(n_jobs=6)(delayed(single_cap_neutral)(col,x,df) for col in df.columns)  # 市值因子自身不需要
    df.update(pd.concat(ret, axis=1))
    return df

def single_ind_neutral(col,formula,df):
    print(col)
    if col not in ['ret','cap']:
        try:
            if df[col].isnull().sum()!=len(df):
                return pd.DataFrame(smf.ols(formula.replace('Y', col), data=df.loc[df[col].dropna().index, :].fillna(0)).fit().resid,
                             index=df[col].dropna().index, columns=[col]).loc[df.index,:]
        except:
            return pd.DataFrame(df[col], columns=[col])

    else:
        return pd.DataFrame(df[col],columns=[col])

def industry_neutral(df0):
    '行业中性'
    df=df0.reset_index(level=0,drop=True)
    field_x=[x for x in df.columns if x[:1] not in ['M','r','c']]
    field_fac=[x for x in df.columns if x[:1] in ['M','r','c']]
    # field_x=[u'商贸',u'非日常生活消费品',u'公用事业',u'房地产',u'工业',u'信息技术',u'能源',u'原材料',u'金融',u'医疗保健','','']
    field_dic={x1:'x'+str(x2) for (x1,x2) in zip(field_x,list(range(1,(len(field_x)+1),1)))}
    formula='Y~'
    df.columns=[field_dic[col] if (col in field_dic) is True else col for col in df.columns]
    for i in range(len(field_dic)):
        if i==0:
            formula=formula+list(field_dic.values())[i]
        else:
            formula=formula+'+'+list(field_dic.values())[i]
    # df = df.dropna(how='all', axis=1)
    field_y=[col for col in df.columns if (col not in list(field_dic.values()))]  # ret 和因子值,cap不用
    result=Parallel(n_jobs=6)(delayed(single_ind_neutral)(col,formula,df) for col in field_y)
    df.update(pd.concat(result, axis=1))
    return df[field_fac]

def __single_neutral(col,formula,df):
    print(col)
    if col not in ['ret','cap']:
        try:
            if df[col].isnull().sum()!=len(df):
                return pd.DataFrame(smf.ols(formula.replace('Y', col), data=df.loc[df[col].dropna().index, :].fillna(0)).fit().resid,
                             index=df[col].dropna().index, columns=[col]).loc[df.index,:]
        except:
            return pd.DataFrame(df[col], columns=[col])

    else:
        return pd.DataFrame(df[col],columns=[col])

def single_neutral(df0):
    df=df0.reset_index(level=0,drop=True)
    field_x=[x for x in df.columns if x[:1] not in ['M','r']]
    field_dic={x1:'x'+str(x2) for (x1,x2) in zip(field_x,list(range(1,(len(field_x)+1),1)))}

    formula='Y~'
    df.columns=[field_dic[col] if (col in field_x and col not in ['cap']) else col for col in df.columns]
    for i in range(len(field_dic)):
        if i==0:
            formula=formula+list(field_dic.values())[i]
        else:
            formula=formula+'+'+list(field_dic.values())[i]
    field_y=[col for col in df.columns if col[:1] in ['M','r','c']]
    result=Parallel(n_jobs=6)(delayed(__single_neutral)(col,formula,df) for col in field_y)
    df.update(pd.concat(result, axis=1))
    return df

def neutral(fac_ret_cap,field=None,ind='sw'):
    # 增加并保存行业数据(1)
    fac_ret_cap=add_industry(fac_ret_cap,ind)
    field=['ind','cap'] if field is None else field
    if len(field) == 1:
        if 'ind' in field:
            __industry_neutral=lambda x:industry_neutral(x)
            new_fac_ret_cap=fac_ret_cap.groupby(level=0).apply(__industry_neutral)
        else:
            __cap_neutral=lambda x:cap_neutral(x)
            new_fac_ret_cap=fac_ret_cap.groupby(level=0).apply(__cap_neutral)
    elif len(field) ==2:
        new_fac_ret_cap=fac_ret_cap.groupby(level=0).apply(lambda x:single_neutral(x))
    cols=[col for col in new_fac_ret_cap.columns if col[:1] not in ['x']]
    df=new_fac_ret_cap[cols]
    return df

if __name__=='__main__':
    fac_ret_cap= pd.read_hdf(NEUTRAL_FAC_RET_CAP_881001)
    fac_ret_cap=neutral(fac_ret_cap,ind='sw')
    print(fac_ret_cap)
    fac_ret_cap.to_hdf(NEUTRAL_FAC_RET_CAP_881001,'df')
    print('neutral has Done!')