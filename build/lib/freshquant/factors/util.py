# -*- coding: utf-8 -*-
import itertools
import numpy as np
from factors.get_data import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math


def equal_interval(x):
    ##等区间，固定比例法
    x1 = x.rank(ascending=True,pct=True)
    x2 = x[(x1<0.99)*(x1>0.01)]
    return x2

def h(Zplus,Zminus,i,j,p):
    a=Zplus[i]
    b=Zminus[j]
    if a==b:
        return p-1-i-j
    else:
        return (a+b)/(a-b)

def naive_medcouple(X):
    X=X.sort_index()
    xm=X.median()
    xscale=2*X.max()
    Zplus=[]
    Zminus=[]
    for x in X.values:
        if x>=xm:
            Zplus.append((x-xm)/xscale)
        elif x<=xm:
            Zminus.append((x-xm)/xscale)
    p=len(Zplus)
    q=len(Zminus)
    H=[]
    for i in range(0,p):
        for j in range(0,q):
            H.append(h(Zplus,Zminus,i,j,p))
    return pd.DataFrame(H).median()[0]

def boxplot_adj(x):
    # x=x.dropna(axis=0,how='any')
    md=x.median()
    Q1=x[x<=md].median()#25%
    Q3=x[x>=md].median()# 75%
    IQR=Q3-Q1
    mc=naive_medcouple(x)
    if mc >=0:
        L=Q1-1.5*math.exp(-3.5*mc)*IQR
        U=Q1+1.5*math.exp(4*mc)*IQR
    elif mc<=0:
        L=Q1-1.5*math.exp(-4*mc)*IQR
        U=Q1+1.5*math.exp(3.5*mc)*IQR
    x=x[(x<U)*(x>L)]
    return x

def MAD(x2):
    # x2 = x2.dropna(axis=0,how='any')
    md = x2.median()
    MAD = pd.DataFrame(abs(x2.values - md)).median()
    up = md + 3 * MAD.values
    down = md - 3 * MAD.values
    x2 = x2[(x2< up[0]) * (x2 > down[0])]
    return x2

def STD(x2):
    # x2 = x2.dropna(axis=0,how='any')
    MEAN = x2.mean()
    STD=x2.std()
    up = MEAN + 3 * STD
    down = MEAN - 3 * STD
    x2 = x2[(x2< up) * (x2 > down)]
    return x2

def extreme_process(data, num=3, method='mad'):
    """
    :param data: pd.DataFrame or pd.Series
    :param num: int
        超过num个标准差（或MAD）即认为是极值
    :param method: str
        极值的评判标准
            'mad': mean absolute deviation
            'std': standard deviation
            'boxplot'
            'fixrate'
    """
    if isinstance(data, pd.Series):
        data_ = data.to_frame()
    else:
        data_=data.copy()
    factor_name = get_factor_name(data_)
    temp=data_[factor_name]
    temp2=data_[['cap','benchmark_returns','ret']]
    if method == 'boxplot':
        data_ =temp.apply(lambda x:boxplot_adj(x),axis=1)
        # data_=pd.concat([boxplot_adj(data_[col]) for col in factor_name],axis=1)
    elif method =='fixrate':
        data_=temp.apply(lambda x:equal_interval(x),axis=1)
    elif method=='mad':
        data_ = temp.apply(lambda x:MAD(x),axis=1)
    elif method =='std':
        data_ = temp.apply(lambda x:STD(x),axis=1)
    return pd.concat([temp2,data_],axis=1)


def data_scale(data, method='normal'):
    """
    数据标准化处理
    Args:
        data (DataFrame): 数据框
        method (str): 标准化方法
        "normal": (x-x.mean())/x.std()
        "cap": (x - cap_weighted_mean of x)/x.std()
    Returns:
        DataFrame 标准化处理后的数据
    Raises:
        ValueError
    """
    data_ = data.copy()

    factor_name = get_factor_name(data_)

    if method == 'normal':
        data_.loc[:, factor_name] = ((data_.loc[:, factor_name] - data_.loc[:, factor_name].mean())
                                     / data_.loc[:, factor_name].std())
    elif method == 'cap':
        cap_weight = (data_.loc[:, 'cap'] / data_.loc[:, 'cap'].sum()).to_frame()
        avg = (data_.loc[:, factor_name].multiply(cap_weight.loc[:,'cap'],axis=0)).sum()
        data_.loc[:, factor_name] = (data_.loc[:, factor_name] -avg) / data_.loc[:, factor_name].std()
    else:
        raise ValueError('标准化算法现在仅支持normal与cap')

    return data_


def cut_group(data_, num_group, col_name=None, ascending=False):
    """
    对于给定的数据（普通的DataFrame或Series），按照指定的列（col_name）
    进行排序（升序或降序）并分为num_group组

    :param data_: pd.Series of pd.DataFrame
    :param col_name: str
        根据该列值的大小进行排序
    :param num_group: int
        分成的组数
    :param ascending: Bool
        True: ascending, False: Descending

    :return pd.DataFrame
        在原先的data后增加一列，每个元素对应的是该行对应的组数
    """
    data = data_.copy()
    if isinstance(data, pd.DataFrame):
        data = data.loc[:, col_name]

    data_len = len(data)
    avg_element = data_len // num_group
    remains = data_len % num_group
    each_group = [avg_element] * num_group
    if remains:
        for idx in range(0, remains):
            each_group[idx] += 1
    each_group = np.array(each_group)
    each_group = each_group.cumsum()
    try:
        idx = data.rank(method='first', na_option='bottom', ascending=ascending)
    except:
        print(idx)
        print((data.index[0]))
        print(col_name)
        print('error occurred in cut_group')
    groups = pd.Series(index=idx.index)
    start = 0
    for grp, end in enumerate(each_group):
        mask = (idx > start) & (idx <= end)
        groups[mask] = ''.join(['Q', str(grp + 1)])
        # groups[mask] = grp
        start = end
    groups = groups.tolist()
    data['group'] = groups
    return groups


def get_grouped_data(fac_ret_cap, col_name, num_group, ascending):
    """
    对MultiIndex的数据(level=0)进行分组

    :param fac_ret_cap: MultiIndex pd.DataFrame
    :param col_name: str
        根据该列值的大小进行排序
    :param num_group: int
        分成的组数
    :param ascending: Bool
        True: ascending, False: Descending

    :return pd.DataFrame
        在原先的data后增加一列，每个元素对应的是该行对应的组数
    """
    dfg = fac_ret_cap.groupby(level=0).apply(lambda frame: cut_group(frame, col_name, num_group, ascending))
    return dfg


def get_factor_name(fac_ret):
    """
    在单因子分析中,有因子,因子对应的下期收益率, benchmark下期收益率, 市值, 分组列,本函数取得因子列的列名称.

    Args:
        fac_ret: 一个数据框

    Returns:
        str: 因子名称
    """
    keep_columns = ['cap', 'benchmark_returns', 'ret','group']
    factor_name = set(fac_ret.columns) - set(keep_columns)
    # assert len(factor_name) == 1, "there should be only one factor, got {}".format(factor_name)
    # factor_name = factor_name.pop()
    return list(factor_name)


def window(seq, n=2, longest=False):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    Args:
        longest: if True, get full length of seq,
        e.g. window([1,2,3,4], 3, longest=True) --->
        (1,2,3), (2,3,4), (3,4,None), (4,None,None)
    """
    if longest:
        it = itertools.chain(iter(seq), [None] * (n - 1))
    else:
        it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def __frame_to_rank(direction, standardized_factor):
    '''
    根据因子方向调整正负。
    '''
    standardized_factor_copy = standardized_factor.copy()
    # transfer True to -1, False to 1
    asc = {}
    for k, v in list(direction.items()):
        new_value = -1 if v else 1
        asc[k] = new_value
    for col in standardized_factor_copy.columns:
        if col not in ['cap']:
            standardized_factor_copy[col]=standardized_factor[col]*asc[col]
    return standardized_factor_copy

def form_fenceng(fenceng, fac_codes):
    '''
    根据ic计算各情景因子权重矩阵..
    '''
    df = pd.read_csv('/media/jessica/00001D050000386C/SUNJINGJING/dev/csf-factors/factors/stats_M004023.csv')
    lst = list(set(fac_codes)|set(['dt', 'Unnamed: 0']))
    df = df[df['Unnamed: 0'] != 'all']
    df = df[lst]
    df['Unnamed: 0']=df['Unnamed: 0'].map(lambda x:('{}_{}').format('cap',x))
    df = df.set_index(['dt', 'Unnamed: 0']).sort_index()
    ret=df.sort_index(level=0)
    ret = ret.groupby(level=0).apply(lambda x: x.reset_index(level=0, drop=True).T)
    # 计算IC_IR,并根据icir进行加权
    ic=ret.swaplevel(0, 1).sort_index()
    ir={}
    for ix in fac_codes:
        ir[ix]=ic.xs(ix).rolling(12,min_periods=1).mean()/ic.xs(ix).rolling(12,min_periods=1).std()
    ret=pd.Panel(ir).to_frame(filter_observations=False)
    ret = ret.groupby(level=0).apply(lambda x: x.reset_index(level=0, drop=True).T)

    ret = ret.abs().fillna(0)  # na用o填充，全部变为绝对值
    ret=ret.groupby(level=0).apply(lambda x: x / x.sum())  # 计算百分比
    ret=ret.fillna(0.0)
    return ret

def drop_middle(df):
    import numpy as np
    x=df.rank(method='average', na_option='keep')
    l=len(x)
    high=('{}_high').format('cap')
    low = ('{}_low').format('cap')
    if x.loc[:, 'cap'].notnull().sum()==0:
        x.loc[:, high]=np.nan
        x.loc[:, low]=np.nan
    else:
        x.loc[:, high] =pd.cut(x.loc[:, 'cap'], bins=9, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]).values
        x.loc[:, low] = pd.cut(x.loc[:, 'cap'], bins=9, labels=[-9, -8, -7, -6, -5, -4, -3, -2, -1]).values
    x=x.drop('cap',1)
    return x.abs()

def fenceng_score(df):
    '''
    对于不同个股，按照披露值进行打分，并得到个股情景权重
    '''
    if not isinstance(df,pd.DataFrame):
        df=pd.DataFrame(df)
    ret = df.groupby(level=0).rank(method='average', na_option='keep')
    __drop_middle = lambda x: drop_middle(x)
    ret2 = ret.groupby(level=0).apply(__drop_middle).sort_index()
    ret2=ret2.fillna(0.0)
    df3=ret2.groupby(level=0).apply(lambda x:x.reset_index(level=0, drop=True).T)
    df3 = df3.sort_index().groupby(level=0).apply(lambda x: x / x.sum())  # 计算百分比
    return df3

'''
neutral_util
'''
def add_industry(df,ind):
    '''
    输入fac_ret_cap,加上n列行业数据
    返回new_fac_ret_cap
    '''
    all_stocks=df.index.get_level_values(1).unique().tolist()
    df_ind=get_stock_industry(all_stocks,ind)
    df_ind.code=[x+'_SH_EQ' if x[0] in ['6'] else x+'_SZ_EQ' for x in df_ind.code]
    df_ind=df_ind.set_index('code')
    dict_ind={inx:df_ind.loc[inx,'level1_name'] for inx in df_ind.index}
    ind_sery=df.index.get_level_values(1).to_series().map(dict_ind)
    ind_dummies=pd.get_dummies(ind_sery)
    for col in ind_dummies.columns:
        df[col]=ind_dummies[col].values
    print('industry has been added')
    return df

def __single_neutral(col,formula,df):
    print(col)
    if col not in ['ret','cap','benchmark_returns']:
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
    field_x=[x for x in df.columns if x[:1] not in ['M','r','b']]
    field_dic={x1:'x'+str(x2) for (x1,x2) in zip(field_x,list(range(1,(len(field_x)+1),1)))}

    formula='Y~'
    df.columns=[field_dic[col] if (col in field_x and col not in ['cap']) else col for col in df.columns]
    for i in range(len(field_dic)):
        if i==0:
            formula=formula+list(field_dic.values())[i]
        else:
            formula=formula+'+'+list(field_dic.values())[i]
    field_y=[col for col in df.columns if col[:1] in ['M','r','c','b']]
    result=Parallel(n_jobs=6)(delayed(__single_neutral)(col,formula,df) for col in field_y)
    df.update(pd.concat(result, axis=1))
    return df


