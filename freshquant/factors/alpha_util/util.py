# -*- coding: UTF-8 -*-
"""
功能性函数
"""
import os
import datetime
import pandas as pd
import pickle
import numpy as np
from collections import OrderedDict
import pymongo
from dateutil.parser import parse
from sqlalchemy import create_engine
from csf_utils import get_mongo_connection
import datetime
import math
import time
from collections import OrderedDict, MutableMapping
from functools import wraps

import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import create_engine
from alpha_util.alpha_config import DEFALUT_HOST,DEFALUT_HOST_COMPONETS
# 找出指定的交易日

import logging
import re

pwd = os.getcwd()
TRADE_CAL_PATH = os.path.join(pwd, 'trade_cal.csv')



logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def m_end_td(date_index):
    """
    生成每个月最后一个交易日
    """
    month_end_list = []
    ret = OrderedDict()
    for dt in date_index:
        temp_y = dt.year
        temp_m = dt.month
        temp_d = dt.day
        if temp_y in list(ret.keys()):
            if temp_m in list(ret[temp_y].keys()):
                if temp_d > ret[temp_y][temp_m]:
                    ret[temp_y][temp_m] = temp_d
            else:
                ret[temp_y][temp_m] = temp_d
        else:
            ret[temp_y] = OrderedDict()
            ret[temp_y][temp_m] = temp_d
    for y in list(ret.keys()):
        for m in list(ret[y].keys()):
            month_end_list.append(datetime.datetime(y, m, ret[y][m]))
    return month_end_list


def w_end_td(date_index):
    """
    生成每个周最后一个交易日
    """
    week_end_list = []
    n = len(date_index)
    for i in range(0, n - 1):
        w0 = datetime.datetime.weekday(date_index[i])
        dif = date_index[i + 1] - date_index[i]
        if w0 == 4:
            week_end_list.append(date_index[i])
        elif w0 == 3 and dif >= datetime.timedelta(4):
            week_end_list.append(date_index[i])
        elif w0 == 2 and dif >= datetime.timedelta(5):
            week_end_list.append(date_index[i])
        elif w0 == 1 and dif >= datetime.timedelta(6):
            week_end_list.append(date_index[i])
        elif w0 == 0 and dif >= datetime.timedelta(7):
            week_end_list.append(date_index[i])
        else:
            pass
        week_end_list.append(date_index[-1])
        return week_end_list


def q_end_td(date_index):
    """
    生成每个季度最后一个交易日
    """
    q_end_list = []
    temp_list = m_end_td(date_index)
    for dt in temp_list:
        if dt.month in [3, 6, 9, 12]:
            q_end_list.append(dt)
    return q_end_list


# 获取股票相关信息

def get_index_components(idx_code, date):
    """
    new
    获取指定时间SAM指数及常用指数成份股代码
    @输入:
    idx_code: str，指数代码
    date: 日期，'2015-08-31'
    @返回：
    components: list, 成分股代码列表
    """
    # date = date if date else str(datetime.datetime.today().date())
    # print date
    conn = set_mongo_cond(DEFALUT_HOST_COMPONETS)
    conn_sam = set_mongo_cond('122.144.134.95')
    sam_tb = conn_sam.ada.index_specimen_stock
    com_tb = conn.ada.index_members_a
    date = date.replace('-','')
    if idx_code[0] in list('0123456789'):
        alrec = com_tb.find(
            {"$or": [{"p_code": idx_code, "in_dt": {"$lte": date}, "out_dt": {"$gt": date}},
                     {"p_code": idx_code, "in_dt": {"$lte": date}, "out_dt": None},
                     ]
             },
            projection = {'_id':0,'s_code':1}
        )
        com =list(alrec)
        components = pd.DataFrame(com).s_code.tolist()
        components = [secu+'_SH_EQ' if secu.startswith('6') else secu + '_SZ_EQ' for secu in components]
        # components = list(set(components) - set(get_not_trade_stock(date)))  # 剔除停牌股\st股票
    else:
        date = date.replace('-', '')
        alrec = sam_tb.find(
            {"idxcd": idx_code, "st": {"$lte": date}, "et": {"$gte": date}}, {
                '_id': 0, "secus": 1, "st": 1})
        components = []
        rec_lst = list(alrec)
        if rec_lst:
            components = [i['secu'] for i in rec_lst[0]['secus']]
    return components


def get_not_trade_stock(date):
    '''
    jyzt!="N"   交易状态 N通常状态；
    zqjb !="N"   证券级别  N 表示正常状态
    tpbz == "F"  停牌标志   T-停牌
    ## engine = create_engine('mysql://pd_team:pd_team321@!@122.144.134.21/ada-fd')
    ## sql_trade = (""" SELECT * FROM `ada-fd`.hq_stock_trade where dt = '{}' """).format(date)
    '''
    engine = create_engine('mysql://pd_team:pd_team321@!@122.144.134.95/ada-fd')
    sql_trade = (""" SELECT * FROM `ada-fd`.hq_stock_trade where dt = '{}' """).format(date)
    trade = pd.read_sql_query(sql_trade, engine).set_index('tick')
    st = trade.query(' zqjb == "s" or zqjb == "*" ').index.tolist()
    sql_tp = (""" SELECT tick FROM `ada-fd`.hq_stock_tp where dt = '{}' """).format(date)
    tingpai = pd.read_sql_query(sql_tp, engine)['tick'].tolist()
    __stocks=list(set(st) | set(tingpai))
    _stocks = [x for x in __stocks if x.startswith('0') or x.startswith('3') or x.startswith('6')]
    stocks = [str(x) + ('_SH_EQ') if x.startswith('6') else str(x) + ('_SZ_EQ') for x in _stocks]
    return stocks


def get_stock_lst_date(codes):
    """
    查询股票上市日期, 用来过滤数据
    输入：codes，单个股票代码或股票代码列表
    返回：{'code': 'dt'} 字典格式
    """
    if not isinstance(codes, list):
        codes = [codes]
    _, _, tb_base_stock = set_mongo_cond(DEFALUT_HOST, 'ada', 'base_stock')
    alrec = tb_base_stock.find(
        {"code": {"$in": codes}}, {"_id": 0, "code": 1, "ls.dt": 1})
    dt_lst = list(alrec)
    ret = {}
    for i in dt_lst:
        ret[i['code']] = i['ls']['dt']
    return ret


def get_stock_cap(codes, start_date=None, end_date=None):
    """
    用于查询股票流通市值
    输入参数：
    codes: 单个股票代码或股票代码列表
    start_date: 开始日期
    end_date: 结束日期
    """
    if DEFALUT_HOST == '192.168.0.222':
        _, _, tb = set_mongo_cond(DEFALUT_HOST, db_name='metrics_test',
                                  tb_name='comm_idx_price')
    else:
        _, _, tb = set_mongo_cond(db_name='metrics', tb_name='comm_idx_price')

    if not isinstance(codes, list):
        codes = [codes]
    alrec = tb.find({}, {})
    pass


# 数据库操作相关


def get_close_ndays_back(stocks, date, days):
    """
    获取前n天的股价
    """
    if isinstance(stocks, list):
        stocks = [stocks]
    if len(stocks[0]) > 6:
        stocks = [i[0:6] for i in stocks]
    engine = ut.set_sql_con()
    sql = "select dt, tick, close from hq_price where dt <=%s and tick in %s " % (
        date, tuple(stocks))
    get_sql = pd.read_sql(engine, sql)
    get_sql['tick'] = get_sql['tick'].map(
        lambda x: x + '_SH_EQ' if x[0] == '6' else x + 'SZ_EQ')
    ar = get_sql.ix[:, ['dt', 'tick']].T.values
    mul_idx = pd.MultiIndex.from_arrays(ar)
    ts = pd.Series(index=mul_idx, data=get_sql.close.values)
    df_pr = ts.unstack()
    return df_pr


# 数据结构操作

def flatten(d, parent_key='', sep='.'):
    '''
    字典flatten
    '''
    items = []
    for k, v in list(d.items()):
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(list(flatten(v, new_key, sep=sep).items()))
        else:
            items.append((new_key, v))
    return dict(items)


def write_to_sql():
    '''
    往sql写数
    '''
    pass


def write_to_mongo(args, mongo_host, db_name, tb_name, mode='insert'):
    '''
    往mongo写数据
    '''
    _, _, tb = set_mongo_cond(mongo_host, db_name, tb_name)

    if mode == 'insert':
        tb.insert(args)


def set_mongo_cond(host='122.144.134.95', db_name=None, tb_name=None):
    """
    mongo连接设置
    host: ip_address
    db_name: database name
    tb_name: table name
    """
    conn = MongoClient(host, 27017)
    ret = [conn]
    if db_name:
        db = conn[db_name]
        ret.append(db)
        if tb_name:
            tb = db[tb_name]
            ret.append(tb)
    ret = tuple(ret)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def set_sql_con(sql_addr=None):
    """
    sql连接设置
    sql_addr: 'mysql://ada_user:ada_user@122.144.134.3/ada-fd'
    """
    sql_default = 'mysql://ada_user:ada_user@122.144.134.3/ada-fd'
    engine = create_engine(sql_addr if sql_addr else sql_default)
    return engine


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time()
        print(('{} cost {} seconds'.format(func.__name__, t1 - t0)))
        return ret

    return wrapper


def form_mongo_to_df(tb_name, pos, filters):
    """
    输入mongo表连接，查询字段名(全字段，如quant.f18)，过滤条件
    tb_conn: mongo表连接
    pos: 查找的字段名
    filters：过滤条件，字典格式
    返回DataFrame格式
    """
    if DEFALUT_HOST == '192.168.250.200':
        _, db_metrics = set_mongo_cond(host=DEFALUT_HOST,
                                       db_name='metrics')
    else:
        _, db_metrics = set_mongo_cond(db_name='metrics')

    tb = {
        'pr': db_metrics.comm_idx_price_his_a,
        'ltm': db_metrics.comm_idx_quant_his_a,
        'ytd': db_metrics.comm_idx_quant_ytd_his_a,
        'tech': db_metrics.comm_idx_tech_his_a
    }

    projection_dict = {}
    for k in pos:
        key = k.replace('.', '#')
        val = ''.join(["$", k])
        projection_dict[key] = val
    projection_direct = {
        'pr': dict(_id=False, dt=True, secu=True, **projection_dict),
        'ltm': dict(_id=False, y=True, secu=True, **projection_dict),
        'ytd': dict(_id=False, y=True, secu=True, **projection_dict),
        'tech': dict(_id=False, dt=True, secu=True, **projection_dict)
    }

    pipeline = [
        {"$match": filters},
        {"$project": projection_direct[tb_name]}
    ]

    all_rec = tb[tb_name].aggregate(pipeline)

    df = pd.DataFrame(list(all_rec))
    df.columns = [col.replace("#", ".") for col in df.columns]

    return df


class InfoCof(object):
    """
    信息系数，三种算法：
    普通信息系数：因子暴露值与收益率的相关系数
    排序信息系数：因子暴露值排序与收益率排序的信息系数
    风险调整信息系数：经均值方差确定权重后的信息系数
    """

    def __init__(self, df_fac=None, df_ret=None, cov=None, method='normal'):
        self.fac = df_fac
        self.ret = df_ret
        self.cov = cov

    @staticmethod
    def _normal_IC(df_fac, df_ret):
        """
        信息系数计算
        """
        ret = pearsonr(df_fac, df_ret)
        return list(ret)

    @staticmethod
    def _rank_IC(df_fac, df_ret):
        """
        排序信息系数
        """
        ret = spearmanr(df_fac, df_ret)
        return list(ret)

    @staticmethod
    def _risk_IC(df_fac, df_ret, cov):
        """
        风险调整信息系数
        cov协方差矩阵
        TODO: check error
        """
        n = len(df_fac)
        W = np.ones([n]) / n
        rf = 0.02
        R = df_ret.values
        target = lambda W: 1 / \
                           ((sum(W * R) - rf) / math.sqrt(
                               np.dot(np.dot(W, cov), W)))
        b = [(0., 1.) for i in range(n)]  # boundary condition
        c = (
            {'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # summation condition
        optimized = scipy.optimize.minimize(
            target, W, method='SLSQP', bounds=b, constraints=c)
        weights = optimized.x
        df_ret_w = df_ret * weights
        ret = pearsonr(df_fac, df_ret_w)
        return list(ret)

    def IC(self, df_fac=None, df_ret=None, cov=None, method='normal'):
        df_fac = df_fac if not df_fac.empty else self.fac
        df_ret = df_ret if not df_ret.empty else self.ret
        cov = cov if cov != None else self.cov
        try:
            df = pd.concat([df_fac, df_ret]).dropna()
        except Exception as e:
            logger.exception(e)
            raise
        if not df.empty:
            if method == 'normal':
                ret = self._normal_IC(df_fac, df_ret)
            elif method == 'rank':
                ret = self._rank_IC(df_fac, df_ret)
            elif method == 'risk':
                ret = self._risk_IC(df_fac, df_ret, cov)
        else:
            logger.critical(
                'critical error occured! df_fac:{}, df_ret{}'.format(
                    df_fac.empty, df_ret.empty))
            ret = [np.nan, np.nan]
        return ret



# ××××××××××找出指定的交易日(周末，月末，季末)××××××××××
class EndTradeDay(object):
    """
    找出给定日期序列固定周期的最后一个交易日，并生成csv文件
    """

    def __init__(self, date_index=None, freq=None):
        """
        """
        self.freq = freq
        self.date_index = date_index

    @staticmethod
    def _m_end_td(date_index):
        """生成每个月最后一个交易日"""
        month_end_list = []
        ret = OrderedDict()
        for dt in date_index:
            temp_y = dt.year
            temp_m = dt.month
            temp_d = dt.day
            if temp_y in list(ret.keys()):
                if temp_m in list(ret[temp_y].keys()):
                    if temp_d > ret[temp_y][temp_m]:
                        ret[temp_y][temp_m] = temp_d
                else:
                    ret[temp_y][temp_m] = temp_d
            else:
                ret[temp_y] = OrderedDict()
                ret[temp_y][temp_m] = temp_d
        for y in list(ret.keys()):
            for m in list(ret[y].keys()):
                month_end_list.append(datetime.datetime(y, m, ret[y][m]))
        return month_end_list

    @staticmethod
    def _w_end_td(date_index):
        """生成每个周最后一个交易日"""
        week_end_list = []
        n = len(date_index)
        # print n
        for i in range(0, n - 1):
            w0 = datetime.datetime.weekday(date_index[i])
            dif = date_index[i + 1] - date_index[i]
            if w0 == 4:
                week_end_list.append(date_index[i])
            elif w0 == 3 and dif >= datetime.timedelta(4):
                week_end_list.append(date_index[i])
            elif w0 == 2 and dif >= datetime.timedelta(5):
                week_end_list.append(date_index[i])
            elif w0 == 1 and dif >= datetime.timedelta(6):
                week_end_list.append(date_index[i])
            elif w0 == 0 and dif >= datetime.timedelta(7):
                week_end_list.append(date_index[i])
        week_end_list.append(date_index[-1])
        return week_end_list

    def _q_end_td(self, date_index):
        """生成每个季度最后一个交易日"""
        q_end_list = []
        temp_list = self._m_end_td(date_index)
        for dt in temp_list:
            if dt.month in [3, 6, 9, 12]:
                q_end_list.append(dt)
        return q_end_list

    def get_all_end_td(self):
        """生成周、月、季的最后一个交易日"""

        trade_cal = pd.read_csv(TRADE_CAL_PATH, header=None, index_col=0)
        date_index = [parse(i) for i in trade_cal.index.values]
        ret_w = self._w_end_td(date_index)
        ret_m = self._m_end_td(date_index)
        ret_q = self._q_end_td(date_index)
        ret = {'w': [str(i.date()) for i in ret_w],
               'm': [str(i.date()) for i in ret_m],
               'q': [str(i.date()) for i in ret_q],
               }
        with open('end_td.txt', 'wb') as f:
            pickle.dump(ret, f)



def get_stock_industry(codes, sam_level=1):
    """
    获取股票所属行业

    opt.fin_industry_product
    opt.fin_industry_product_his_a
    取得 industry.code


    opt.dict_product_rs
    通过 code 和 
    取得 ind.cd [csf]


    ada.dict_industry
    通过 code
    取得 ancestors[0] (一级)

    """
    host = '122.144.134.95'
    if not isinstance(codes, list):
        codes = list(codes)
    _, opt, _ = get_mongo_connection(host=host, db_name='opt')
    _, ada, _ = get_mongo_connection(host=host, db_name='ada')
    fin_p = opt['fin_industry_product']
    fin_p_his = opt['fin_industry_product_his_a']
    dic_p = opt['dict_product_rs']
    dic_ind = ada['dict_industry']

    all_rec_1 = fin_p.find(
        {"secu": {"$in": codes},
         "industry.code": {"$ne": "0x0x"}},
        {"secu": 1, "industry.code": 1, "_id": 0})

    stocks = []
    ind_code = []
    for i in list(all_rec_1):
        if 'industry' in i:
            stocks.append(i['secu'])
            ind_code.append(i['industry']['code'])
    # print len(codes)
    # stocks = sorted(list(set(stocks)))
    # print len(stocks)

    if len(set(stocks)) < len(codes):
        sub_code = [i for i in codes if i not in stocks]
        all_rec_2 = fin_p_his.find(
            {"secu": {"$in": sub_code}, "industry.code": {"$ne": "0x0x"}},
            {"secu": 1, "industry.code": 1, "_id": 0})
        for i2 in list(all_rec_2):
            if 'industry' in i2:
                if i2['secu'] not in stocks:
                    stocks.append(i2['secu'])
                    ind_code.append(i2['industry']['code'])

    df = pd.DataFrame({'code': stocks, "ind": ind_code}).set_index('code')

    ind_code_set = list(set(ind_code))

    all_rec_3 = dic_p.find(
        {"code": {"$in": ind_code_set}}, {"ind": 1, "code": 1, "_id": 0})

    ind_code1 = []
    csf = []
    for i3 in list(all_rec_3):
        if i3['code'] not in ind_code1:
            ind_code1.append(i3['code'])
            for j in i3['ind']:
                cd = 'none'
                if j['t'] == 'csf':
                    cd = j['cd']
                    csf.append(cd)

    csf_dic = dict(list(zip(ind_code1, csf)))

    df = df.applymap(lambda x: csf_dic[x] if x in csf_dic else np.nan)

    all_rec_4 = dic_ind.find(
        {"code": {"$in": csf}}, {"ancestors": 1, "code": 1, "_id": 0})

    csf1 = []
    csf0 = []
    for i4 in list(all_rec_4):
        if 'ancestors' in i4:
            csf1.append(i4['code'])
            csf0.append(i4['ancestors'][sam_level - 1])

    csf_dic1 = dict(list(zip(csf1, csf0)))
    # df = df.to_frame()
    df = df.applymap(lambda x: csf_dic1[x] if x in csf_dic1 else np.nan)
    all_rec_5 = dic_ind.find(
        {"code": {"$in": csf0}}, {"code": 1, "zhsname": 1, "_id": 0})

    sam0 = []
    name0 = []
    for i5 in list(all_rec_5):
        if 'zhsname' in i5:
            sam0.append(i5['code'])
            name0.append(i5['zhsname'])

    name_dic = dict(list(zip(sam0, name0)))
    df = df.applymap(lambda x: name_dic[x] if x in name_dic else np.nan)

    return df


def _mongo_to_df(db_name=None, tb_name=None, filters=None, fields=None):
    _, db = get_mongo_connection(db_name=db_name)

    tb = db[tb_name]
    filters_keys = list(filters.keys())
    for k in filters_keys:
        tb.create_index([(k, pymongo.ASCENDING)])
    tb.create_index(list(zip(filters_keys, [pymongo.ASCENDING] * len(filters_keys))))
    all_rec = tb.find(filters, fields)
    gen = (flatten(rec) for rec in all_rec)
    return pd.DataFrame(gen)


# ×××××××××收益率评价指标××××××××××
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


def cal_mad(df):
    """
    绝对均值离差
    df: DataFrame

    """
    df_submean = df - df.median()
    df_abs = df_submean.abs()
    ret = df_abs.median()
    return ret


# @profile
def handle_extreme(df, num=3, method='mad'):
    """
    极值处理，method可选mad, std
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    mu = df.median()   # 序列的中位数
    ind = cal_mad(df) if method == 'mad' else df.std()   # ind 为|fi-mu|的中位数
    ret = df
    try:
        ret = df.clip(lower=mu - num * ind, upper=mu + num * ind, axis=1)
    except Exception as e:
        lst = []
        for col in df.columns:
            if df.loc[:, col].isnull().all():
                lst.append(df.loc[:, col])
            else:
                mu = df.loc[:, col].mean()
                ind = cal_mad(df.loc[:, col]) if method == 'mad' else df.loc[:,
                                                                             col].std()
                lst.append(df.loc[:, col].clip(lower=mu - num * ind,
                                               upper=mu + num * ind))
        ret = pd.concat(lst, axis=1)
    return ret


# ××××××××××数据处理××××××××××××××××


class scale(object):
    def __init__(self, df=None, s_cap=None, method='normal'):
        self.data = df
        self.cap = s_cap
        self.method = method

    @staticmethod
    def scale_normal(df_fac):
        """
        标准化处理, 普通标准化
        """
        ret = (df_fac - df_fac.mean()) / df_fac.std()
        # ret = df_fac.apply(lambda x: (x-np.nanmean(x))/np.nanstd(x))
        return ret

    @staticmethod
    def scale_cap(df_fac, s_cap):
        """
        标准化处理， 市值标准化
        df_fac: 因子横截面数据
        s_cap: 流通市值数据
        """
        if isinstance(s_cap, pd.Series):
            cap = s_cap
        elif isinstance(s_cap, pd.DataFrame):
            col = s_cap.columns[0]
            cap = pd.Series(data=s_cap[col].values, index=s_cap.index)
        fun = lambda x: (x - np.sum(x * cap) / cap.sum()) / x.std()
        ret = df_fac.apply(fun)
        return ret

    @staticmethod
    def scale_industry(df_fac):
        """
        标准化处理， 行业标准化
        df_fac: 因子横截面数据
        st_mean: 因子行业均值
        st_std: 因子行业标准差
        """
        st_mean, st_std = industry_mean_std()
        ret = (df_fac - st_mean) / st_std
        return ret

    # @profile
    def scale(self, data, method='normal', s_cap=None):
        """
        对以上各方法的统一接口
        """
        data = data if not self.data else self.data
        s_cap = s_cap if not self.cap else self.cap
        method = method if not self.method else self.method
        if method == 'normal':
            ret = self.scale_normal(data)
        elif method == 'cap':
            ret = self.scale_cap(data, s_cap)
        elif method == 'industry':
            ret = self.scale_industry(data)
        return ret

    def industry_mean_std(self, stocks, factors, rep_term):
        """
        根据SAM4级行业分类
        """

        pass

    def sector_mean_std():
        """
        风格分类均值与标准差
        """

        pass

    def find_collinearity_columns(correlation):
        """
        本函数找出多重共线性的列。
        基本思路：
        0.首先看correlation是否满秩。如果不满秩，说明存在多重共线性。
        1. 找到correlation里面绝对值最大的row_idx,和col_idx,假设为A和C.
        2.计算A列/C列与其他列的相关系数绝对值的均值，如果A列与其他列相关系数更大，则剔除A列，反之亦然。记录下来A列名
        重复以上步骤
        @params: correlation, 相关系数矩阵.dataframe
        @returns: list of column names.
        """
        bad_columns = []
        while True:
            rank = np.linalg.matrix_rank(correlation.values)
            if rank == correlation.shape[0]:
                break

            correlation_copy = correlation.copy()
            correlation = correlation.abs()
            correlation.values[np.triu_indices_from(correlation.values,
                                                    0)] = 0.0  # 把上三角（包括对角线部分）设置为0.
            col_idx, row_idx = correlation.unstack().argmax()  # (col_idx, row_idx)
            if correlation_copy.ix[row_idx, :].mean() > correlation_copy.ix[:,
                                                                            col_idx].mean():
                bad_column = row_idx
            else:
                bad_column = col_idx
            bad_columns.append(bad_column)
            # 把该列名称从相关系数矩阵的行/列里去掉
            correlation_copy.drop(bad_column, axis=0, inplace=True)
            correlation_copy.drop(bad_column, axis=1, inplace=True)
            correlation = correlation_copy
        return bad_columns


def get_trade_calendar():
    file_path = os.path.abspath(__file__)
    dir_name = os.path.split(file_path)[0]
    csv_file = os.path.join(dir_name, 'trade_cal.csv')
    trade_cal = pd.read_csv(csv_file, names=['date_time', 'total_day'],
                            index_col=[0], parse_dates=True)
    return trade_cal


class GetSql(object):
    def __init__(self,
                 host="122.144.134.95",
                 db_name='ada-fd',
                 username='pd_team',
                 password='pd_team321@!',
                 isIndex=False):
        self.sql_addr = 'mysql+mysqlconnector://%s:%s@%s/%s' % (
            username, password, host, db_name)
        self.engine = self.set_sql_con()
        self.isIndex = isIndex

    def set_isIndex(self, isIndex=False):
        self.isIndex = isIndex

    def set_sql_con(self, sql_addr=None):
        """
        sql连接设置
        sql_addr: 'mysql://ada_user:ada_user@122.144.134.3/ada-fd'
        """
        engine = create_engine(sql_addr if sql_addr else self.sql_addr)
        return engine

    @staticmethod
    def _check_codes(stocks):
        if not isinstance(stocks, list) and not isinstance(stocks, np.ndarray):
            stocks = [stocks]
        if len(stocks[0]) > 6:
            stocks = [i[0:6] for i in stocks]
        stocks = [('000000' + str(int(i)))[-6:] for i in stocks]
        return stocks

    def _get_from_sql(self, sql):
        engine = self.engine
        if self.isIndex:
            sql = sql.replace('hq_price', 'hq_index')
        get_sql = pd.read_sql(sql, engine)
        if not self.isIndex:
            get_sql['tick'] = get_sql['tick'].map(
                lambda x: x + '_SH_EQ' if x[0] == '6' else x + '_SZ_EQ')
        get_sql = get_sql.sort_values(by='dt')
        ar = get_sql.ix[:, ['dt', 'tick']].T.values
        mul_idx = pd.MultiIndex.from_arrays(ar)
        ts = pd.Series(index=mul_idx, data=get_sql.close.values)
        df_pr = ts.unstack()
        return df_pr


    def jessica_get_from_sql(self, sql):
        engine = self.engine
        if self.isIndex:
            sql = sql.replace('hq_price', 'hq_index')
        get_sql = pd.read_sql(sql, engine)
        if not self.isIndex:
            get_sql['tick'] = get_sql['tick'].map(
                lambda x: x + '_SH_EQ' if x[0] == '6' else x + '_SZ_EQ')
        get_sql = get_sql.sort_values(by='dt')
        ar = get_sql.ix[:, ['dt', 'tick']].T.values
        mul_idx = pd.MultiIndex.from_arrays(ar)
        ts = pd.Series(index=mul_idx, data=get_sql.inc.values)
        df_pr = ts.unstack()
        return (df_pr.iloc[1:,]/100+1).product()-1

    def get_n_days_back(self, stocks, end_date, days):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price
                     WHERE dt <='%s' AND tick = '%s'
                     ORDER BY dt DESC LIMIT %d""" % (
                end_date, stocks[0], days)
        else:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt <='%s' AND tick IN %s
                     ORDER BY dt DESC LIMIT %d""" % (
                end_date, tuple(stocks), days)
        return self._get_from_sql(sql)

    def get_between_dates(self, stocks, start_date, end_date):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price
                     WHERE dt between '%s' AND '%s' AND tick = %s
                     ORDER BY dt ASC""" % (
                start_date, end_date, stocks[0])
        else:
            sql = """SELECT dt, tick, close
                     FROM hq_price
                     WHERE dt between '%s' AND '%s' AND tick in %s
                     ORDER BY dt ASC""" % (start_date, end_date, tuple(stocks))
        return self._get_from_sql(sql)


    def jessica_get_between_dates(self, stocks, start_date, end_date):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, inc
                     FROM hq_price
                     WHERE dt between '%s' AND '%s' AND tick = %s
                     ORDER BY dt ASC""" % (
                start_date, end_date, stocks[0])
        else:
            sql = """SELECT dt, tick, inc
                     FROM hq_price
                     WHERE dt between '%s' AND '%s' AND tick in %s
                     ORDER BY dt ASC""" % (start_date, end_date, tuple(stocks))
        return self.jessica_get_from_sql(sql)

    def get_in_spec_dates(self, stocks, date_lst):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price
                     WHERE dt IN %s AND tick = %s
                     ORDER BY dt ASC""" % (
                tuple(date_lst), stocks[0])

        elif len(date_lst) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price
                     WHERE dt = %s AND tick IN %s """ % (
                date_lst[0], tuple(stocks))

        else:
            sql = """SELECT dt, tick, close
                     FROM hq_price
                     WHERE dt IN %s AND tick IN %s
                     ORDER BY dt ASC""" % (
                tuple(date_lst), tuple(stocks))
        return self._get_from_sql(sql)


class FactorData(object):
    def __init__(self, name, return_analysis, IC_analysis, turnover_analysis,
                 code_analysis):
        self.name = name
        self.return_analysis = return_analysis
        self.IC_analysis = IC_analysis
        self.turnover_analysis = turnover_analysis
        self.code_analysis = code_analysis


class ReturnAnalysis(object):
    def __init__(self, name=None, return_statistics=None,
                 group_return_mean=None, group_return_cumulative=None):
        self.name = name
        self.return_statistics = return_statistics
        self.group_return_mean = group_return_mean
        self.group_return_cumulative = group_return_cumulative


class ICAnalysis(object):
    def __init__(self, name=None, IC_series=None, IC_statistics=None,
                 groupIC=None, IC_decay=None):
        self.name = name
        self.IC_series = IC_series
        self.IC_statistics = IC_statistics
        self.groupIC = groupIC
        self.IC_decay = IC_decay


class TurnOverAnalysis(object):
    def __init__(self, name=None, buy_signal=None, auto_correlation=None,
                 turnover=None):
        self.name = name
        self.buy_signal = buy_signal
        self.auto_correlation = auto_correlation
        self.turnover = turnover


class CodeAnalysis(object):
    def __init__(self, name=None, industry_analysis=None, cap_analysis=None,stock_list=None):
        self.name = name
        self.industry_analysis = industry_analysis
        self.cap_analysis = cap_analysis
        self.stock_list=stock_list


class IndustryAnalysis(object):
    def __init__(self, name=None, gp_mean_per=None, gp_industry_percent=None):
        self.name = name
        self.gp_mean_per = gp_mean_per
        self.gp_industry_percent = gp_industry_percent


def form_dt_index(start_date, end_date, freq):
    """
    获取每个调仓期的具体日期：
    self.freq=='m' 则返回每月最后一个交易日日期；
    self.freq=='w' 则返回每周最后一个交易日日期；
    self.freq=='q' 则返回每季最后一个交易日日期；
    返回列表['2014-01-31']
    """
    start = start_date[0:8] + '01'
    end = end_date[0:8] + '01'
    trade_cal = get_trade_calendar()
    current_calendar = trade_cal.ix[start_date:end_date, :]
    funcs = {
        'w': lambda x: x.week,
        'm': lambda x: x.month,
        'q': lambda x: x.quarter
    }
    ret = [str(data.index[-1].date()) for (year, func), data in
           current_calendar.groupby([current_calendar.index.year,funcs[freq](current_calendar.index)])]
    return ret


def generate_report_dates(dt_index):
    """
    生成涉及到的报告期,逻辑：
    1,2,3月底用上一年Q3的数据
    4,5,6,7月底用今年Q1的数据
    8,9月底用今年Q2的数据
    10，11，12用今年Q3的数据
    """
    date_lst = dt_index[0:-1]
    q_list = []
    for dt in date_lst:
        dt_sl = dt.split('-')
        yr = int(dt_sl[0])
        mt = int(dt_sl[1])
        if mt in [1, 2, 3]:
            q_list.append("".join([str(yr - 1), '-09-30']))
        elif mt in [4, 5, 6, 7]:
            q_list.append("".join([str(yr), '-03-31']))
        elif mt in [8, 9]:
            q_list.append("".join([str(yr), '-06-30']))
        elif mt in [10, 11, 12]:
            q_list.append("".join([str(yr), '-09-30']))
    ret = pd.Series(data=q_list, index=dt_index[0:-1])
    return ret


if __name__ == '__main__':
    code = ['600000']
    get_stock_industry(code)