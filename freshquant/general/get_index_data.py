# -*- coding: utf-8 -*-
__author__ = 'jessica'

import pandas as pd
from general.mongo_data import set_mongo_cond
from general.config import DEFALUT_HOST,DEFALUT_HOST_COMPONETS
from sqlalchemy import create_engine
from general.trade_calendar import form_dt_index


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
    __stocks = list(set(st) | set(tingpai))
    _stocks = [x for x in __stocks if x.startswith('0') or x.startswith('3') or x.startswith('6')]
    stocks = [str(x) + ('_SH_EQ') if x.startswith('6') else str(x) + ('_SZ_EQ') for x in _stocks]
    return stocks


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
    conn_sam = set_mongo_cond('192.168.250.200')
    sam_tb = conn_sam.ada.index_specimen_stock
    com_tb = conn.ada.index_members_a
    date = date.replace('-','')

    # 881001
    if idx_code[0] in ['8']:
        sql_address = 'mysql+mysqlconnector://pd_team:pd_team123@!@192.168.250.200/ada-fd'
        engine = create_engine(sql_address)
        sql = """SELECT * FROM hq_price_before WHERE dt = '%s' and tick < '700000' """ % (date)
        stocks = pd.read_sql(sql, engine).tick.tolist()
        components = [secu+'_SZ_EQ' if secu[0] in ['0','3'] else secu+'_SH_EQ' for secu in stocks]
    elif idx_code[0] in list('012345679'):
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

def get_dynamic_components(idx_code,start,end,freq='m'):
    dt_index=form_dt_index(start,end,freq)
    rets = []
    for dt in dt_index[:-1]:
        ret = get_index_components(idx_code, dt)
        ret = list(zip([dt] * len(ret), ret))
        rets.extend(ret)
    dynamic_stocks=pd.MultiIndex.from_tuples(rets, names=['dt', 'secu'])
    # all_stocks = list(set(dynamic_stocks.levels[1]))
    return dynamic_stocks

def get_index_market_data(code,start,end):
    """
    code:'000300'
    mysql+mysqlconnector://ada_user:ada_user@122.144.134.3/ada-fd
    """
    __stocks = get_index_components(code,date=end)
    stocks =[secu[:6] for secu in __stocks]
    sql_address = 'mysql+mysqlconnector://pd_team:pd_team321@!@122.144.134.21/ada-fd'
    engine=create_engine(sql_address)
    sql = """SELECT * FROM hq_price_before WHERE dt >= '%s' and dt <='%s' AND tick IN %s ORDER BY dt DESC""" % (start,end, tuple(stocks))
    df=pd.read_sql(sql, engine)
    return df

if __name__ == '__main__':
    print(get_index_components('000300', '2016-06-30', del_three=False))
    print('Done!')
