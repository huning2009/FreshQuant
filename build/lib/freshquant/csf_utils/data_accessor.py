# coding: utf8
import os
import re
from functools import reduce

import logbook
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from .csf_utils import get_mongo_connection
from .csf_utils import merge_dicts

pd.set_option('display.max_rows', 10)
from .csf_utils import batch
from .csf_utils import basestring
import csf

logger = logbook.Logger(__name__)
csf.config.set_token("be49d98a6311b992ce9c5269156da9f7",
                     "1tNY78z2bPhWcHbcIKLxKpKcTQo=")

engine = create_engine(
    'mysql+mysqlconnector://mysqldb:chinascope1234@mysqldb-01.csilcsoh66yg.rds.cn-north-1.amazonaws.com.cn/ced?charset=utf8')


def get_index_sam_map2():
    """
    取得指数-sam 对应关系, 这里的指数指的是能从metrics.csf_index_daily取得数据的指数．
    Returns(DataFrame): 包含 index sam的数据框
    """
    client, _, _ = get_mongo_connection()

    index_sam_map = get_index_sam_map()

    # get unique index code from csf_index_daily
    indexcd_cursor = client['metrics'].csf_index_daily.find({}, projection={'idxcd': 1}).distinct('idxcd')
    csf_index = pd.DataFrame(list(indexcd_cursor), columns=['index_code'])

    df = pd.merge(index_sam_map, csf_index, on=['index_code'], how='inner')

    return df


def get_index_components(idx_code, date):
    """
    获取指定时间SAM指数及常用指数成份股代码
    @输入:
    idx_code: str，指数代码
    date: 日期，'2015-08-31'
    @返回：
    components: list, 成分股代码列表
    """
    # date = date if date else str(datetime.datetime.today().date())
    # print date
    conn, _, _ = get_mongo_connection()
    sam_tb = conn.ada.index_specimen_stock
    com_tb = conn.metrics.idx_stock_history

    if idx_code[0] in list('0123456789'):
        alrec = com_tb.find(
            {"idxcd": idx_code, "st": {"$lte": date}, "et": {"$gte": date}}, {
                '_id': 0, "secus": 1, "st": 1})
        components = list(alrec)[0]['secus']
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


def get_csf_index_price(index_code, start_date=None, end_date=None):
    """
    Get csf index price data from mongodb
    :param index_code, csf index code or code lst
    :param start_date, start date of the price you wanna get
    :param end_date, end date of the price you wanna get
    :return: index_price
    """
    if not isinstance(index_code, list):
        index_code = [index_code]

    filters = {}
    filters['idxcd'] = {'$in': index_code}

    trdt = {}

    start_date_ = None
    end_date_ = None
    try:
        start_date_ = pd.Timestamp(start_date).strftime('%Y%m%d')
    except (AttributeError, ValueError) as e:
        print(e)
    try:
        end_date_ = pd.Timestamp(end_date).strftime('%Y%m%d')
    except (AttributeError, ValueError) as e:
        print(e)

    if start_date_:
        trdt['$gte'] = start_date_
    if end_date:
        trdt['$lte'] = end_date_

    if trdt:
        filters['trdt'] = trdt

    _, _, tb = get_mongo_connection(db_name='metrics', tb_name='csf_index_daily')

    projection = {'index_code': '$idxcd', 'close': '$cv', 'date': '$trdt', '_id': 0}

    price_rec = tb.aggregate(pipeline=[
        {"$match": filters},
        {"$project": projection}
    ])

    df_origin = pd.DataFrame(list(price_rec))

    if df_origin.empty:
        print('No close data for %s' % index_code)
        return df_origin
    else:
        if len(index_code) == 1:
            df_ret = df_origin.ix[:, ['date', 'close']].set_index('date')
            df_ret.columns = index_code
        else:
            df_origin = df_origin.drop_duplicates()
            df_ret = df_origin.pivot('date', 'index_code', 'close')

        df_ret.index = pd.to_datetime(df_ret.index)
        dif_codes = sorted(set(index_code) - set(df_ret.columns))
        if dif_codes:
            print('No close data for %s' % dif_codes)
        return df_ret


def get_trade_calendar():
    file_path = os.path.abspath(__file__)
    dir_name = os.path.split(file_path)[0]
    csv_file = os.path.join(dir_name, 'trade_cal.csv')
    trade_cal = pd.read_csv(csv_file, names=['date_time', 'total_day'],
                            index_col=[0], parse_dates=True)
    return trade_cal


# --------------------------------------------------------------------------------------------
# get macro data
# --------------------------------------------------------------------------------------------

def get_index_sam_map():
    """
     得到所有数库行业指数和数库产品对应关系
    :return: DataFrame, columns: index_name, zh_name, sam_code
    """
    _, db, tb = get_mongo_connection(host='122.144.134.95', db_name='ada',
                                     tb_name='dict_index')
    filters = {'serie': 'SAM'}
    #     filters = {'idxcd': index_name}
    projections = {"_id": 0, "samcode": "$samcode", 'index_code': "$idxcd",
                   'index_zh_name': "$idxnm.szh"}
    pipeline = [{"$match": filters}, {"$project": projections}]
    all_rec = tb.aggregate(pipeline)
    rec_lst = list(all_rec)
    df = pd.DataFrame(rec_lst)
    return df


def get_sam_acmr_map(samcode):
    """
    samcode <---> acmr code
    :param samcode: samcode
    :return: DataFrame
    """
    tem = csf.get_product_macro_map(industry_code=samcode)
    return tem


def get_indicator_id_via_samcode(samcode, freq='M'):
    """
    samcode ----> indicator_id
    :param freq: D 日度;W 周次;M 月度;Q 季度;S 半年度;A 年度;T 十天. 'ALL'
    :param samcode: sam 代码
    :return: DataFrame
    """
    frames = []
    for c in np.unique(samcode):
        sql = """SELECT  id as indicator_id, cname_tpl, type, timesort as freq
                FROM ced_indicator_new where prd LIKE '%{}%'""".format(c)
        df = pd.read_sql_query(sql, engine)
        frames.append(df)
    ret = pd.concat(frames, ignore_index=True)
    ret.loc[:, 'indicator_id'] = ret.indicator_id.astype(int)
    if freq.upper() in {'M', 'D', 'W', 'Q', 'S', 'A', 'T'}:
        ret = ret.query('freq == "{}"'.format(freq))
    return ret.drop_duplicates().reset_index(drop=True)


def get_macrodata_via_indicator_id(indicator_id):
    """
    data code ----> data
    :param indicator_id:
    :return: DataFrame
    """
    tup = tuple(indicator_id)
    # noinspection SqlNoDataSourceInspection
    sql = """select indicator_id, ayearmon, data
    From ced_data_new
    where indicator_id in {}
    """.format(tup)
    df = pd.read_sql_query(sql, engine)
    return df


def get_macro_data(keyword):
    """
    返回2007-01-01之后的宏观经济数据
    Args:
        keyword: 关键词，如 u"房地产"
    Returns:
        DataFrame: 宏观经济数据, 如果没有同期数据或者同比/环比, 则利用累计值来计算．
    """

    # step 1. get sam via keyword.
    index_sam_map = get_index_sam_map_via_keyword(keyword)
    # step 2. get indicator_id
    ids = get_indicator_id_via_samcode(index_sam_map.samcode)

    macro_data = get_macrodata_via_indicator_id(ids.indicator_id.unique())
    #     macro_data.loc[:, 'keyword'] = keyword
    macro_data.loc[:, 'datacode_timetype'] = macro_data.datacode.add('_').add(
        macro_data.timetype)
    data = macro_data.set_index(['ayearmon', 'datacode_timetype'])[
        'data'].unstack().sort_index()
    data.index = data.index.str.replace("(\d{4})(\d{2})", "\\1-\\2")
    start, end = data.index.min(), data.index.max()

    data.index = pd.DatetimeIndex(data.index)

    full_index = pd.DatetimeIndex(start=start, end=end, freq='MS', name='date')

    data = data.reindex(full_index)

    for datacode in macro_data.datacode:
        key = datacode
        dangqi = key + "_" + 'b0101'
        accumulative = key + '_' + 'b0301'
        huanbi = key + '_' + 'b0102'
        tongbi = key + '_' + 'b0103'

        if dangqi not in data.columns:
            try:
                data.loc[:, dangqi] = data.loc[:, accumulative].diff()
            except KeyError:
                print(u'code {} 没有当期值，　也没有累计值'.format(key))

        if huanbi not in data.columns:
            try:
                data.loc[:, huanbi] = data.loc[:, dangqi].pct_change() * 100
            except KeyError:
                print(u'无法计算 code {} 的环比，　没有当期值, 且无法从其他数据获取当期值'.format(key))

        if tongbi not in data.columns:
            try:
                data.loc[:, huanbi] = (data.loc[:, dangqi] - data.loc[:,
                                                             dangqi].shift(
                    12)) / data.loc[:, dangqi].shift(
                    12) * 100
            except KeyError:
                print(u'无法计算 code {} 的同比比，　没有当期值, 且无法从其他数据获取当期值'.format(key))
    data[~np.isfinite(data)] = np.nan
    data.to_hdf(keyword + u'_' + u'macro.hdf', key='a')
    return data


def get_macro_data_dict_form(data):
    """
    预先将DataFrame 转成dict, drop NA
    :param data:
    :return:
    """
    dic = {}
    for col in data.loc[:, data.columns.sort_values()]:
        dic[col] = data.loc['2007-01-01':, col].dropna()
    return dic


# --------------------------------------------------------------------------------------------
# get index data
# --------------------------------------------------------------------------------------------
def get_index_factors(keyword):
    # step1. 根据keyword, 取得相关指数
    index_sam_map = get_index_sam_map_via_keyword(keyword)
    index_name_list = index_sam_map.index_name.unique().tolist()

    # _, db, tb_quarterly = get_mongo_connection(db_name='metrics', tb_name='comm_idx_factor_fin_his_a')
    # quarterly_data = tb_quarterly.find({'idxcd':{"$in": index_name_list}}, {'_id':0, 'upt':0})

    # get_monthly_data
    _, db, tb_daily = get_mongo_connection(db_name='metrics', tb_name='comm_idx_factor_his_a')
    cursor = tb_daily.find({'idxcd': {"$in": index_name_list}}, {'_id': 0, 'upt': 0, 'crt': 0})
    daily_data = pd.DataFrame(list(cursor))
    return daily_data


def get_index_sam_map_via_keyword(keyword):
    index_sam_map = get_index_sam_map2()
    index_sam_map = index_sam_map[index_sam_map.index_zh_name.str.contains(keyword)]
    index_sam_map = index_sam_map[index_sam_map.index_code.str.match('[^EW_|FD_]')]
    return index_sam_map


# --------------------------------------------------------------------------------------------
# get stock data
# --------------------------------------------------------------------------------------------

def get_stock_factors(keyword):
    """

    :param keyword:
    :return: dict, key is stock+#+factor
    """
    # step1. 根据keyword, 取得相关指数
    # index_sam_map = get_index_sam_map_via_keyword(keyword)
    # index_name_list = index_sam_map.index_name.unique().tolist()

    # step2. 根据keyword,　查找base_org表, 返回org.id

    org_id = get_org_id_via_keyword(keyword)

    stock = get_stock_code_via_orgid(org_id)

    factor_lists = csf.get_stock_factor_list()
    factor_lists = sorted(factor_lists.query("level == 2").code.tolist())
    logger.debug('starting getting stock data')
    # stock_factors_data = csf.get_stock_factor_by_codes(stock.tick.tolist()[:5], factor_lists[:5])
    stock_factors_data = [csf.get_stock_factor_by_codes(stock.tick.tolist(), list(subgroup)) for subgroup in
                          batch(factor_lists, 5)]
    logger.debug('finishing getting stock data')

    stock_factors_data = pd.concat(stock_factors_data, ignore_index=True)
    stock_factors_data.loc[:, 'date'] = pd.DatetimeIndex(stock_factors_data.date.str[:-2] + '01')
    stock_factors_data = stock_factors_data.set_index(['code', 'date', 'cd']).unstack()
    stock_factors_data.columns = [col[1] for col in stock_factors_data.columns]
    stock_factors_data.to_hdf(keyword + u'_' + u'stock_factors.hdf', key='a')
    return stock_factors_data


def get_stock_factors_from_mongo(keyword):
    """
    从mongo中取得因子数据．
    :param keyword:
    :return: dict, key is stock+#+factor
    Notes:
        所有数据都是取得季末最后一天．
        对字段为dt的数据，日期标签是最后交易日．
        对字段为y的数据，日期标签为季末最后一天(不一定是交易日)．
        为了统一，　将所有的日期改成了01日．
        例如，　2010-03-01, 表示的是2010第一季度的数据，01没有特别的意义．
    """
    # step1. 根据keyword,　查找base_org表, 返回org.id
    org_id = get_org_id_via_keyword(keyword)

    # step2. 根据org_id, 查找stock_code(*_SH_EQ),tick(000001)
    stock = get_stock_code_via_orgid(org_id)

    # step3. 取得因子code与mongo数据库中字段的对应关系.
    code_field_map = pd.read_excel('/home/phil/dev/alpha/alpha/alpha_algo/quant_dict_factors_all.xlsx',
                                   sheetname='factors')
    code_field_map = code_field_map.query('stat == 1').dropna()
    code_field_map.loc[:, 'tb'] = code_field_map.tb.str.replace('metrics.', '')
    code_field_map = {k: v for k, v in code_field_map.groupby('tb')}

    _, db, _ = get_mongo_connection(db_name='metrics')

    tb1 = u'comm_idx_price_his_a'
    tb2 = u'comm_idx_quant_his_a'
    tb3 = u'comm_idx_tech_his_a'
    tb4 = u'comm_idx_quant_ytd_his_a'

    # tb1, daily
    # get every quarter end
    trade_cal = get_trade_calendar()['2007-01-01':]
    quarter_end = trade_cal.groupby([trade_cal.index.year, trade_cal.index.quarter]).tail(1)
    quarter_end = [str(dt.date()) for dt in quarter_end.index]

    data1 = db[tb1].aggregate(pipeline=[
        {"$match": {'secu': {"$in": stock.code.unique().tolist()}, 'dt': {'$in': quarter_end}}},
        {'$project': merge_dicts(dict(zip(code_field_map[tb1].code, '$' + code_field_map[tb1].fd)),
                                 {'_id': 0, 'date': {'$concat': [{'$substr': ['$dt', 0, 7]}, '-01']}, 'code': '$secu'})}
    ])
    data1 = pd.DataFrame(list(data1))

    # tb2
    data2 = db[tb2].aggregate(pipeline=[
        {"$match": {'secu': {"$in": stock.code.unique().tolist()}, 'fp': {'$gte': '2007'}}},
        {'$project': merge_dicts(dict(zip(code_field_map[tb2].code, '$' + code_field_map[tb2].fd)),
                                 {'_id': 0, 'date': {'$concat': [{'$substr': ['$y', 0, 7]}, '-01']}, 'code': '$secu'})}
    ])
    data2 = pd.DataFrame(list(data2))

    # # tb3
    # data3 = db[tb3].aggregate(pipeline=[
    #     {"$match": {'secu': {"$in": stock.code.unique().tolist()}, 'dt': {'$in': quarter_end}}},
    #     {'$project': merge_dicts(dict(zip(code_field_map[tb3].code, '$' + code_field_map[tb3].fd)),
    #                              {'_id': 0, 'date': {'$concat': [{'$substr': ['$dt', 0, 7]}, '-01']}, 'code': '$secu'})}
    # ])
    # data3 = pd.DataFrame(list(data3))

    # data4
    data4 = db[tb4].aggregate(pipeline=[
        {"$match": {'secu': {"$in": stock.code.unique().tolist()}, 'fp': {'$gte': '2007'}}},
        {'$project': merge_dicts(dict(zip(code_field_map[tb4].code, '$' + code_field_map[tb4].fd)),
                                 {'_id': 0, 'date': {'$concat': [{'$substr': ['$y', 0, 7]}, '-01']}, 'code': '$secu'})}
    ])
    data4 = pd.DataFrame(list(data4))

    stock_factors_data = reduce(lambda x, y: pd.merge(x, y, on=['date', 'code'], how='outer'),
                                [data1, data2, data4])

    stock_factors_data.loc[:, 'date'] = pd.DatetimeIndex(stock_factors_data.date)
    stock_factors_data = stock_factors_data.set_index(['code', 'date'])
    stock_factors_data.to_hdf(keyword + u'_' + u'stock_factors.hdf', key='a')
    return stock_factors_data


def get_stock_code_via_orgid(org_id):
    # step3. 通过org.id 查找其股票代码, 过滤出中国的股票
    _, _, base_stock = get_mongo_connection(db_name='ada', tb_name='base_stock')
    code_reg = re.compile('[036]\d{5}_S[ZH]_EQ')
    id_list = org_id._id.tolist()
    stock = base_stock.aggregate(pipeline=[{"$match": {'org.id': {"$in": id_list}, 'code': code_reg}},
                                           {"$project": {'code': 1, 'tick': 1, 'abbr_szh': '$abbr.szh'}}
                                           ])
    stock = pd.DataFrame(list(stock))
    return stock


def get_org_id_via_keyword(keyword):
    _, _, base_org = get_mongo_connection(db_name='ada', tb_name='base_org')
    pattern = re.compile(keyword)
    org = base_org.aggregate(pipeline=[
        {"$unwind": '$ind.csf'},
        {'$project': {'zh_name': "$name.szh", '_id': 1, 'ind_csf_szh': "$ind.csf.szh"}},
        {"$match": {'ind_csf_szh': pattern}}
    ])
    org_id = pd.DataFrame(list(org))
    return org_id


def get_stock_factors_data_dict_form(stock_factors_data):
    code_field_map = pd.read_excel('/home/phil/dev/alpha/alpha/alpha_algo/quant_dict_factors_all.xlsx',
                                   sheetname='factors')
    code_field_map = code_field_map.query('stat == 1').dropna()
    code_field_map = dict(zip(code_field_map.code, code_field_map.sdt))
    stock_factors_data = stock_factors_data.loc[:, code_field_map.keys()]
    dic = {}
    for k, v in stock_factors_data.groupby(level=0):
        frame = v.reset_index(level=0, drop=True)
        dic[k] = {col: frame.loc[code_field_map[col]:, col].dropna() for col in frame.columns}
    return dic


def get_supply_chain_relation(prime_code, level=1, ptyp=None, rtyp=None, rdegree=None):
    """

    :param rdegree:
    :return:
    """
    filters = {'prime.cd': prime_code}
    if level:
        level_ = [level] if isinstance(level, int) else level
        level_ = [int(_) for _ in level_]
        filters['related.level'] = {'$in': level_}
    if ptyp:
        ptyp_ = [ptyp] if isinstance(ptyp, basestring) else ptyp
        ptyp_ = [_.upper() for _ in ptyp_]
        filters['ptyp'] = {'$in': ptyp_}
    if rtyp:
        rtyp_ = [rtyp] if isinstance(rtyp, basestring) else rtyp
        rtyp_ = [_.upper() for _ in rtyp_]
        filters['related.rtyp'] = {'$in': rtyp_}
    if rdegree:
        rdgree_ = [rdegree] if isinstance(rdegree, int) else rdegree
        rdgree_ = [int(_) for _ in rdgree_]
        filters['related.rdegree'] = {'$in': rdgree_}

    _, opt, _ = get_mongo_connection(db_name='opt')
    cursor = opt.supply_chain_relation.aggregate(pipeline=[
        {'$match': filters},
        {'$project': {'prime_cd': '$prime.cd',
                      'prime_szh': '$prime.szh',
                      'related_cd': '$related.name.cd',
                      'related_szh': '$related.name.szh',
                      'rtype': '$related.rtyp',
                      'level': '$related.level',
                      'rdegree': '$related.rdegree',
                      'ptyp': 1,
                      'pdegree': 1,
                      '_id': 0}},
    ])
    df_supply_chain_relation = pd.DataFrame(list(cursor))

    return df_supply_chain_relation


def get_dict_product_rs_map(level='4'):
    """

    :return:
    """

    level_ = [level] if isinstance(level, basestring) else  level
    _, _, dict_product_rs = get_mongo_connection(db_name='opt', tb_name='dict_product_rs')
    cursor = dict_product_rs.aggregate(pipeline=[
        {'$unwind': '$ind'},
        {'$match': {'level': {'$in': level_}, 'valid': '1', 'ind.t': 'csf'}},
        {'$project': {'szh_name': '$name.szh', 'csf_ind': '$ind.cd', 'code': 1, '_id': 0}},
    ])
    df = pd.DataFrame(list(cursor))
    return df


def get_fin_node_map():
    """

    :return:
    """
    _, opt, _ = get_mongo_connection(db_name='opt')
    fin_node_cursor = opt.fin_product_node.aggregate(pipeline=[
        {'$unwind': '$nodes'},
        {'$match': {'nodes.level': 1, 'secu': {'$regex': re.compile('^[036]')}}},
        {'$project': {'secu': 1, 'nodes_code': '$nodes.code', 'nodes_name_szh': '$nodes.name.szh',
                      'nodes_level': '$nodes.level', '_id': 0}}
    ])

    df_fin_node = pd.DataFrame(list(fin_node_cursor))

    node_secu_map = df_fin_node.groupby('nodes_code').agg({'secu': lambda x: tuple(x),
                                                           'nodes_name_szh': lambda x: x.iloc[-1]
                                                           }).sort_index()
    return node_secu_map
