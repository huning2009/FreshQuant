# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import csf
from sqlalchemy import create_engine
from general.util import set_mongo_cond
from general.mongo_data import general_form_mongo_to_df
from alpha_util.alpha_config import ALL_STOCKS
from .utils_stock import batch
from joblib import Parallel, delayed
import os
from six import string_types

def get_all_stocks_basic():
    # 获取证券基本信息base_stock
    df_base=general_form_mongo_to_df('192.168.250.200','ada','base_stock',
                                     pos=['code','abbr.szh','ls.dt','ls.edt'],
                                     filters={'mkt.code':{'$in':['1001','1002','1003','1012']},
                                              'ls.edt':None})
    df_base=df_base.rename(columns={'code':'secu','abbr.szh':'szh','ls.dt':'listing_date'})
    return df_base

def get_trade_calendar():
    """
    从trade_cal.csv文件读取交易日历数据
    """
    file_path = os.path.abspath(__file__)
    dir_name = os.path.split(file_path)[0]
    csv_file = os.path.join(dir_name, 'trade_cal.csv')
    trade_cal = pd.read_csv(
        csv_file,
        names=['date_time', 'total_day'],
        index_col=[0],
        parse_dates=True)
    return trade_cal


def get_cap_data_from_api(index_code, start_date, end_date, freq='M'):
    """
    总市值数据
    :param index_code: str, 指数代码，如"000300"
    :param start_date: str, 开始日期，如"2008-04-30"
    :param end_date: str，结束日期，如"2015-12-31"
    :param freq: str，换仓周日，周"W"、月"M"、季"Q"
    :return: pd.DataFrame，因子值
    """
    return get_raw_factor_from_api('M004023', index_code, start_date, end_date, freq)


def get_cap_data_from_mongo(index_code, start_date, end_date, freq='M'):
    """
    总市值数据
    :param index_code: str, 指数代码，如"000300"
    :param start_date: str, 开始日期，如"2008-04-30"
    :param end_date: str，结束日期，如"2015-12-31"
    :param freq: str，换仓周日，周"W"、月"M"、季"Q"
    :return: pd.DataFrame，因子值
    """
    return get_raw_factor_from_mongo('M004023', index_code, start_date, end_date, freq)


def get_stock_lst_date_from_api(codes):
    """
    股票首次上市日期
    Args:
        codes (list): 股票代码列表
    Returns:
        DataFrame, 两列, 一列是code, 六位股票代码, 一列是listing_date
    """
    ipo_info = Parallel(
        n_jobs=20, backend='threading', verbose=5)(
        delayed(csf.get_stock_ipo_info)(code, field=['code', 'dt'])
        for code in codes)
    ipo_info = pd.concat(ipo_info, ignore_index=True)
    ipo_info.loc[:, 'code'] = ipo_info.code.str.slice(0, 6)
    ipo_info = ipo_info.rename(columns={'dt': 'listing_date'})
    return ipo_info

def get_stock_lst_date_local(codes):
    """
    股票首次上市日期
    Args:
        codes (list): 股票代码列表,有后缀
    Returns:
        DataFrame, 两列, 一列是code, 六位股票代码, 一列是listing_date
    """
    pos = ['code','ls.dt']
    filters = {"code": {"$in": codes}}
    df=general_form_mongo_to_df(host='192.168.250.200',db_name='ada',tb_name='base_stock',pos=pos,filters=filters)
    df=df.rename(columns={'ls.dt':'listing_date','code':'secu'})
    return df


def get_csf_index_factor_data():
    """
    获取数库行业指数因子数据
    :return:
    """

    pass


def get_industries(stocks):
    """
    获取股票对应的行业代码
    Args:
        stocks(list) : 股票列表

    Returns:
        DataFrame: 股票及其对应行业代码
    """
    return [csf.get_stock_csf_industry(
        codes, field=['code', 'level2_name']) for codes in batch(
        stocks, n=90)]


def get_stock_industry_from_mongo(codes, ind):
    """
    获取股票所属行业
    sam_level=1,'sw','zx'

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
    if ind == 'csf':
        sam_level = 1
        host = '192.168.250.200'
        if not isinstance(codes, list):
            codes = list(codes)
        _, opt = set_mongo_cond(host=host, db_name='opt')
        _, ada = set_mongo_cond(host=host, db_name='ada')
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
        dic_ind = df.to_dict()['ind']
    if ind == 'sw':
        pure_codes = [secu[:6] for secu in codes]
        s_ind = pd.read_csv(ALL_STOCKS, index_col=0)['ind']
        dic_ind = {inx.replace('.', '_') + '_EQ': s_ind.loc[inx] for inx in s_ind.index if inx[:6] in pure_codes}
    return dic_ind


def get_stock_industry_from_api(codes):
    """
    股票所属数库一级行业
    :param codes: list, 股票代码列表
    :return: 股票与对应行业
    """
    codes_len = len(codes)
    fields = ['code', 'secu_name', 'level1_name', 'level1_code']

    if codes_len > 100:
        cutter = list(range(0, codes_len, 100))
        cutter.append(codes_len - 1)
        dict_cutter = list(zip(cutter[0:-1], cutter[1:]))
        df = pd.DataFrame()
        for i, j in dict_cutter:
            sub_codes = codes[i:j]
            temp = csf.get_stock_industry(sub_codes, field=fields)
            df = pd.concat([df, temp])
        return df
    else:
        return csf.get_stock_industry(codes, field=fields)

def get_stock_market_data(code,start,end,field=None):
    """
    mysql+mysqlconnector://ada_user:ada_user@122.144.134.3/ada-fd
    """
    sql_address = 'mysql+mysqlconnector://pd_team:pd_team123@!@192.168.250.200/ada-fd'
    engine=create_engine(sql_address)
    if not isinstance(code,list) and len([code])==1:
        sql = """SELECT * FROM hq_price_before WHERE dt >= '%s' and dt <='%s' AND tick = %s ORDER BY dt DESC""" % (
        start, end, code)
    else:
        sql = """SELECT * FROM hq_price_before WHERE dt >= '%s' and dt <='%s' AND tick IN %s ORDER BY dt DESC""" % (start,end, tuple(code))
    df=pd.read_sql(sql, engine)
    return df

class GetPrice(object):
    def __init__(self,
                 host="192.168.250.200",
                 db_name='ada-fd',
                 username='pd_team',
                 password='pd_team123@!',
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

    def _get_from_sql(self, sql, all=False):
        '''生成想要的格式'''
        engine = self.engine
        if self.isIndex:
            sql = sql.replace('hq_price_before', 'hq_index')
        elif all == True:
            sql = sql.replace('close', 'open,close,high,low,vol,inc')
        get_sql = pd.read_sql(sql, engine)
        if not self.isIndex:
            get_sql['tick'] = get_sql['tick'].map(
                lambda x: x + '_SH_EQ' if x[0] == '6' else x + '_SZ_EQ')
        get_sql = get_sql.sort_values(by='dt')
        ar = get_sql.ix[:, ['dt', 'tick']].T.values
        mul_idx = pd.MultiIndex.from_arrays(ar)
        if all == False:
            data = pd.Series(index=mul_idx, data=get_sql.close.values)
            data = data.unstack()
        else:
            data = pd.DataFrame(index=mul_idx, columns=['open', 'close', 'high', 'low', 'vol', 'inc'],
                                data=get_sql[['open', 'close', 'high', 'low', 'vol', 'inc']].values)
        return data

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
        return (df_pr.iloc[1:, ] / 100 + 1).product() - 1

    def get_n_days_back(self, stocks, end_date, days):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
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

    def get_n_days_after(self, stocks, start_date, days):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt <='%s' AND tick = '%s'
                     ORDER BY dt DESC LIMIT %d""" % (
                start_date, stocks[0], days)
        else:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt <='%s' AND tick IN %s
                     ORDER BY dt DESC LIMIT %d""" % (
                start_date, tuple(stocks), days)
        return self._get_from_sql(sql)

    def get_between_dates(self, stocks, start_date, end_date):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt between '%s' AND '%s' AND tick = %s
                     ORDER BY dt ASC""" % (
                start_date, end_date, stocks[0])
        else:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt between '%s' AND '%s' AND tick in %s
                     ORDER BY dt ASC""" % (start_date, end_date, tuple(stocks))
        return self._get_from_sql(sql)

    def jessica_get_between_dates(self, stocks, start_date, end_date):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, inc
                     FROM hq_price_before
                     WHERE dt between '%s' AND '%s' AND tick = %s
                     ORDER BY dt ASC""" % (
                start_date, end_date, stocks[0])
        else:
            sql = """SELECT dt, tick, inc
                     FROM hq_price_before
                     WHERE dt between '%s' AND '%s' AND tick in %s
                     ORDER BY dt ASC""" % (start_date, end_date, tuple(stocks))
        return self.jessica_get_from_sql(sql)

    def get_in_spec_dates(self, stocks, date_lst, all=False):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt IN %s AND tick = %s
                     ORDER BY dt ASC""" % (
                tuple(date_lst), stocks[0])

        elif len(date_lst) == 1:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt = %s AND tick IN %s """ % (
                date_lst[0], tuple(stocks))

        else:
            sql = """SELECT dt, tick, close
                     FROM hq_price_before
                     WHERE dt IN %s AND tick IN %s
                     ORDER BY dt ASC""" % (
                tuple(date_lst), tuple(stocks))
        return self._get_from_sql(sql, all=all)


class GetMinute(object):
    def __init__(self,
                 host="192.168.100.33",
                 db_name='wande',
                 username='csf_quant',
                 password='csf_quant@2015',
                 isIndex=False):
        self.sql_addr = 'mysql://%s:%s@%s/%s' % (
            username, password, host, db_name)
        self.engine = self.set_sql_con()

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
        return stocks

    def _get_from_sql(self, sql):
        '''生成想要的格式'''
        engine = self.engine
        get_sql = pd.read_sql(sql, engine)
        get_sql = get_sql.sort_values(by=['ndate', 'ntime'])
        return get_sql

    def get_between_dates(self, stocks, start_date, end_date):
        stocks = self._check_codes(stocks)
        if len(stocks) == 1:
            sql = """ SELECT code,ndate,ntime,nopen,nclose,ivolume FROM BS_WIND_H_T_KLINE_FD_CQ_MIN_%s
            WHERE ivolume != 0 and match_items !=0 and wind_code = %s AND ndate BETWEEN %s AND %s """ % (
                start_date[:4], stocks[0], start_date, end_date)
        else:
            sql = """ SELECT code,ndate,ntime,nopen,nclose,ivolume FROM BS_WIND_H_T_KLINE_FD_CQ_MIN_%s
            WHERE ivolume != 0 and match_items !=0 and wind_code IN %s AND ndate BETWEEN %s AND %s """ % (
                start_date[:4], tuple(stocks), start_date, end_date)
        return self._get_from_sql(sql)


def get_raw_factor_from_api(factors, index_code, start_date, end_date, freq='M'):
    """
    原始因子值（未经处理过）
    :param factors: str or list, 因子代码"M009006"或因子代码列表["M009006", "M009007"]
    :param index_code: str, 指数代码，如"000300"
    :param start_date: str, 开始日期，如"2008-04-30"
    :param end_date: str，结束日期，如"2015-12-31"
    :param freq: str，换仓周期，周"W"、月"M"、季"Q"，每个周期的最后一个交易日
    :param filter: dict, 股票筛选
    :return: pd.DataFrame，因子值
    """
    # csf.get_stock_factor 一次最多取５个因子
    frame_list = []
    for b in batch(factors, 5):
        temp = csf.get_stock_factor_by_index(
            factors=b,
            index=index_code,
            start_date=start_date,
            end_date=end_date,
            freq=freq)
        frame_list.append(temp)
    frame = pd.concat(frame_list, ignore_index=True)
    df = pd.pivot_table(
        frame, values='value', index=['date', 'code'], columns=['cd'])
    return df


def get_raw_factor_from_mongo(factors, index_code, start_date, end_date, freq='M'):
    """
    一次性获取所有调仓期，所有股票的因子数据

    Args:
        FACTORS_DETAIL_PATH:
        fac_lst:
        rpt_terms:
        all_stocks:
        dt_index:
        multi_index: (dt,secu)

    Returns:
        object:
    """
    return


def get_benchmark_return_from_api(bench_code, start_date, end_date, dt_index):
    """
    BenchMark收益率

    Args:
        bench_code (str): benchMark代码，如'000300'
        start_date (str): 开始日期
        end_date (str): 结束日期
        dt_index (list): a list of str, 月末/周末/季末 dt
    Returns:
        DataFrame
    """
    field = ['close']
    df = csf.get_index_hist_bar(
        index_code=bench_code,
        start_date=start_date,
        end_date=end_date,
        field=field)
    if isinstance(df.index[0], pd.Timestamp):
        df.index = df.index.map(lambda dt: str(dt.date()))
        df.index.name = 'date'
    price = df[field].ix[dt_index, :].rename(
        columns={'close': 'benchmark_returns'}).sort_index()
    ret = price.pct_change().shift(-1).dropna()
    return ret


def get_stock_returns_from_api(stocks, start_date, end_date, freq):
    """
    获取股票收益率
    Args:
        stocks (Iterable): 股票序列
        start_date (str): 开始日期
        end_date (str): 结束日期
        freq (str): 频度, {'W','M','Q'}
                W: 每周
                M: 每月
                Q: 每季度

    Returns:
        DataFrame: multi-index, level0=date, level1=code. 收益率, 仅有一列,列名称为'ret'
    """
    # ret_lst = parallel(n_jobs=8, delayed_func=[
    #     delayed(csf.get_stock_hist_returns)(b, freq, start_date=start_date, end_date=end_date) for b in
    #     batch(stocks, 20)])

    ret_lst = [csf.get_stock_hist_returns(code, freq, start_date, end_date) for code in stocks]
    returns = pd.concat(ret_lst).dropna(how='all')
    returns = returns.set_index(['date', 'code']).sort_index()
    return returns


def prepare_data_from_api(factor_name, index_code, benchmark_code, start_date, end_date, freq):
    """
    获取因子数据,股票市值,股票对应的下期收益率,下期benchmark收益率
    Args:
        benchmark_code: 一个指数代码, 例如'000300'
        factor_name (str): 因子名称, 例如 'M004023'
        index_code (str): 六位指数代码, 例如 '000300'
        start_date (str): 开始日期, YYYY-mm-dd
        end_date (str): 结束日期, YYYY-mm-dd
        freq (str): 数据频率, m=month
    Returns:
        DataFrame: multi-index, level0=date, level1=code. 原始因子, 其下期收益率, 市值, benchmark下期收益率
    """
    if isinstance(factor_name, string_types):
        factor_name_ = [factor_name]
    elif isinstance(factor_name, (list, tuple)):
        factor_name_ = list(factor_name)
    if 'M004023' not in factor_name_:
        factor_name = factor_name_ + ['M004023']
    else:
        factor_name = factor_name_
    factor_name = [str(n) for n in factor_name]
    raw_fac = get_raw_factor_from_api(
        factor_name, index_code, start_date, end_date, freq)
    raw_fac = raw_fac.rename(columns={'M004023': 'cap'})
    if 'M004023' in factor_name_:
        raw_fac.loc[:, 'M004023'] = raw_fac.cap

    dts = sorted(raw_fac.index.get_level_values(0).unique())
    s, e = str(dts[0]), str(dts[-1])

    benchmark_returns = get_benchmark_return_from_api(bench_code=benchmark_code, start_date=start_date,
                                                      end_date=end_date,
                                                      dt_index=dts)
    stocks = sorted([str(c)
                     for c in raw_fac.index.get_level_values(1).unique()])
    returns = get_stock_returns_from_api(stocks, s, e, freq)
    returns.columns = ['ret']

    # 去掉最后一期数据
    inx = raw_fac.index.get_level_values(0).unique()[:-1]
    raw_fac = raw_fac.loc[pd.IndexSlice[inx, :], :]
    fac_ret = raw_fac.join(returns)

    fac_ret = fac_ret.join(benchmark_returns)

    return fac_ret



if __name__ == '__main__':
    print('Done!')
