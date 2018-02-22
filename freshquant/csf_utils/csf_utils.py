# -*- coding: utf-8 -*-
import itertools
import logging
from collections import namedtuple
from functools import wraps
from itertools import cycle, islice, chain
from time import time

import pandas as pd
from ._portability import izip_longest
from pymongo import MongoClient
from sqlalchemy import create_engine

DataBaseHostInfo = namedtuple('DataBaseHostInfo',
                              ['ip', 'username', 'password'])

local = DataBaseHostInfo('127.0.0.1', 'root', 'root')
remote1 = DataBaseHostInfo('122.144.134.95', 'quant_team', 'quant_team321@!')
remote2 = DataBaseHostInfo('122.144.134.21', 'quant_team', 'quant_team321@!')
quant = DataBaseHostInfo('192.168.100.33', 'csf_quant', 'csf_quant@2015')

protocol = ''
try:
    import MySQLdb

    protocol = 'mysql://'
except ImportError:
    protocol = 'mysql+mysqlconnector://'

db_url = {
    'engine_phil_machine_test': '{}root:root@127.0.0.1/test?charset=utf8'.format(protocol),
    'engine_phil_machine': '{}root:root@127.0.0.1/ada-fd?charset=utf8'.format(protocol),
    'engine_company_local': '{}{}:{}@{}/ada-fd?charset=utf8'.format(protocol, remote1.username, remote1.password,
                                                                    remote1.ip),
    'engine_company_outer': '{}{}:{}@{}/ada-fd?charset=utf8'.format(protocol, remote2.username, remote2.password,
                                                                    remote2.ip),
    'engine_quant': '{}{}:{}@{}/quant?charset=utf8'.format(protocol, quant.username, quant.password, quant.ip)
}

engine_company_local = create_engine(db_url['engine_company_local'])
engine_company_outer = create_engine(db_url['engine_company_outer'])
engine_quant = create_engine(db_url['engine_quant'])
engine_phil_machine = create_engine(db_url['engine_phil_machine'])

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import basestring
except ImportError:
    basestring = str

def batch(iterable, size):
    """
    yield iterable in batch.
    :param iterable, an iterable object
    :param size, size of batch

    ------------------------------
    Example:
    In [6]: for b in batch([1,2,3,4,5], 2):
            print(list(b))
   ...:
            [1, 2]
            [3, 4]
            [5]
    """
    source_iter = iter(iterable)
    while True:
        batch_iter = islice(source_iter, size)
        yield chain([next(batch_iter)], batch_iter)


def take(iterable, n):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def paste(first, *rest, **sep):
    """
    合并两个或多个序列，转成字符格式.
    短的序列会不断的重复自身，直到达到最长序列的长度
    与R语言的paste 类似
    Args:
        first, second, ...,
    Returns:
        ret, 合并后的序列
    Examples:
    ----------------------------------------------------
    In [7]: iter1 = [1,2,3,4]

    In [8]: iter2 = 'a'

    In [9]: iter3 = 'xy'

    In [10]: paste(iter1,iter2,iter3,sep=',')
    Out[10]: ['1,a,x', '2,a,y', '3,a,x', '4,a,y']
    """
    separator = sep.pop('sep', '')
    max_len = max(len(first), max(len(it) for it in rest))
    all_iterables = (first,) + rest
    all_iterables = [take(cycle(it), max_len) for it in all_iterables]
    tup = izip_longest(*all_iterables)
    ret = [separator.join(map(str, item)) for item in tup]
    return ret


def get_mongo_connection(host='122.144.134.95', db_name=None, tb_name=None):
    """
    mongo连接设置
    :param host, ip_address
    :param db_name, database name
    :param tb_name, table name
    """
    db = None
    tb = None
    client = MongoClient(host, 27017)
    if db_name:
        db = client[db_name]
        if tb_name:
            tb = db[tb_name]
    return client, db, tb


class GetPriceData(object):
    """
    数据读取的种类：
    (1) 给定日期的前n天
    (2) 给定日期的后几天
    (3) 两个给定日期之间
    (4) 最后一个日期的收盘价

    两种返回类型：
    (1) pandas.DataFrame
    """

    def __init__(self, sid, index=False, database='ada-fd', table=None, data_source=None, engine=None,
                 field=('tick', 'open', 'high', 'low', 'close', 'volume')):
        """

        :param table:
        :param engine:
        :param field:
        :@params data_source,
        :@params sid, the security ID of the stock
        """
        if engine is not None:
            self.engine = engine
        elif len(data_source) > 1:
            if data_source[1] == 'local':
                self.engine = engine_phil_machine
            elif data_source[1] == 'remote1':
                self.engine = engine_company_local
            elif data_source[1] == 'remote2':
                self.engine = engine_company_outer
        else:
            self.engine = engine_company_outer
        self.ticks = sid
        if table is not None:
            self.__table = table
        else:
            self.__table = 'hq_index' if index else 'hq_price'

        self.field = field

    def __get_data(self, sql):
        """

        :rtype : object
        """
        ret = None
        try:
            ret = pd.read_sql_query(sql, self.engine, chunksize=5000)
        except Exception as e:
            logger.critical(e.message)
        ret = pd.concat(ret, axis=0)
        ret.rename(columns={'vol': 'volume'}, inplace=True)
        ret.set_index(['dt'], inplace=True)
        ret = ret.sort_index()
        ret.index = pd.DatetimeIndex(ret.index)
        field = self.field or ret.columns
        field = set(field) - set(['dt'])
        ret = ret.loc[:, field]
        return ret

    def get_n_days_backward(self, end_date, n_days):
        """
        this function get the price data of n_days backward from the end_date to the end_date
        :param end_date: the querying end date
        :param n_days: n_days backward
        :return: the DataFrame
        """
        sql = """
            SELECT dt,open,high,low,close,vol FROM `ada-fd`.`{table}` WHERE
            tick = '{tick}' and dt <= '{dt}' and vol > 0
            order by dt DESC LIMIT {n}
              """.format(
            table=self.__table,
            tick=self.ticks,
            dt=end_date,
            n=n_days)

        ret = self.__get_data(sql)
        if len(ret) != n_days:
            err_str = """you are requesting {} rows for tick {},
             however, database returns {} rows."""
            raise ValueError(err_str.format(n_days, self.ticks, len(ret)))
        return ret

    def get_n_days_forward(self, start_date, n_days):
        """
        this function get the price data of n_days forward from the start_date
        to the end_date
        :param start_date: the querying end date
        :param n_days: n_days forward
        :return: the DataFrame
        """
        sql = """
            SELECT dt,'open',high,low,close,vol FROM `ada-fd`.{table}
            WHERE tick = '{tick}' and dt >= '{start_date}' and vol > 0
            order by dt ASC LIMIT {n_days}
        """.format(table=self.__table,
                   tick=self.ticks,
                   start_date=start_date,
                   n_days=n_days)

        ret = self.__get_data(sql)

        if len(ret) != n_days:
            err_str = """you are requesting {} rows for tick {},
             however, database returns {} rows."""
            raise ValueError(err_str.format(n_days, self.ticks, len(ret)))
        return ret

    def get_between_dates(self, start_date, end_date):
        """
        this function get the price data between start_date and end_date
        :param start_date: the querying start date
        :param end_date: the querying end date
        :return: the DataFrame
        """

        sql = """
            SELECT dt,open,high,low,close,vol FROM `ada-fd`.{table}
            WHERE tick = '{tick}' and dt between '{start_date}'
            and '{end_date}' and vol > 0
            order by dt ASC
        """.format(table=self.__table,
                   tick=self.ticks,
                   start_date=start_date,
                   end_date=end_date)

        ret = self.__get_data(sql)
        return ret

    def get_in_spec_dates(self, date_lst):
        stocks = ensure_tuple(self.ticks)
        date_lst_ = ensure_tuple(date_lst)
        sql = """SELECT dt, tick, close
                 FROM hq_price
                 WHERE dt IN %s AND tick IN %s
                 ORDER BY dt ASC""" % (
            tuple(date_lst_), tuple(stocks))
        return self.__get_data(sql)

    def get_last_price(self):

        sql = "SELECT dt,open,high,low,close,vol as volume " \
              "FROM `ada-fd`.%s " \
              "WHERE tick = '%s' and vol != 0 order by dt DESC LIMIT 1" \
              % (self.__table, self.ticks)

        ret = self.__get_data(sql)

        return ret

    def batch_get_n_days_backward(self, dt, n_days):
        ticks_ = ensure_tuple(self.ticks)
        sql = """
            SELECT *
            from (
             select *,
             @num := if(@grp=tick,@num+1,1) as row_number,
             @grp := tick as dummy
             from {table}
             JOIN (SELECT @grp:=NULL, @num:=0) AS vars
             where tick in {ticks} and dt < '{dt}' and vol > 0
             order by tick, dt desc
            ) as x
            where x.row_number <= {n_days};
            """.format(ticks=ticks_, table=self.__table, dt=dt, n_days=n_days)
        # logger.debug(sql)
        ret = self.__get_data(sql)
        return ret

    def batch_get_between_dates(self, start_date, end_date):
        ticks_ = ensure_tuple(self.ticks)
        sql = """
            SELECT *
            FROM {table}
            where tick in {ticks} and dt between '{start}' and '{end}'
            and vol > 0
            order by tick, dt desc
            """.format(ticks=ticks_, table=self.__table, start=start_date,
                       end=end_date)
        ret = self.__get_data(sql)
        # for tick, data in ret.groupby(['tick']):
        #     yield tick, data.sort_index()
        return ret

    def batch_get_data(self, lookback, start_date, end_date):
        """
        lookback: n_days before start_date for all stocks
        :param end_date:
        :param start_date:
        :param lookback:
        """
        ret1 = self.batch_get_between_dates(start_date, end_date)
        ret2 = self.batch_get_n_days_backward(start_date, lookback)
        ret = ret1.combine_first(ret2)
        # for tick, data in ret.groupby(['tick']):
        #     yield tick, data.drop(['tick'], axis=1)
        return ret


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time()
        ret = func(*args, **kwargs)
        t1 = time()
        print('{} cost {} seconds'.format(func.__name__, t1 - t0))
        return ret

    return wrapper


def rolling(seq, window=2, longest=False):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    Args:
        longest: if True, get full length of seq,
        e.g. window([1,2,3,4], 3, longest=True) --->
        (1,2,3), (2,3,4), (3,4,None), (4,None,None)
    """
    n = window
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


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def ensure_tuple(ticks):
    ticks_ = ''
    if len(ticks) > 0:
        if isinstance(ticks, basestring):
            ticks_ = tuple([ticks])
        elif isinstance(ticks, (tuple, list)):
            # make u'000001' --> '000001', mysql does not support u'000001'
            ticks_ = tuple(ticks)

        if len(ticks_) == 1:
            ticks_ = "('{}')".format(ticks_[0])
        return ticks_
    else:
        raise ValueError('ticks is None or empty. ticks:{}'.format(ticks))


def show_chinese_character(ax):
    import os
    import itertools
    from matplotlib.font_manager import FontProperties
    if os.name == 'posix':
        fname = r'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    elif os.name == 'nt':
        fname = r"c:\windows\fonts\simsun.ttc"
    font = FontProperties(fname=fname, size=10)

    for label in itertools.chain(ax.get_xticklabels(), ax.legend().texts, [ax.title]):
        label.set_fontproperties(font)
    return ax
