# -*- coding: UTF-8 -*-
import pandas as pd

# from csf_utils import engine_company_outer as engine
from sqlalchemy import create_engine
from joblib import Parallel, delayed

engine = create_engine(
    'mysql://pd_team:pd_team321@!@122.144.134.21/ada-fd?charset=utf8')

def get_data(date_tuple):
    start_date, end_date = date_tuple
    sql_str = """Select dt, tick, open, high, low, close, vol as volume
    From hq_price_after
    where dt between '{}' and '{}'
    """
    sql_str = sql_str.format(start_date, end_date)
    print(sql_str)
    print(('start getting data from {} to {}'.format(start_date, end_date)))
    df = pd.read_sql_query(sql_str, engine, chunksize=10000)
    df = pd.concat(df)
    df.to_csv(''.join([start_date, "____", end_date, ".csv"]))
    print(('data from {} to {} are downloaded'.format(start_date, end_date)))

def merge_data():
    import glob
    g=glob.glob('*-31.csv')
    frames = [pd.read_csv(p, dtype={"tick": object}) for p in g]
    df = pd.concat(frames)
    return df

def download_index_data(start_date,end_date):
    sql_str = """Select dt, tick, close
    From hq_index
    where dt between '{}' and '{}' and tick = 000300
    """
    sql_str = sql_str.format(start_date, end_date)
    print(sql_str)
    print(('start getting data from {} to {}'.format(start_date, end_date)))
    df = pd.read_sql_query(sql_str, engine, chunksize=10000)
    df = pd.concat(df)
    df.to_csv(''.join([start_date, "____", end_date, ".csv"]))
    print(('data from {} to {} are downloaded'.format(start_date, end_date)))

def update_data(df):
    return df

if __name__ == '__main__':
    # 更新数据
    df=pd.read_csv('hq_price_after.csv')
    new_df=update_data(df)
    print('Done!')

    # 下载股票数据
    # dt_rng = range(2014,2015)
    # start_dates = [str(year) + '-01-01' for year in dt_rng]
    # end_dates = [str(year) + '-12-31' for year in dt_rng]
    # dt_tuple = zip(start_dates, end_dates)
    # Parallel(n_jobs=1)(delayed(get_data)(d) for d in dt_tuple)

    # 拼接数据
    # df=merge_data()
    # df.to_csv('hq_price_after.csv')
    # print df.head()

    # 下载指数数据
    # start_date='2006-01-01'
    # end_date='2016-08-20'
    # download_index_data(start_date,end_date)
