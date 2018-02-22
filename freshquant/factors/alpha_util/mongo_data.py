# -*- coding: UTF-8 -*-
from pymongo import MongoClient
import pandas as pd


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


def form_mongo_to_df(tb_name, pos, filters):
    """
    输入mongo表连接，查询字段名(全字段，如quant.f18)，过滤条件
    tb_conn: mongo表连接
    pos: 查找的字段名
    filters：过滤条件，字典格式
    返回DataFrame格式
    """
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

if __name__=="__main__":
    stock = ['603726_SH_EQ']
    start = "2016-05-20"
    end = "2016-06-20"
    filters = {
        "dt": {"$lte": end, "$gte": start},
        'secu': {"$in": stock},
    }
    df=form_mongo_to_df('pr', ['pb', 'pe'], filters)
    print(df)
    print('Done!')
