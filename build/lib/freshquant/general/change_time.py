# -*- coding: utf-8 -*-
__author__ = 'jessica'

import pytz,datetime

def beijing_to_utc(dt_lst):
    """
    将列表转换
    """
    local=pytz.timezone("Asia/Shanghai")
    dt_lst=[datetime.datetime.strptime(dt,"%Y-%m-%d") for dt in dt_lst]
    local_dt=[local.localiza(dt,is_dst=None) for dt in dt_lst]
    utc_dt=[dt.astimezone(pytz.utc) for dt in local_dt]
    return utc_dt

