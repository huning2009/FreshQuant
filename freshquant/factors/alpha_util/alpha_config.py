# -*- coding: UTF-8 -*-
'''
alpha 相关路径设置
'''
import os

data_pwd = 'E:/SUNJINGJING/Strategy/ALpha/data/origin'
FAC_RET_CAP_000905= os.path.join(data_pwd,'fac_ret_cap_000905.hd5')
FACTORS_DETAIL_PATH = os.path.join(data_pwd,'quant_dict_factors_all.csv')
FACTORS_DETAIL_PATH2 = os.path.join(data_pwd,'other_factors.csv')

FAC_RET_CAP_881001= os.path.join(data_pwd,'fac_ret_cap_881001.hd5')
WIND_A= os.path.join(data_pwd,'881001.csv')
FAC_RET_CAP= os.path.join(data_pwd,'fac_ret_cap.hd5')

strategy_pwd = 'E:/SUNJINGJING/Strategy'
TRADE_CAL_PATH = os.path.join(strategy_pwd,'utils_strategy/trade_cal.csv')

ALL_STOCKS='E:/SUNJINGJING/Strategy/data/stocks_ind.csv'

# Mongo数据库
DEFALUT_HOST = '122.144.134.95'              # 因子数据 95或者33mysql.quant
DEFALUT_HOST_COMPONETS = '122.144.134.95'   # 200 ada.index_members_a   历史成分股

# sql 数据  95 mysql.ada-fd.hq_price

# 方法选择
EX_METHOD = 'mad'  #　标准化std, mad
SCALE_METHOD = 'normal'  # normal, cap, sector

analysis_pwd = '/media/jessica/00001D050000386C/SUNJINGJING/Strategy/ALpha/data/analysis'
GOOD_FACS='/media/jessica/00001D050000386C/SUNJINGJING/Strategy/ALpha/data/analysis/good_fac.hd5'

# 全部a股高开低收
BACK_VIEW='/media/jessica/00001D050000386C/SUNJINGJING/Strategy/data/back_view_2016_0817_no_null.csv'

