# -*- coding: UTF8  -*-
"""
事件驱动：
传统事件驱动：高管增持、定增破发、股权激励、业绩超预期和业绩预增
网络文本挖掘事件驱动：公告披露，新闻热度变化及股吧情绪异常
一致预期事件驱动：实际指标偏离预测指标事件、一致预期截面优势、一致预期变化异常
"""
import numpy as np
import pandas as pd
import os
from functools import reduce
import datetime
from sqlalchemy import create_engine

from general.mongo_data import general_form_mongo_to_df
from general.get_stock_data import get_all_stocks_basic
from freshquant.event.world_event import *

from freshquant.event.event_analysis import *
pd.set_option("display.max_rows", 12)
pd.set_option("display.max_columns", 12)

"""
返回的数据最好是固定格式：secu，dt，event_type,fac1,fac2,fac3
方便做多事件策略，并且在事件基础上打分
"""

class Jessica_EVENT(object):
    def __init__(self,stg,indexcode=None,window1=None,window2=None,ret_kind='cap'):
        self.stg=stg
        self.indexcode=indexcode
        self.window1=window1
        self.window2=window2
        self.ret_kind='cap'
        self.ret=self.get_event_ret()
    def get_event_ret(self):
        lst_ret=[getattr(self,self.stg[i])() for i in range(len(self.stg))]
        if len(lst_ret)==1:
            df=lst_ret[0]
        else:
            df=reduce(lambda x,y:pd.merge(x,y,on=['dt','secu'],how='outer'),lst_ret)
        df=df.set_index(['dt','secu']).sort_index()
        # df=df[df.sum(axis=1)==df.shape[1]]
        return df
    def zengchi(self):
        df=World_Event(event='zengchi').data
        # 筛选增持,中小市值，增持力度》0.05%，公告及时性，低PB
        df=df[(df['rate'] >0.05)&(df['direction']=='增持')
              &(df['timing']<=datetime.timedelta(3))&(df['股东类型']=='高管')
              &(df['lasting']<=datetime.timedelta(2))&(df['变动部分参考市值(万元)']>=300)]
        df=df.drop_duplicates(['dt','secu'])
        df['zengchi']=1
        return df[['dt','secu','zengchi']]
    def performance(self):
        df = World_Event(event='performance').data
        # 先处理预告次数超过1的公司数据
        # 筛选业绩预增、业绩略增和扭亏
        df=df[(df['typ'].isin(['预增','略增','扭亏']))&(df['预告次数']==1)]
        df=df[df['同比增长下限(%)']>30]
        df['performance']=1
        return df[['dt','secu','performance']]
    def jiejin(self):
        """
        筛选指标：
         观测时间：临近解禁前后
         流通市值：市值不应过大
        打分优选指标：
         企业特征：优先选择民营企业
         短期业绩增长：以净利润增长情况确定业绩爆发的可能性
         筹码不大幅分散： 筹码大幅分散股票风险较高
         短期股价涨幅不高： 寻找短期涨幅不高的股票
         短期关注度低：通过技术指标寻找未被市场过多关注的个股
        """
        df=World_Event(event='jiejin').data
        return df
    def employee_ownership(self):
        """员工持股"""
        df = World_Event(event='employee_ownership').data
        df=df[df.dt.notnull()]
        df = df[df['secu'].map(lambda x: 'S' in str(x))]
        # df=df[df['方案进度'].map(lambda x:'董事会通过' in x or '实施完成' in x)]
        df =df[(df['股票来源'].map(lambda x:('非公开发行' in str(x))|('竞价转让' in str(x))))
               &(df['占总股本(%)']<2)&(df['高管认购比例']>10)]
        df=df.drop_duplicates(['dt','secu'])
        df['employee_ownership']=1
        return df[['dt','secu','employee_ownership']]
    def equity_incentives(self):
        """股权激励"""
        df = World_Event(event='equity_incentives').data
        df = df[df['secu'].map(lambda x: 'S' in str(x))]
        df=df[(df['激励标的物']=='股票')&(df['方案进度']=='董事会预案')]
        df['equity_incentives']=1
        return df[['dt','secu','equity_incentives']]
    def analysist(self):
        """分析师评级"""
        return df.set_index(['dt', 'secu']).sort_index()
    def Stocks_Heavily_Held(self):
        df=World_Event(event='Stocks_Heavily_Held').data
        return df.set_index(['dt', 'secu']).sort_index()
    def Add_Issuance(self):
        """增发"""
        df = World_Event(event='Add_Issuance').data
        return df.set_index(['dt', 'secu']).sort_index()
    def capital_flow(self):
        """资金流向"""
        return df.set_index(['dt', 'secu']).sort_index()
    def profit_distribution(self):
        """利润分配"""
        df = World_Event(event='profit_distribution').data
        return df.set_index(['dt', 'secu']).sort_index()
    def earlier_annual_report(self):
        """提早发布年报"""
        df = World_Event(event='earlier_annual_report').data
        return df.set_index(['dt', 'secu']).sort_index()
    def no_announcement(self):
        """
        长期不发公告：计算每个股票所有公告距离上公告的天数。
        d>=60,基本能实现超额收益
        """
        df = World_Event(event='all_announcements').data
        # 计算公告发布天数
        df=df.set_index('secu')
        df.dt=df.dt.map(lambda x:pd.Timestamp(x))
        df.loc[:,'days']=df.groupby(level=0).apply(lambda x:x.sort_values(by='dt').dt.diff()).values
        df=df[df.days.notnull()]
        df.days=df.days.map(lambda x:int(str(x).split('days')[0]))
        #days>60
        df=df[df.days>=60]
        df.loc[:,'dt_buy']=df.dt-datetime.timedelta(60)
        df.loc[:,'dt_sell']=np.where(df.days>80,20,2)
        df.dt_sell=df.dt+df.dt_sell.map(datetime.timedelta)
        return df


# if __name__=='__main__':
    # stg=['performance']
    # ins=Jessica_EVENT(stg=stg,indexcode='000001',window1=30,window2=100,ret_kind='cap')
    # print (ins.ret)
    # print ('Done!')
