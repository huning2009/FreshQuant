
# -*- coding: UTF8  -*-
"""
world_alpha:该模块用于计算因子，一次只计算一个因子，因子分析相关模块抽离出去。

https://www.ricequant.com/community/topic/2129/?utm_source=xueqiu#share-source-code_content_11662_1145008
rank(x)             截面rank
delay(x,d)          d天之前的x值
correlation（x,y,d)  过去d天的x和y的时间序列相关系数
covariance(x,y,d)    过去d天的x和y的时间序列协方差
scale(x,a)            sum(abs(x))=a,通常a为1
delta(x,d)             x（t)-x(t-d)
signedpower(x,a)       x的a次幂
decay_linear(x,d)
"""
import os
import numpy as np
import pandas as pd
from util_world_alpha import *
import datetime

from collections import OrderedDict

from general.get_index_data import get_index_components
from general.trade_calendar import form_dt_index
from general.get_stock_data import GetPrice
from alpha_util.alpha_config import BACK_VIEW

def form_df(pth,fac,basic=None,filter_ipo=False):
    pth = os.path.join(pth,fac+'.csv')
    df = pd.read_csv(pth,index_col=0).T.stack().reset_index()
    df=df.rename(columns={'level_0':'dt','level_1':'code'})
    df.dt=df.dt.map(lambda x:str(pd.Timestamp(x))[:10])
    if filter_ipo is True:
        if basic is None:
            print ('Basic is None..')
        df = df.reset_index().join(basic[['ipo']], on='code')
        df.loc[:, 'days'] = df.dt.map(lambda x: pd.Timestamp(x)) - df.ipo.map(lambda x: pd.Timestamp(x))
        df = df[df.days >= datetime.timedelta(180)]
        df=df.drop(['ipo','days','index'],axis=1)
    df=df.set_index(['dt','code']).sort_index()
    df.columns = [fac]
    return df

class WORLD_ALPHA(object):
    def __init__(self,fac=None,risk=False,filter_ipo=False,index=None, start=None, end=None, freq=None):
        self.index = index
        self.start = start
        self.end = end
        self.freq=freq
        self.fac=fac
        self.risk=risk
        self.filter_ipo=filter_ipo
        # self.pth='D:\strategy\\utils_strategy\world_alpha\Factor_Library'
        self.pth='E:\SUNJINGJING\Strategy\\utils_strategy\world_alpha\Factor_Library'
        # self.dt_index=form_dt_index(self.start,self.end,self.freq)
        # self.term_stock_codes = self._get_idx_his_stock()
        # self.all_stocks = self.get_all_stock()
        self.basic=self.get_basic()
        self.BETA=form_df(self.pth,'BETA')
        self.BP=form_df(self.pth,'BP')
        self.mkt=form_df(self.pth,'mkt')
        self.info=self.get_info()
        self.data=getattr(self,fac)() if fac is not None else None

    def get_info(self):
        pth = os.path.join(self.pth,'info.csv')
        df = pd.read_csv(pth,index_col=0)
        df=pd.get_dummies(df,drop_first=True)
        self.info=df
        return df
    def get_basic(self):
        pth = os.path.join(self.pth,'basic.csv')
        df = pd.read_csv(pth,index_col=0)
        return df
    def _get_idx_his_stock(self):
        """
        获取所选指数self.codes每一个调仓期的历史成份股
        调用ut模块的函数
        @返回：
        ret: dict，{dt: stock_list}
        """
        print('get idx his stock')
        rets = []
        for dt in self.dt_index:
            ret = get_index_components(self.index, dt,self.del_three)
            ret = list(zip([dt] * len(ret), ret))
            rets.extend(ret)
        return pd.MultiIndex.from_tuples(rets, names=['dt', 'secu'])

    def get_all_stock(self):
        """
        获取所有调仓期涉及到的所有股票代码
        @返回：
        ret: list, 股票代码列表
        """
        print('get_all_stock')
        ret = list(set(self.term_stock_codes.levels[1]))
        return ret

    def get_all_data(self):
        """
        收益率数据: stock_term_return
        基准收益率数据： benchmark_term_return
        """
        print('get all data!')
        data_name_dic = OrderedDict([
            ('stock_data', self._get_stock_data),
            ('benchmark_data', self._get_benchmark_data)
        ])
        for func_name, func in data_name_dic.items():
            func()

    def _get_stock_data(self):
        """
        逐个调仓期,根据当期的股票代码（历史成份股记录）读取下期收益率
        收益率 = (下期股价-当期股价)/当期股价
        @返回：ret: dict, {dt: df}
        """
        print('get stock data!')
        if BACK_VIEW:
            df_pr=pd.read_csv(BACK_VIEW,dtype={'tick': object},encoding='utf8',index_col=['dt'],parse_dates=True).loc[self.start:self.end]
            df_pr['tick']=df_pr['tick'].map(lambda x : x.zfill(6)+'_SZ_EQ' if x[0] in ['0','3'] else x.zfill(6)+'_SH_EQ')
            df_pr2 = df_pr.loc[pd.DatetimeIndex(self.dt_index), :]
            df_pr2 = df_pr2.reset_index().set_index(['tick','dt']).sort_index()
            df_pr=df_pr.reset_index().set_index(['tick','dt']).sort_index()
            if self.freq not in ['d']:
                period_inc=df_pr2.groupby(level=0).apply(lambda x: x.reset_index(level=0,drop=True)['close'].pct_change())
                period_inc=period_inc.reset_index().set_index(['dt','tick']).sort_index()
                period_inc.columns=['period_inc']
            df_pr['inc'] = df_pr.groupby(level=0).apply(lambda x:x.reset_index(level=0,drop=True)['close'].pct_change())
            df_pr=df_pr.reset_index().set_index(['dt','tick']).sort_index()
        else:
            get_sql = GetPrice()
            get_sql.set_isIndex(False)
            df_pr = get_sql.get_in_spec_dates(self.all_stocks,self.dt_index,all=True)   # 获取高开低收,all 默认为False
        ## 调整格式
        self.all_data['stock_data'] = df_pr
        self.all_data['open']=df_pr.unstack()['open']
        self.all_data['close']=df_pr.unstack()['close']
        self.all_data['high']=df_pr.unstack()['high']
        self.all_data['low']=df_pr.unstack()['low']
        self.all_data['vol']=df_pr.unstack()['volume']
        self.all_data['inc'] = df_pr.unstack()['inc']
        if self.freq not in ['d']:
            self.all_data['period_inc']=period_inc
        else:
            self.all_data['period_inc']=df_pr.unstack()['inc']
        return df_pr

    def _get_benchmark_data(self):
        print('get benchmark data')
        get_sql = GetPrice()
        get_sql.set_isIndex(True)
        price = get_sql.get_in_spec_dates(self.index, self.dt_index)
        self.all_data['benchmark_data'] = price
        return price

    def get_all_alpha(self):
        '''
        运行并计算选定的alpha
        '''
        print('Get all alpha!')
        if not self.all_data:
            self.get_all_data()
        alpha_name=self.alpha_name
        for name in alpha_name:
            f=getattr(self,name)
            f()
    def INFO(self):
        "股票基本信息：行业"
        return form_df(self.pth,'INFO')
    def mkt(self):
        return form_df(self.pth,'mkt')
    def FCFP(self):
        "FCFP=自由现金流/总市值"
        return form_df(self.pth,'FCFP')
    def est_growth_low(self):
        ""
        data=form_df(self.pth,'est_growth_low',self.basic,filter_ipo=self.filter_ipo)
        return data
    def est_growth_date(self):
        ""
        df=form_df(self.pth,'est_growth_date')
        df.est_growth_date=df.est_growth_date.map(lambda x:str(pd.Timestamp(x))[:10] if x !='0' else x)
        return df

    def PE_FWD12M(self):
        "未来12月的预期市盈率"
        if self.risk is False:
            data = form_df(self.pth, 'PE_FWD12M')
            return data
        else:
            pth=os.path.join(self.pth,'risk_PE_FWD12M.csv')
            if os.path.exists(pth):
                ret=pd.read_csv(pth)
                ret.dt=ret.dt.map(lambda x:str(pd.Timestamp(x))[:10])
                ret=ret.set_index(['dt','code']).sort_index()
            else:
                data=form_df(self.pth, 'PE_FWD12M')
                data=data.reset_index().join(self.basic[['ipo']],on='code')
                data=data.query('dt>=ipo')
                del data['ipo']
                data=data.join(self.info,on='code').set_index(['dt','code']).sort_index()
                df=pd.concat([data,self.BP,self.BETA,self.mkt],axis=1)
                ret=df.groupby(level=0).apply(lambda df:risk_rejust(df,'PE_FWD12M'))
                ret.to_csv(pth)
            return ret

    def CONSENSUS(self):
        "综合评级"
        if self.risk is False:
            data = form_df(self.pth, 'CONSENSUS')
            return data
        else:
            pth=os.path.join(self.pth,'risk_CONSENSUS.csv')
            if os.path.exists(pth):
                ret=pd.read_csv(pth)
                ret.dt = ret.dt.map(lambda x: str(pd.Timestamp(x))[:10])
                ret = ret.set_index(['dt', 'code']).sort_index()
                return ret
            else:
                data=form_df(self.pth, 'CONSENSUS')
                data=data.reset_index().join(self.basic[['ipo']],on='code')
                data=data.query('dt>=ipo')
                del data['ipo']
                data=data.reset_index().join(self.info,on='code').set_index(['dt','code']).sort_index()
                df=pd.concat([data,self.BP,self.BETA,self.mkt],axis=1)
                ret=df.groupby(level=0).apply(lambda df:risk_rejust(df,'CONSENSUS'))
                ret.to_csv(pth)
                return ret

    def LSF(self):
        "流动性冲击因子（股票当前非流动性水平相对于其历史平均水平的变化率）"
        return form_df(self.pth,'LSF')

    def RCVB(self):
        "rank correlation of value and beta"
        return form_df(self.pth,'RCVB')

    def alpha101(self):
        '''
        (close-open)/(high-low+0.001)
        '''
        print('Cal alpha 101!')
        ret=(self.all_data['close']-self.all_data['open'])/(self.all_data['high']-self.all_data['low']+0.001)
        self.alpha['alpha101']=ret
        return ret

    def alpha26(self):
        '''

        '''
        print('Cal alpha 26!')
        if 'stock_data' not in self.all_data:
            self.get_all_data()
        ret = pd.corr(method='spearman')
        self.alpha['alpha53'] = ret.diff(9)
        return ret

    def alpha12(self):
        '''
        sign(delta(volume, 1)) * (-1 * delta(close, 1))
        当x>0，sign(x)=1; 当x=0，sign(x)=0; 当x<0， sign(x)=-1；
        '''
        print('Cal alpha 12!')
        if 'stock_data' not in self.all_data:
            self.get_all_data()
        ret=np.sign(self.all_data['vol'].diff())*(1)*(self.all_data['close'].diff())
        # ret=ret.dropna(how='all')
        ret = ret.loc[pd.DatetimeIndex(self.dt_index), :]
        self.alpha['alpha12']=ret
        return ret

    def alpha53(self):
        '''
        (2∗close−low−high)/(close−low) 与9日之前的差
        '''
        print('Cal alpha 53!')
        if 'stock_data' not in self.all_data:
            self.get_all_data()
        ret = (2*self.all_data['close']-self.all_data['low']-self.all_data['high'])/(self.all_data['close']-self.all_data['low']+0.001)
        ret=ret.diff(9)
        ret=ret.loc[pd.DatetimeIndex(self.dt_index),:]
        self.alpha['alpha53'] = ret
        return ret

if __name__ == '__main__':
    pass
