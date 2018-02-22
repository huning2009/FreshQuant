# coding:utf-8
__author__ = 'jessica.sun'

from freshquant.event.world_event import World_Event
from freshquant.event.util_event import *
from sqlalchemy import create_engine
import pickle
import os

class event_analysis2(object):
    def __init__(self,N1,N2,stock,event=None,data=None,pr=None,start=None,both=True,inx='000300'):
        self.N1=N1
        self.N2=N2
        self.stock=stock
        self.start=start
        self.both=both
        self.inx=inx
        self.data=World_Event(start_date=start,event=event).data if data is None else data
        self.pr=pr
        self.ret_analysis=self.ret_analysis()
        # self.price_in=self.price_in()
        # self.summary=self.summary()
    def ret_analysis(self):
        """
         df,columns=['dt','secu]
         'dt'为字符串
         'secu':有后缀
         """
        df=self.data.sort_values(by='dt',ascending=True)
        df=df.drop_duplicates(['dt','typ'])
        start = df.dt.min()
        end = df.dt.max()
        # 将事件分配到最近的交易日
        df.loc[:, 'dt'] = align_to_trading_days(df[['dt']])
        # 股票行情数据用后复权
        # # 指数行情数据
        # sql_inx = """SELECT dt,tick,close,inc,vol From hq_index where  dt >= '%s' and dt <= '%s' and tick = '%s' ORDER BY dt ASC """ % (
        # __start, __end, self.inx)
        # pr_inx = pd.read_sql_query(sql_inx, engine)
        # pr_inx.dt = pr_inx.dt.map(str)
        # 去停牌
        # df = pd.merge(df, pr, left_on=['dt', 'typ'], right_on=['dt', 'tick'], how='inner').dropna()
        # df = df[df.vol > 0][['dt', 'secu']]
        # 指数和股票：事件发生前后N天的收益率序列
        if self.both:
            # lst_ret_inx = [get_before_and_after_n_days(row, self.N1, pr_inx, index=True) for n,row in df.iterrows()]
            lst_ret = [get_before_and_after_n_days(row, self.N1, self.pr, index=False) for n,row in df.iterrows()]
        else:
            # lst_ret_inx = [get_after_n_days(row, self.N2, pr_inx, index=True) for n,row in df.iterrows()]
            lst_ret = [get_after_n_days(row, self.N2, self.pr, index=False) for n,row in df.iterrows()]
        # ret_inx = pd.concat(lst_ret_inx) / 100
        ret = pd.concat(lst_ret) / 100
        cum_ret = (ret + 1).cumprod(axis=1) - 1
        f_name=os.path.join(os.getcwd(),'temp',self.stock+'.pkl')
        cum_ret.to_pickle(f_name)
        # # exess_ret = ret - ret_inx
        # # exess_cum_ret = (exess_ret + 1).cumprod(axis=1) - 1
        # prob=cum_ret.apply(lambda x: (x > 0).sum() / len(cum_ret))
        # # prob_exess=exess_cum_ret.apply(lambda x: (x > 0).sum() / len(exess_cum_ret))
        # df=pd.concat([pd.DataFrame(cum_ret.mean(),columns=['ret']),
        #               pd.DataFrame(prob,columns=['prob'])],axis=1)
        return df
    def summary(self):
        "统计各个月份的事件数量"
        df=self.data
        df.dt=df.dt.apply(pd.Timestamp)
        df=df.drop_duplicates(['dt','secu'])
        df=df.set_index('dt')
        ret={str(y)+str(m).zfill(2):[data.count()] for (y,m),data in df.secu.groupby([df.index.year,df.index.month])}
        ret=pd.DataFrame(ret).T
        ret.columns=['num']
        return ret

    def price_in(self):
        return

class event_analysis(object):
    def __init__(self,N1,N2,event=None,data=None,start=None,both=True,inx='000300'):
        self.N1=N1
        self.N2=N2
        self.start=start
        self.both=both
        self.inx=inx
        self.data=World_Event(start_date=start,event=event).data if data is None else data
        self.ret_analysis=self.ret_analysis()
        # self.price_in=self.price_in()
        # self.summary=self.summary()
    def ret_analysis(self):
        """
         df,columns=['dt','secu]
         'dt'为字符串
         'secu':有后缀
         """
        df=self.data.sort_values(by='dt',ascending=True)
        df=df.drop_duplicates(['dt','secu'])
        start = df.dt.min()
        end = df.dt.max()
        # 将事件分配到最近的交易日
        df.loc[:, 'dt'] = align_to_trading_days(df[['dt']])
        # 去掉股票代码后缀
        all_stocks = [str(x)[:6] for x in df.secu.unique()]
        # 股票行情数据用后复权
        __start = str((pd.Timestamp(start) - datetime.timedelta(2 * self.N1)).date())
        __end = str((pd.Timestamp(end)+ datetime.timedelta(2 * self.N2)).date())
        engine = create_engine('mysql+mysqlconnector://pd_team:pd_team123@!@192.168.250.200/ada-fd')
        if len(all_stocks) >1:
            sql_secu = """SELECT dt,tick,close,inc,vol From hq_price_after where  dt >= '%s' and dt <= '%s' and tick in %s ORDER BY dt ASC """ \
                       % (__start, __end, tuple(all_stocks))
        else:
            sql_secu = """SELECT dt,tick,close,inc,vol From hq_price_after where  dt >= '%s' and dt <= '%s' and tick = %s ORDER BY dt ASC """ \
                       % (__start, __end, all_stocks[0])
        pr = pd.read_sql_query(sql_secu, engine)
        pr.dt = pr.dt.map(str)
        pr.tick = pr.tick.map(lambda x: str(x)[:6] + '_SH_EQ' if str(x)[0] is '6' else str(x)[:6] + '_SZ_EQ')
        df.secu=df.secu.map(lambda x: str(x)[:6] + '_SH_EQ' if str(x)[0] is '6' else str(x)[:6] + '_SZ_EQ')
        # 指数行情数据
        sql_inx = """SELECT dt,tick,close,inc,vol From hq_index where  dt >= '%s' and dt <= '%s' and tick = '%s' ORDER BY dt ASC """ % (
        __start, __end, self.inx)
        pr_inx = pd.read_sql_query(sql_inx, engine)
        pr_inx.dt = pr_inx.dt.map(str)
        # 去停牌
        df = pd.merge(df, pr, left_on=['dt', 'secu'], right_on=['dt', 'tick'], how='inner').dropna()
        df = df[df.vol > 0][['dt', 'secu']]
        # 指数和股票：事件发生前后N天的收益率序列
        if self.both:
            lst_ret_inx = [get_before_and_after_n_days(row, self.N1, pr_inx, index=True) for n,row in df.iterrows()]
            lst_ret = [get_before_and_after_n_days(row, self.N1, pr, index=False) for n,row in df.iterrows()]
        else:
            lst_ret_inx = [get_after_n_days(row, self.N2, pr_inx, index=True) for n,row in df.iterrows()]
            lst_ret = [get_after_n_days(row, self.N2, pr, index=False) for n,row in df.iterrows()]
        ret_inx = pd.concat(lst_ret_inx) / 100
        ret = pd.concat(lst_ret) / 100
        exess_ret = ret - ret_inx
        cum_ret = (ret + 1).cumprod(axis=1) - 1
        exess_cum_ret = (exess_ret + 1).cumprod(axis=1) - 1
        prob=cum_ret.apply(lambda x: (x > 0).sum() / len(cum_ret))
        prob_exess=exess_cum_ret.apply(lambda x: (x > 0).sum() / len(exess_cum_ret))
        df=pd.concat([pd.DataFrame(cum_ret.mean(),columns=['ret']),pd.DataFrame(exess_cum_ret.mean(),columns=['exess_ret']),
                      pd.DataFrame(prob,columns=['prob']),pd.DataFrame(prob_exess,columns=['prob_exess'])],axis=1)
        return df
    def summary(self):
        "统计各个月份的事件数量"
        df=self.data
        df.dt=df.dt.apply(pd.Timestamp)
        df=df.drop_duplicates(['dt','secu'])
        df=df.set_index('dt')
        ret={str(y)+str(m).zfill(2):[data.count()] for (y,m),data in df.secu.groupby([df.index.year,df.index.month])}
        ret=pd.DataFrame(ret).T
        ret.columns=['num']
        return ret

    def price_in(self):
        return

# if __name__=='__main__':
#     ins = event_analysis(event='zengchi',N1=30,N2=30,start='2017-01-01',both=True,inx='000001')
#     print (ins.summary)
#     # import matplotlib.pyplot as plt
#     # plt.figure()
#     # ret.ret_analysis[0].mean().plot()
#     # ret.ret_analysis[1].mean().plot()
#     # plt.show()
#     print ('Done!')
#     print('Done!')
#     print('Done!')

