# -*- coding: utf-8 -*-

"""
@Time    : 2018/1/18 15:31
@Author  : jessica.sun

事件驱动：
传统事件驱动：高管增持、定增破发、股权激励、业绩超预期和业绩预增
网络文本挖掘事件驱动：公告披露，新闻热度变化及股吧情绪异常
一致预期事件驱动：实际指标偏离预测指标事件、一致预期截面优势、一致预期变化异常
"""
import numpy as np
import pandas as pd
import os
import glob
from functools import reduce
import datetime
from sqlalchemy import create_engine
from utils_strategy.general.mongo_data import general_form_mongo_to_df
from utils_strategy.general.get_stock_data import get_all_stocks_basic
from utils_strategy.general.trade_calendar import get_nearest_trade_day,get_next_n_trade_day

pd.set_option("display.max_rows", 20)

class World_Event(object):
    """
    每个事件的最后结果必须有dt和secu（有后缀）字段，注意columns名称。
    df: dt--事件发生日
        secu--股票代码（有后缀）
        rpt -- 对应报告期（2016-12-31）
    输出的三种格式dataframe
    1. 基本数据
    2. 合并格式
    3. 最近的数据:即enddate
    """
    def __init__(self,event=None,typ=None,start_date=None, end_date=None):
        self.event = event
        self.typ=typ
        self.pth='E:\SUNJINGJING\Strategy\EVENT\\事件库'
        self.start_date = start_date if start_date is not None else '2010-01-01'
        self.end_date = end_date if end_date is not None else '2017-12-31'
        self.host_200='122.144.134.200'
        self.host_4 = '122.144.134.4'
        self.engine_21=create_engine('mysql+mysqlconnector://pd_team:pd_team321@!@122.144.134.21/ada-fd')
        self.stock_name_dict=self.get_name_dict()
        self.data=getattr(self,event)()

    def get_name_dict(self):
        stock_basic=get_all_stocks_basic()
        dic=dict(stock_basic[['secu','szh']].to_dict(orient='split')['data'])
        return dic

    def off_cap(_dt=None):
        dt = datetime.datetime.strftime(datetime.date.today(), '%Y-%m-%d') if _dt is None else _dt
        # 获取证券基本信息base_stock
        df_base = general_form_mongo_to_df('122.144.134.4', 'ada', 'base_stock',
                                           pos=['code', 'abbr.szh', 'ls.code', 'ls.dt', 'ls.edt'],
                                           filters={'mkt.code': {'$in': ['1001', '1002', '1003', '1012']},
                                                    'ls.edt': None})
        # 判断是否为摘帽概念
        temp_dt = str(datetime.datetime.strptime(dt, '%Y-%m-%d') - relativedelta(years=1))[:10]
        engine = create_engine('mysql+mysqlconnector://pd_team:pd_team321@!@122.144.134.21/ada-fd')
        sql_cap = """select dt,tick from hq_stock_trade where zqjb = 'P' and dt >= '%s' """ % (temp_dt)
        df_cap = pd.read_sql(sql_cap, engine)
        df_cap = df_cap[df_cap.tick.map(lambda x: x[0] in ['0', '3', '6'])]
        df_cap.tick = df_cap.tick.map(lambda x: x + '_SH_EQ' if x[0] is '6' else x + '_SZ_EQ')
        df_cap['摘帽'] = '摘帽'
        df_cap = df_cap.rename(columns={'dt': 'cap_dt', 'tick': 'code'})
        df = pd.merge(df_base, df_cap, on='code', how='outer')
        return df

    def zengchi(self):
        """数据来源于wind，一次提取半年数据"""
        if os.path.exists(os.path.join(self.pth,'zengchi.xlsx')):
            df=pd.read_excel(os.path.join(self.pth,'zengchi.xlsx'))
        else:
            lst=glob.glob('E:\SUNJINGJING\Strategy\EVENT\报告\__增持\*.xlsx')
            df=pd.concat([pd.read_excel(f,skiprows=[-1,-2]) for f in lst])
            df.to_excel(os.path.join(self.pth,'zengchi.xlsx'))
        df=df.rename(columns={'代码':'secu','公告日期':'dt','市值':'mkt','方向':'direction',
                              '变动起始日期':'dt0','变动截止日期':'dt1','变动数量占流通股比(%)':'rate'})

        df['lasting']=df.dt1-df.dt0
        df['timing']=df.dt-df.dt1
        df.dt=df.dt.map(lambda x:str(pd.Timestamp(x))[:10])
        if self.start_date is not None:
            df=df[df.dt>=self.start_date]
        return df
    def performance(self):
        if os.path.exists(os.path.join(self.pth,'all_performance.hd5')):
            df=pd.read_hdf(os.path.join(self.pth,'all_performance.hd5'))
        else:
            pth=os.path.join(self.pth,'performance.xlsx')
            xls=pd.ExcelFile(pth)
            lst=[]
            for sheet_name in xls.sheet_names:
                temp=pd.read_excel(xls,sheet_name,skiprows=[0,1,-1])
                lst.append(temp)
            df=pd.concat(lst)
            df=df.dropna(how='all')
            df.to_hdf(os.path.join(self.pth,'all_performance.hd5'),'df')
        df=df.drop_duplicates()
        df=df.rename(columns={'代码':'secu','预告日期':'dt','报告期':'rpt','预警类型':'typ'})
        df.dt=df.dt.map(lambda x: str(x)[:10])
        return df
    def jiejin(self):
        pth = os.path.join(self.pth, 'jiejin.xlsx')
        df=pd.read_excel(pth,skiprows=[0,1,-1])
        return df
    def employee_ownership(self):
        "员工持股"
        pth = os.path.join(self.pth, 'employee_ownership_wind.xlsx')
        df=pd.read_excel(pth)
        df = df.rename(columns={'证券代码': 'secu', '董事会预案日': 'dt'})
        df.dt = df.dt.map(lambda x: str(pd.Timestamp(x))[:10])
        # df=pd.read_excel(pth,skiprows=[0,1,-1])
        # df=df.rename(columns={'代码':'secu','首次公告日':'dt'})
        return df
    def equity_incentives(self):
        "股权激励"
        pth = os.path.join(self.pth, 'equity_incentives.xlsx')
        df=pd.read_excel(pth,skiprows=[0,-1],sheetname='股权激励实施明细')
        df = df.rename(columns={'代码': 'secu', '最新公告日期': 'dt'})
        df.dt=df.dt.map(lambda x:str(x)[:10])
        return df
    def Stocks_Heavily_Held(self):
        pth = os.path.join(self.pth, 'equity_incentives.xlsx')
        df=pd.read_excel(pth,skiprows=[0,-1],sheetname='股权激励实施明细')
        df = df.rename(columns={'代码': 'secu', '最新公告日期': 'dt'})
        df.dt=df.dt.map(lambda x:str(x)[:10])
        return df
    def Add_Issuance(self):
        pth = os.path.join(self.pth, 'Add_Issuance.xlsx')
        df1=pd.read_excel(pth,skiprows=[0,1,-1],sheetname='增发预案')
        df2 = pd.read_excel(pth, skiprows=[0,1,-1],sheetname='增发实施')
        df3 = pd.read_excel(pth, skiprows=[0,-1],sheetname='定向增发发行资料')
        return
    def announcements(self):
        dt1 = self.start_date
        dt2 = self.end_date
        df = general_form_mongo_to_df(host='122.144.134.95', db_name='news', tb_name='announcement',
                                      pos=['title', 'secu.cd', 'pdt'],
                                      filters={'secu.cd': {'$regex': '_S'},
                                               "typ":self.typ,
                                               'pdt': {"$gte": datetime.datetime.strptime(dt1, '%Y-%m-%d'),
                                                       "$lte": datetime.datetime.strptime(dt2, '%Y-%m-%d')}})
        return df
    def all_announcements(self):
        pth = os.path.join(self.pth, 'all_announcements.hd5')
        if os.path.exists(pth):
            df=pd.read_hdf(pth)
        else:
            dt1=self.start_date
            dt2=self.end_date
            df = general_form_mongo_to_df(host='122.144.134.95', db_name='news', tb_name='announcement',
                                          pos=['title','secu.cd', 'pdt'],
                                          filters={'secu.cd': {'$regex': '_S'},
                                                   'pdt': {"$gte": datetime.datetime.strptime(dt1, '%Y-%m-%d'),
                                                           "$lte": datetime.datetime.strptime(dt2, '%Y-%m-%d')}})
            df.to_hdf(pth,'df')
        df=df.rename(columns={'secu.cd':'secu','pdt':'dt'})
        df.secu = df.secu.map(lambda x: [s for s in x if s[0] in ['0', '3', '6']])
        df=df[df.secu.map(lambda x:len(x)>=1)]
        df.secu = df.secu.map(lambda x: x[0])
        return df
    def earlier_annual_report(self):
        dt1=self.start_date
        dt2=self.end_date
        df = general_form_mongo_to_df(host='122.144.134.95', db_name='news', tb_name='announcement',
                                      pos=['title', 'secu.cd', 'pdt'],
                                      filters={'typ': '100101',
                                               'secu.cd': {'$regex': '_S'},
                                               'pdt': {"$gte": datetime.datetime.strptime(dt1, '%Y-%m-%d'),
                                                       "$lte": datetime.datetime.strptime(dt2, '%Y-%m-%d')}})
        # 剔除已取消的报告
        df=df[df.title.map(lambda x:'已取消' not in x)]
        df['rpt_year'] = df.title.str.extract('(\d{4})', expand=False)
        df = df.set_index('rpt_year').groupby(level=0).apply(
            lambda x: x.sort_values(by='pdt').head(int(len(x) * 0.1))).reset_index(drop=True)
        df.pdt = df.pdt.map(lambda x: str(x.date()) if x.hour >= 15 else get_nearest_trade_day(x))
        df = df.rename(columns={'secu.cd': 'secu', 'pdt': 'dt'})
        df['year'] = df['dt'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').year)
        df['secu'] = df['secu'].map(
            lambda x: [secu for secu in x if secu[0] in ['0', '3', '6'] and secu[-6:] in ['_SH_EQ', '_SZ_EQ']]) \
            .map(lambda x: x[0] if len(x) >= 1 else np.nan)
        df = df.dropna(how='any')
        df = df.set_index('year')
        return df
    def profit_distribution(self):
        dt1=self.start_date
        dt2=self.end_date
        sql ="""SELECT secu,plan From equity_bonus_jc where plan >= '%s' and plan <= '%s' and givsr>1"""%(dt1,dt2)
        engine = create_engine('mysql+mysqlconnector://pd_team:pd_team123@!@192.168.250.200/ada-fd')
        df = pd.read_sql_query(sql, engine)
        df['year']=df['plan'].map(lambda x:x.year)
        df=df.rename(columns={'plan':'dt'})
        df=df.set_index('year')
        return
    def base_earnings_preannouncement(self):
        """ 业绩预告：base_earnings_preannouncement """
        df=general_form_mongo_to_df(self.host_95,'ada','base_earnings_preannouncement',
                                    pos=['secu','y','rpt','content','typ','reason'],
                                    filters={'y':{'$gte':self.start_date,'$lte':self.end_date}})
        df['content'] = df['content'].map(lambda x: str(x).split('净利润')[1])
        df=df.rename(columns={'y':'dt','typ':'业绩预告类型','content':'业绩预告内容','rpt':'业绩预告报告期','reason':'原因'})
        df=df.loc[:,['dt','secu','业绩预告类型','业绩预告报告期','业绩预告内容','原因']]
        df3=df[df['dt']==self.end_date][['secu','业绩预告类型','业绩预告内容']]
        return df,df,df3

    def cal_expected(self):
        """归属母公司净利润is_tpl_30
        一致预期归属母公司净利润con_forecast_stk:c4"""
        # 1. 获取业绩预告中披露的净利润预计值
        ins = World_Event(self.start_date, self.end_date)
        df_actual = ins.base_earnings_preannouncement()[0]
        df_actual = df_actual[df_actual['业绩预告报告期'].map(lambda x: x.split('-')[1]) == '12']
        df_actual = df_actual.sort_values(by='dt', ascending=True).drop_duplicates(['业绩预告报告期', 'secu'], keep='last')
        df_actual['sign'] = df_actual['业绩预告内容'].str.extract("(亏损)", expand=False)
        df_actual.sign = df_actual.sign.map(lambda x: -1.0 if x == '亏损' else 1.0)
        temp = df_actual['业绩预告内容'].str.extractall("(\d+[.]?\d+)[万|%]").astype(np.float).rename(columns={0: 'value'}) \
            .reset_index().pivot(index='level_0', columns='match', values='value')
        temp['avg_profit'] = temp.iloc[:, :2].mean(axis=1)
        df_combined = pd.merge(df_actual.reset_index(), temp.reset_index()[['level_0', 'avg_profit']], left_on='index',
                               right_on='level_0', how='outer')
        df_combined = df_combined.drop(['level_0', 'index'], axis=1)
        df_combined['avg_profit'] = df_combined.avg_profit * df_combined.sign
        df_combined['bef'] = df_combined.dt.map(lambda x: get_next_n_trade_day(x))  # 预告公布时间匹配为最新交易日
        # df_combined['current_bar']=df_combined.current_bar.map(lambda x: get_nearest_trade_day(x))#预告公布时间匹配为最新交易日
        # df_combined['next_bar'] = df_combined.next_bar.map(lambda x: get_nearest_trade_day(x))  # 预告公布时间匹配为最新交易日
        print('1')
        # 2.查询预告披露日的一致预期净利润
        filters2 = {'$or': [{'stockcode': row[1].secu[:6], 'tdate': int(row[1].bef.replace('-', '')),
                             'rptdate': int(row[1]['业绩预告报告期'][:4])}
                            for row in df_combined.iterrows()]}
        fore_profit = general_form_mongo_to_df(host='122.144.134.4', db_name='forecast', tb_name='con_forecast_stk',
                                               pos=['stockcode', 'rptdate', 'c4'], filters=filters2)
        fore_profit = fore_profit.rename(columns={'stockcode': 'secu', 'rptdate': '业绩预告报告期'})
        fore_profit['业绩预告报告期'] = fore_profit['业绩预告报告期'].map(lambda x: str(x) + '-12-31')
        fore_profit.secu = fore_profit.secu.map(lambda x: x + '_SH_EQ' if x[0] in ['6'] else x + '_SZ_EQ')
        print('2')
        # 5.将业绩预告公布的和一致预期结合
        df = pd.merge(df_combined, fore_profit, on=['secu', '业绩预告报告期'])
        df['excess_profit'] = (df['avg_profit'] - df['c4']) / df['c4'].abs()
        return df,df,df

    def pre_disclosure_time(self):
        """ 年度报告预披露：pre_disclosure_time """
        df = general_form_mongo_to_df(self.host_95, 'ada', 'pre_disclosure_time',
                                      pos=['secu', 'y', 'order','change1','change2','change3'],
                                      filters={'order': {'$gte': self.start_date, '$lte': self.end_date}})
        for col in ['change1','change2','change3']:
            df.update(pd.DataFrame(df[col].values,columns=['order']))
        df=df[df.secu.map(lambda x:x[0] in ['0','3','6'])]
        df=df.drop(['change1','change2','change3'],axis=1)
        df = df.rename(columns={'order': '年报预约披露', 'y': '披露年报报告期'})
        df2=df.copy()
        df2['dt'] = df2['年报预约披露']
        df2['年报预约披露']='预约披露年报'
        # 获取最近一期年报的报告期
        rpt=self.end_date[:4]+'-12-31' if int(self.end_date[5:7])>4 else str(int(self.end_date[:4])-1)+'-12-31'
        df3=df2[df2['披露年报报告期']==rpt][['secu','dt']].rename(columns={'dt':'年报披露日期'})
        return df,df2,df3

    def cmb_report_score_adjust(self):
        """报告评级调整表:数据有待考证"""
        df1 = general_form_mongo_to_df(self.host_4, 'forecast', 'cmb_report_score_adjust',
                                      pos=['stockcode', 'ccd','pcd', 'csi', 'psi','saf'],
                                      filters={'ccd': {'$gte': self.start_date, '$lte': self.end_date}})
        if len(df1)==0:
            return df1,df1,df1
        else:
            df1=df1[df1['stockcode'].map(lambda x:len(x)==6 and x[0] in ['0','3','6'])]
            df1.ccd=df1.ccd.map(lambda x:x[:10])
            temp_dic={1:'评级未调',2:'评级上调',3:'评级下调',4:'评级未知'}
            df1.saf=df1.saf.map(lambda x:temp_dic[x])
            df1=df1.rename(columns={'ccd':'dt','stockcode':'secu','saf':'评级调整标志','pcd':'上次预测日期','psi':'上次评级','csi':'本次评级'})
            df1 = df1[df1.secu.map(lambda x: x[0] in ['0', '3', '6'])]
            df1.secu=df1.secu.map(lambda x:x+'_SH_EQ' if x[0] is '6' else x+'_SZ_EQ')
            df2=df1[['dt','secu','评级调整标志']]
            df3=df2[df2['dt']==self.end_date][['secu','评级调整标志']]
            return df1,df2,df3

    def base_executive_regulation(self):
        """高管增减持"""
        df1 = general_form_mongo_to_df(self.host_200, 'ada', 'base_executive_regulation',
                                      pos=['secu', 'name.szh', 'cd', 'rd','cirrat','change','after','cause'],
                                      filters={'rd': {'$gte': self.start_date,'$lte': self.end_date}})
        df1=df1.rename(columns={'name.szh':'股份变动人姓名','cd':'变动日期','rd':'填报日期','cirrat':'占流通股本比例',
                                'change':'变动股数','after':'变动后持股','cause':'变动原因'})
        df1 = df1[df1.secu.map(lambda x: x[0] in ['0', '3', '6'])]
        df1['增减持方向'] = df1['变动股数'].map(lambda x: '高管增持' if float(x) > 0 else '高管减持')

        df2 = df1[['secu','填报日期','变动股数','增减持方向','占流通股本比例']]
        df2 = df2.rename(columns={'填报日期':'dt'})
        df3=df2[df2['dt']==self.end_date][['secu','增减持方向','变动股数','占流通股本比例']]
        return df1,df2,df3

    def equity_bonus_jc(self):
        """送转"""
        sql="""select * from equity_bonus_jc where (exrdt >= '%s' and exrdt <= '%s') or (plan >= '%s' and plan <= '%s')"""\
            %(self.start_date,self.end_date,self.start_date,self.end_date)
        df1=pd.read_sql(sql,self.engine_21).drop(['id','upt','sid','aft_bns'],axis=1)
        df1 = df1[df1.secu.map(lambda x: x[0] in ['0', '3', '6'])]
        df1.plan=df1.plan.map(str)
        df1.exrdt=df1.exrdt.map(str)
        df1=df1.rename(columns={'exrdt':'除权日期','givsr':'送转比例','givsr_stock':'送股比例','givsr_transf':'转增比例',
                                'bns':'含税派息','y':'分红年份','plan':'预案日期','cancel':'是否终止'})
        df2=df1[['secu','除权日期','送转比例','送股比例','含税派息','分红年份','转增比例','预案日期','是否终止']]
        df2['dt']=df2['预案日期']
        df2.update(pd.DataFrame(df2['除权日期'].values, columns=['dt']))
        df3=df2[(df2['除权日期']==self.end_date)|(df2['预案日期']==self.end_date)][['secu','送转比例','除权日期','预案日期','含税派息']]
        return df1,df2,df3

    def get_xlsx_data(self):
        """
        获取事件汇总的excel
        """
        dict_df = {e_name:getattr(self, e_name)()[0] for e_name in events}
        writer=pd.ExcelWriter('event_report.xlsx',engine='xlsxwriter',options={'strings_to_numbers':True})
        for e_name,df in dict_df.items():
            if len(df) !=0:
                df['szh']=df.secu.map(self.stock_name_dict)
            df.to_excel(writer, sheet_name=e_name, startcol=0, startrow=0)
        writer.save()

    def get_combined_data(self):
        """
        获取所有需要的事件数据并根据secu和dt合并
        """
        rets = [getattr(self,e_name)()[1] for e_name in events]
        df=reduce(lambda x,y:pd.merge(x,y,on=['secu','dt'],how='outer'),rets) if len(rets)>1 else rets[0]
        df=df.set_index(['dt','secu']).sort_index()
        self.combined_data=df
        return df

    def get_latest_data(self):
        df_base=get_all_stocks_basic()
        rets = [getattr(self,e_name)()[2] for e_name in events]
        df=reduce(lambda x,y:pd.merge(x,y,on=['secu'],how='outer'),rets) if len(rets)>1 else rets[0]
        df_sum=pd.merge(df_base,df,on='secu',how='inner')
        self.latest_data=df_sum
        return df_sum

# if __name__ == '__main__':
#     events=pd.read_csv('events.csv',encoding='gbk')['event'].tolist()
#     start_date = "2016-01-01"
#     end_date = "2016-12-31"
#     ins = World_Event(start_date, end_date,events=events)
#     df=ins.base_executive_regulation()
#     ins.get_xlsx_data()
#     print('Done!')