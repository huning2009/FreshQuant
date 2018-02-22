# -*- coding: utf-8 -*-

"""
@Time    : 2018/1/19 11:15
@Author  : jessica.sun
"""

class EventData(object):
    def __init__(self, name, return_analysis=None, price_in=None, summary=None):
        self.name = name
        self.return_analysis = return_analysis
        self.price_in = price_in
        self.summary = summary


class ReturnAnalysis(object):
    def __init__(self, factor_name=None, return_stats=None, benchmark_return=None,
                 group_mean_return=None, group_cum_return=None):
        self.name = factor_name
        self.return_stats = return_stats
        self.benchmark_return = benchmark_return
        self.group_mean_return = group_mean_return
        self.group_cum_return = group_cum_return


class ICAnalysis(object):
    def __init__(self, name=None, IC_series=None, IC_statistics=None,
                 groupIC=None, IC_decay=None):
        self.name = name
        self.IC_series = IC_series
        self.IC_statistics = IC_statistics
        self.groupIC = groupIC
        self.IC_decay = IC_decay


class TurnOverAnalysis(object):
    def __init__(self, name=None, buy_signal=None, auto_correlation=None,
                 turnover=None):
        self.name = name
        self.buy_signal = buy_signal
        self.auto_correlation = auto_correlation
        self.turnover = turnover


class CodeAnalysis(object):
    def __init__(self, name=None, industry_analysis=None, cap_analysis=None, stock_list=None):
        self.name = name
        self.industry_analysis = industry_analysis
        self.cap_analysis = cap_analysis
        self.stock_list = stock_list


class IndustryAnalysis(object):
    def __init__(self, name=None, gp_mean_per=None, gp_industry_percent=None):
        self.name = name
        self.gp_mean_per = gp_mean_per
        self.gp_industry_percent = gp_industry_percent

