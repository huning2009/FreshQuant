#!/usr/bin/env python
# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats


# ###################################################################
# IC
# ###################################################################


def _plot_IC_bar(IC, ax=None):
    if ax is None:
        ax = plt.gca()
    IC.plot(kind='bar', ax=ax)
    return ax


def _plot_IC_series(IC, window_size=20, ax=None):
    if ax is None:
        ax = plt.gca()
    xticklabels = ax.get_xticklabels()
    rolling_mean = pd.rolling_mean(IC, window_size)
    rolling_mean.plot(ax=ax)
    return ax


def plot_IC_bar_and_series(IC,window_size=20,ax=None):
    if ax is None:
        ax = plt.gca()
    ax = _plot_IC_series(IC, window_size, ax)
    ax = _plot_IC_bar(IC,ax)
    xticklabels = IC.index.values.copy()
    N = len(IC)
    ax.set_xticklabels([''] * N)
    if N > 10:
        step = int(N / 10)
        xticklabels[np.arange(N)%step!=0] = ''
    ax.set_xticklabels(xticklabels)
    ax.set_title('IC')
    return ax



def plot_IC_decay(ic_decay, ax=None):
    if ax is None:
        ax = plt.gca()
    ic_decay.plot(kind='bar', ax=ax, title='IC Decay')
    return ax


def plot_IC_distribution(ic, ax=None, bins=10):
    return plot_distribution(ic, ax, bins=bins)


# ###################################################################
# ret
# ###################################################################
def plot_returns(df):
    """

    parameters: ret_dict, shoud has following keys:
    benchmark, Q1,Q2,...,Q5,value of each key is a TimeSeires

    """
    df = pd.DataFrame(ret_dict)
    df.plot()


def plot_returns_distribution(returns, ax=None, bins=10):
    return plot_distribution(returns, ax, bins=bins)


# ###################################################################
# Turn over
# ###################################################################





# ###################################################################
# 选股结果
# ###################################################################


def plot_insdustry_percent(df, ax=None):
    if ax is None:
        ax = plt.gca()
    # 取得各个style下面的color
    colors_ = _get_colors()
    ax = df.plot(kind='bar',stacked=True, color = colors_,ax=ax, width=1,alpha=0.6)
    ax, font = _show_chinese_character(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=font)
    ax.set_ylim([0,1.0])
    return ax

def plot_industry_mean_percent(df,ax=None):
    colors_ = _get_colors()
    ax = df.plot(kind='bar',ax=ax, color = colors_)
    ax, _ = _show_chinese_character(ax)
    return ax


def _get_colors():
    colors_ = []
    for sty in plt.style.available:
        plt.style.use(sty)
        sty_colors = [ item['color'] for item in  list(plt.rcParams['axes.prop_cycle'])]
        colors_.extend(sty_colors)
    colors_ = list(set(colors_))
    colors_ = [c for c in colors_ if len(c) > 4]
    return colors_


def _show_chinese_character(ax):
    import os
    from matplotlib.font_manager import FontProperties
    if os.name == 'posix':
        fname = r'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    elif os.name == 'nt':
        fname = r"c:\windows\fonts\simsun.ttc"
    font = FontProperties(fname=fname, size=10)
    
    labels = ax.get_xticklabels()+ax.legend().texts+[ax.title]

    for label in labels:
        label.set_fontproperties(font)
    return ax, font



# ###################################################################
# other
# ###################################################################
def plot_distribution(series, ax=None, bins=10):
    if ax is None:
        ax = plt.gca()
    series.plot.hist(ax=ax, normed=1, bins=bins, alpha=0.6)
    # TODO: change bindwidth
    mean, std = series.mean(), series.std()
    min_, max_ = series.min(), series.max()
    x = np.linspace(min_, max_, len(series))
    step = (max_ - min_) / bins
    y = stats.norm.pdf(x, mean, std)
    point_x = np.linspace(min_ + step / 2, max_ - step / 2, bins)
    point_y = stats.norm.pdf(point_x, mean, std)
    ax.plot(x, y)
    ax.set_xlim([min_, max_])
    ax.set_xlabel(series.name)
    # print('type of point_x is{}'.format(type(point_x)))
    ax.scatter(point_x, point_y)
    title = ' '.join((series.name, 'distribution'))
    ax.set_title(title)
    # ax.set_xticklabels(series.index.values.tolist())
    return ax
