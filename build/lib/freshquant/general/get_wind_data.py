# -*- coding: UTF-8 -*-

from WindPy import *
w.start()

def get_minute(stocks,start,end):
    for stock in stocks:
        data=w.wsi(stock,'close,amt',start)
    return
