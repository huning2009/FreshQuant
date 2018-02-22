# -*- coding: utf-8 -*-

"""
@Time    : 2018/1/19 9:11
@Author  : jessica.sun
参数配置优先级：策略代码中配置》命令行传参=run_file|run_code|run_func函数传参》用户配置文件》系统默认配置文件
"""
import click
from rqalpha import run_file

# 回测run_file_demo
config = {
    "base": {
        "start_date": "2010-01-01",
        "end_date": "2017-10-29",
        "frequency":'1d',
        "benchmark": "000300.XSHG",
        "accounts": {
            "stock": 40000000
        }
    },
    "extra": {
        "log_level": "error",
    },
    "mod": {
        "sys_analyser": {
            "enabled": True,
            "plot": True,
            "report": True,
            "output_file": 'out.pkl'},
        "sys_progress": {
            "enabled": True,
            "show": True
        },
        "sys_simulation":{
            "signal":True,
            "matching_type":"next_bar"
        }
    }
}
strategy_file_path = "./backtest_event.py"
run_file(strategy_file_path, config)
print ('Run Done!')

