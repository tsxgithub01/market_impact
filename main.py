# -*- coding: utf-8 -*-
# @time      : 2018/11/1 9:14
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : main.py

from mi_models.api.mi_model import MIModel
from mi_models.utils.utils import get_parent_dir
from mi_models.logger import Logger
import os
import datetime

logger = Logger('log.txt', 'INFO', __name__).get_log()

if __name__ == '__main__':

    db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                 "mincached": 0, "maxcached": 1}
    db_config = {"user": "cust", "pwd": "admin123", "host": "10.200.40.170", "port": 1521, "dbname": "clouddb",
                 "mincached": 0, "maxcached": 1}
    file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')
    mi = MIModel(db_config, file_name)
    train_codes = [('300182', 'XSHE'), ('002001', 'XSHE'), ('603608', 'XSHG')]
    train_codes = [('000300', 'XSHG')]
    results = []
    for sec_code, exchange in train_codes:
        ret = mi.train(sec_code=sec_code, exchange=exchange, start_date='20150101', end_date='20161231',
                       model_name='istar_opt',
                       trained_intervals=[60, 90, 120])
        for min, score in ret.items():
            results.append([sec_code, min, score])
        print('returned score', ret)
    with open('training_results_300_opt', 'w') as outfile:
        for item in results:
            print(item)
            outfile.write(str(item))
    # print(mi.load_model(file_name))
    #
    # for sec_code, exchange in train_codes:
    #     ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=726100, begin_time='20180109 09:25:41',
    #                      end_time='20180709 14:53:13')
    #     print(ret)
