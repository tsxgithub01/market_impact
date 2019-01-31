# -*- coding: utf-8 -*-
# @time      : 2018/11/1 9:14
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : main.py

from mi_models.api.mi_model import MIModel
from mi_models.utils.utils import get_parent_dir
from mi_models.utils.utils import load_json_file
from mi_models.utils.date_utils import datetime_delta
from mi_models.logger import Logger
import os
import json
import datetime

logger = Logger('log.txt', 'INFO', __name__).get_log()

if __name__ == '__main__':
    uat_db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                     "mincached": 0, "maxcached": 1}
    prod_db_config = {"user": "cust", "pwd": "admin123", "host": "10.200.40.170", "port": 1521, "dbname": "clouddb",
                      "mincached": 0, "maxcached": 1}
    start_date = '20150103'
    end_date = '20181231'
    file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')
    mi = MIModel(prod_db_config, file_name)
    # train_codes = [('000300', 'XSHG')]
    train_sec_ids = load_json_file(os.path.join(get_parent_dir(), 'data', 'train_ids.json'))
    train_codes = [(item.split('.')[0], item.split('.')[1]) for item in train_sec_ids][:10]
    train_codes = [('002782', 'XSHE')]
    results = []

    # for sec_code, exchange in train_codes:
    #     ret = mi.train(sec_code=sec_code, exchange=exchange, start_date=start_date, end_date=end_date,
    #                    model_name='linear_nnls',
    #                    trained_intervals=[30, 60, 90])
    #     for min, val in ret.items():
    #         score, mse = val
    #         results.append([sec_code, min, mse, score])
    #     print('returned score', ret)
    #
    # with open('training_results1_linear_{0}_{1}_{2}'.format(train_codes[0], start_date, end_date), 'w') as outfile:
    #     for item in results:
    #         print(item)
    #         outfile.write(str(item))
    # print(mi.load_model(file_name))

    lst = []
    start_time = '20190201 09:33:00'
    f = '%Y%m%d %H:%M:%S'
    # datetime.datetime.strptime(f,'%Y%m%d %H:%M:%S')
    for i in range(20):
        next_time = datetime_delta(dt=start_time, format=f, minutes=1)
        lst.append((start_time, next_time))
        start_time = next_time
    print(lst)
    s0=11.98
    rr= []
    for start_time, end_time in lst:
        ret = mi.predict(sec_code='002782', exchange='XSHE', quantity=-5000000 / len(lst),
                         begin_time=start_time,
                         end_time=end_time)
        rr.append(ret)

        print(ret)
    ret_lst = [item[0][1] for item in rr]
    prices = [abs(s0*item) for item in ret_lst if not(item != item)]
    print(ret_lst)
    print(prices)
    print(sum(prices))


    # for sec_code, exchange in train_codes:
    #     ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=5000000, begin_time='20190130 09:31:41',
    #                      end_time='20190130 09:32:00')
    #     print
    #     ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=5000000, begin_time(ret)='20190130 10:00:41',
    #                      end_time='20190130 11:00:00')
    #     print(ret)
    #     ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=5000000, begin_time='20190130 11:00:41',
    #                      end_time='20190130 13:31:00')
    #     print(ret)
    #     ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=5000000, begin_time='20190130 13:31:41',
    #                      end_time='20190130 15:00:00')
    #     print(ret)
