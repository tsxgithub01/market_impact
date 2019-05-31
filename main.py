# -*- coding: utf-8 -*-
# @time      : 2018/11/1 9:14
# @author    : yuxiaoqi
# @file      : main.py

from mi_models.api.mi_model import MIModel
from mi_models.utils.utils import get_parent_dir
from mi_models.utils.utils import load_json_file
from mi_models.logger import Logger
import os
import pprint

logger = Logger('log.txt', 'INFO', __name__).get_log()


def _main(env='PROD'):
    uat_db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                     "mincached": 0, "maxcached": 1}
    prod_db_config = {"user": "cust", "pwd": "admin123", "host": "10.200.40.170", "port": 1521, "dbname": "clouddb",
                      "mincached": 0, "maxcached": 1}
    start_date = '20170103'
    end_date = '20181231'
    file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')
    db_config = uat_db_config if env == 'UAT' else prod_db_config
    mi = MIModel(db_config, file_name)
    train_sec_ids = load_json_file(os.path.join(get_parent_dir(), 'data', 'train_ids.json'))
    train_codes = [(item.split('.')[0], item.split('.')[1]) for item in train_sec_ids]
    results = []

    for sec_code, exchange in train_codes:
        ret = mi.train(sec_code=sec_code, exchange=exchange, start_date=start_date, end_date=end_date,
                       model_name='linear_nnls',
                       trained_intervals=[30, 60, 90])
        for min, val in ret.items():
            score, mse = val
            results.append([sec_code, min, mse, score])
        # with open(
        #         'mi_models/data/results/training_results_linearnnls_{0}_{1}_{2}'.format(train_codes[0], start_date, end_date),
        #         'w') as outfile:
        #     for item in results:
        #         outfile.write(str(item))
    mi.load_model(file_name)
    for sec_code, exchange in train_codes:
        begin_time = '20180709 09:25:41'
        end_time = '20180709 14:53:13'
        quantity = 726100
        ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=quantity, begin_time=begin_time,
                         end_time=end_time)
        tmp_impact, perm_impact, total_impact, instant_impact = ret[0]
        logger.info("股票:{0},从:{1} 到:{2},下单量：{3} 预测暂时市场冲击：{4}， 永久市场冲击：{5}，总市场冲击：{6}，"
                    "瞬时冲击：{7}".format('{0}.{1}'.format(sec_code, exchange), begin_time, end_time, quantity, tmp_impact,
                                      perm_impact, total_impact, instant_impact))


if __name__ == '__main__':
    _main('PROD')
