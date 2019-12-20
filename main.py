# -*- coding: utf-8 -*-
# @time      : 2018/11/1 9:14
# @author    : yuxiaoqi
# @file      : main.py

from mi_models.api.mi_model import MIModel
from mi_models.utils.utils import get_parent_dir
from mi_models.utils.utils import load_json_file
from mi_models.utils.logger import Logger
import os

logger = Logger('log.txt', 'INFO', __name__).get_log()


def _main(env='PROD'):
    uat_db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                     "mincached": 0, "maxcached": 1}
    prod_db_config = {"user": "gfangm", "pwd": "Gfangm1023_cms2019", "host": "10.200.40.170", "dbname": "clouddb",
                      "mincached": 1, "maxcached": 4, "port": 1521}
    start_date = '20191008'
    end_date = '20191129'

    # file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')  # train and predict with istar model
    file_name = os.path.join(get_parent_dir(), 'conf', 'nyu_params') #train and predict with nyu model
    db_config = uat_db_config if env == 'UAT' else prod_db_config
    mi = MIModel(db_config, file_name)
    train_sec_ids = load_json_file(os.path.join(get_parent_dir(), 'data', 'train_ids.json'))
    train_codes = [(item.split('.')[0], item.split('.')[1]) for item in train_sec_ids]
    results = []

    # train models
    for sec_code, exchange in train_codes:
        ret = mi.train(sec_code=sec_code, exchange=exchange, start_date=start_date, end_date=end_date,
                       model_name='istar',
                       trained_intervals=[30, 60, 90])
        for min, val in ret.items():
            score, mse = val
            results.append([sec_code, start_date, end_date, min, mse, score])
        #save results
        with open(
                'mi_models/data/results/train_results_istar'.format(train_codes[0], start_date, end_date),
                'a') as outfile:
            for item in results:
                for _line in item:
                    outfile.write('{0}\n'.format(str(_line)))

    # predict results
    mi.load_model(file_name)
    for sec_code, exchange in train_codes:
        begin_time = '20191202 09:25:41'
        end_time = '20191202 14:53:13'
        quantity = 726100
        #if use_default is True, use the default params, else use the trained params
        ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=quantity, begin_time=begin_time,
                         end_time=end_time, features=['Q_ADV', 'POV', 'SIGMA'], model_name='istar', use_default=False)
        # ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=quantity, begin_time=begin_time,
        #                  end_time=end_time, features=['X', 'VT', 'SIGMA', 'T'], model_name='nyu_opt', use_default=True)
        tmp_impact, perm_impact, total_impact, instant_impact = ret[0][0]
        logger.info("股票:{0},从:{1} 到:{2},下单量：{3} 预测暂时市场冲击：{4}， 永久市场冲击：{5}，总市场冲击：{6}，"
                    "瞬时冲击：{7}".format('{0}.{1}'.format(sec_code, exchange), begin_time, end_time, quantity, tmp_impact,
                                      perm_impact, total_impact, instant_impact))


if __name__ == '__main__':
    _main('PROD')
