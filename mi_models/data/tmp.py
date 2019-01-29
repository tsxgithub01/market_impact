# -*- coding: utf-8 -*-
# @time      : 2019/1/28 14:17
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : tmp.py

import pandas as pd
import json
from quant_models.utils.oracle_helper import OracleHelper

def func():
    df = pd.read_csv('cust_no_0161511266.csv')
    ticker = list(df['SEC_CODE'])
    db_config = {"user": "cust", "pwd": "admin123", "host": "10.200.40.170", "port": 1521, "dbname": "clouddb",
                 "mincached": 0, "maxcached": 1}
    obj = OracleHelper()
    test_lst = []
    for t in ticker:
        tic = str(t)
        if len(tic) < 6:
            tic = '{0}{1}'.format('0'*(6-len(tic)),tic)
        sql_str = "SELECT TICKER_SYMBOL,EXCHANGE_CD FROM MD_SECURITY WHERE TICKER_SYMBOL = '{0}' ".format(tic)
        # print(sql_str)
        rows, desc = obj.execute_query(sql_str)
        # print(rows)
        try:
            for item in rows:
                if item[1]:
                    test_lst.append('{0}.{1}'.format(item[0], item[1]))
                    print(item)
        except Exception as ex:
            # print(t)
            print(ex)
    with open('train_ids.json', 'w') as outfile:
        i_data = json.dumps(test_lst)
        outfile.write(i_data)

    with open('train_ids.json') as infile:
        i_data = infile.read()
        return json.loads(i_data)



if __name__ == '__main__':
    ret = func()
    print(ret)