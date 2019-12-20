# -*- coding: utf-8 -*-
# @time    : 2018/9/10 17:15
# @author  : huangyu
# @file    : oracle_helper.py

import cx_Oracle
from DBUtils.PooledDB import PooledDB


class OracleHelper(object):
    def __init__(self, params):
        self._pool = PooledDB(cx_Oracle,
                              user=params.get("user"),
                              password=params.get("pwd"),
                              dsn="%s:%s/%s" % (params.get("host"), params.get("port"), params.get("dbname")),
                              mincached=int(params.get("mincached")),
                              maxcached=int(params.get("maxcached")),
                              blocking=True,
                              threaded=True,
                              encoding='UTF-8')

    def execute_query(self, sql):
        conn = self._pool.connection()
        cursor = conn.cursor()
        print(sql)
        cursor.execute(sql)
        results = cursor.fetchall()
        _desc = cursor.description
        cursor.close()
        conn.close()
        desc = [item[0] for item in _desc]
        return results, desc

    def execute_sql(self, sql):
        conn = self._pool.connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()



