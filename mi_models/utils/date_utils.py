# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : date_utils.py

import datetime
from logger import Logger
from utils.oracle_helper import OracleHelper

logger = Logger('log.txt', 'INFO', __name__).get_log()


def get_dates_statics(start_date='', end_date=''):
    config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
              "mincached": 0, "maxcached": 1}
    db_obj = OracleHelper(config)
    sql_str = ('''select *
              from md_trade_cal
              where EXCHANGE_CD in ('XSHE','XSHG') and CALENDAR_DATE>= TO_DATE({}, 'YYYYMMDD') and CALENDAR_DATE<= TO_DATE({},'YYYYMMDD')''').format(
        start_date, end_date)
    ret, desc = db_obj.execute_query(sql_str)
    cols = [item[0] for item in desc]
    return ret, cols


def datetime_delta(dt=None, format=None, days=0, hours=0, minutes=0, seconds=0, output_format=None):
    if isinstance(dt, datetime.datetime):
        return dt + datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if isinstance(dt, str):
        if not format:
            logger.error('Format missing in datetime_delta for datetime:{0}'.format(dt))
            return None
        dt_time = datetime.datetime.strptime(dt, format)
        dt_time = dt_time + datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return dt_time.strftime(output_format) if output_format else dt_time.strftime(format)


def get_all_trading_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyymmdd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([item[1].strftime('%Y%m%d') for item in rows if item[3] == 1]))


def get_prev_trading_date(curr_date=''):
    rows, cols = get_dates_statics(curr_date, curr_date)
    return list(set([item[12].strftime('%Y%m%d') for item in rows]))


def get_all_month_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyydddd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([item[7].strftime('%Y%m%d') for item in rows]))


def get_all_quarter_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyydddd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([item[9].strftime('%Y%m%d') for item in rows]))


def get_all_year_end_dates(start_date='', end_date=''):
    '''
    format for start_date and end_date: 'yyyymmdd'
    :param start_date:
    :param end_date:
    :return: list of trading dates, with format 'yyyymmdd'
    '''
    rows, cols = get_dates_statics(start_date, end_date)
    return list(set([item[11].strftime('%Y%m%d') for item in rows]))


# def is_trading_date(curr_date=None):
#     if curr_date:
#         start_date = datetime_delta(dt=curr_date, format='%Y-%m-%d', days=-31, output_format='%Y-%m-%d')
#         end_date = datetime_delta(dt=curr_date, format='%Y-%m-%d', days=31, output_format='%Y-%m-%d')
#         dates = get_all_trading_dates(start_date, end_date)
#         return curr_date in dates

