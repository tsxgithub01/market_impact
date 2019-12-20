# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi
# @file      : oracle_data_fetcher.py


from datetime import date
from collections import defaultdict
from ..utils.utils import get_config
from ..utils.logger import Logger

config = get_config()
logger = Logger('log.txt', 'INFO', __name__).get_log()


class DataFetcher(object):
    def __init__(self, db_obj=None):
        if not db_obj:
            logger.error("Fail to init DataFetcher for empty db_obj")
        self.db_obj = db_obj

    def get_dates_statics(self, start_date='', end_date=''):
        sql_str = ('''select *
                  from cust.md_trade_cal
                  where EXCHANGE_CD in ('XSHE','XSHG') and CALENDAR_DATE>= TO_DATE({}, 'YYYYMMDD') and CALENDAR_DATE<= TO_DATE({},'YYYYMMDD')''').format(
            start_date, end_date)
        ret, desc = self.db_obj.execute_query(sql_str)
        cols = [item[0] for item in desc]
        return ret, cols

    def _get_sql_query(self, start_date=None, end_date=None, sec_codes=[], filter='',
                       orderby='', groupby='', table_name='CUST.EQUITY_PRICEMIN', exchangecd='XSHG'):
        start_date = start_date or 20120105
        end_date = end_date or date.today().strftime('%Y%m%d')
        if str(start_date)[:-2] != str(end_date)[:-2]:
            logger.warn('Should pass the date in the same month')
        if table_name == 'CUST.EQUITY_PRICEMIN':
            _table_name = '{0}{1}'.format(table_name,
                                          str(start_date)[:-2]) if table_name == 'CUST.EQUITY_PRICEMIN' else table_name
            sqlstr = ('''
                    select *
                      from {}
                      where EXCHANGECD in ('XSHE','XSHG') and DATADATE>='{}' and DATADATE<='{}'
                      ''').format(_table_name, start_date, end_date)
            ticker_key = 'TICKER'
        elif table_name == 'CUST.MKT_EQUD':
            sqlstr = ('''
                      select *
                      from CUST.MKT_EQUD
                      where EXCHANGE_CD in ('XSHE','XSHG') and TRADE_DATE>=TO_DATE('{}','YYYYMMDD') and TRADE_DATE<=TO_DATE('{}','YYYYMMDD')
                      ''').format(start_date, end_date)
            ticker_key = 'TICKER_SYMBOL'
        if sec_codes and len(sec_codes) > 1:
            sqlstr = '{0} and {1} in ({2})'.format(sqlstr, ticker_key, ','.join(sec_codes))
        elif sec_codes and len(sec_codes) == 1:
            sqlstr = '{0} and {1} = {2}'.format(sqlstr, ticker_key, sec_codes[0])
        if filter:
            sqlstr = '{0} and {1}'.format(sqlstr, filter)
        if orderby:
            sqlstr = '{0} Order by {1}'.format(sqlstr, orderby)
        if groupby:
            sqlstr = '{0} group by {1}'.format(sqlstr, groupby)
        return sqlstr

    def get_market_mins(self, startdate='', enddate='', sec_codes=[], filter='',
                        orderby='', groupby='', table_name='CUST.EQUITY_PRICEMIN'):
        '''
        Fetch the minute level data from tonglian in oracle
        Return the rows of values: ['DATADATE', 'TICKER', 'EXCHANGECD', 'SHORTNM', 'SECOFFSET', 'BARTIME', 'CLOSEPRICE',
         'OPENPRICE', 'HIGHPRICE', 'LOWPRICE', 'VOLUME', 'VALUE', 'VWAP']
         e.g. [(20180605, 600237, 'XSHG', '铜峰电子', 11640, '11:14', 4.46, 4.47, 4.47, 4.46, 68600, 306094.0, 4.462000000000001)]
        :param startdate: int
        :param enddate: int
        :param sec_codes: tuple of str
        :param filter:
        :param orderby:
        :param groupby:
        :return: rows, col_names; rows: list of tuple from the results; col_names: list of strings of the colume name of
                the table
        '''
        if not self.db_obj:
            logger.error("Fail in get_market_mins for empty db_obj")
        rows, col_names = self.get_dates_statics(startdate, enddate)
        all_trading_dates = [item[1].strftime('%Y%m%d') for item in rows if item[3] == 1]
        grouped_dates = defaultdict(list)

        for d in all_trading_dates:
            yymm = d[:6]
            rows, columns = [], []
            grouped_dates[yymm].append(int(d))
        total_len = len(grouped_dates)
        cnt = 0
        logger.info("Start query data in get_market_mins for query_date:{0}".format(len(grouped_dates)))
        for k, v in grouped_dates.items():
            cnt += 1
            logger.debug("query the {0} th table {1} out of {2}".format(cnt, k, total_len))
            v = sorted(v)
            sqlstr = self._get_sql_query(v[0], v[-1], sec_codes, filter, orderby,
                                         groupby, table_name)
            tmp_rows, desc = self.db_obj.execute_query(sqlstr)
            columns = [item[0] for item in desc]
            rows.extend(tmp_rows)
        logger.info("Done query data in get_market_mins for query_date:{0}".format(len(grouped_dates)))
        return rows, columns

    def get_market_daily(self, startdate='', enddate='', sec_codes=[], filter='',
                         orderby='', groupby='', table_name='CUST.MKT_EQUD'):
        if not self.db_obj:
            logger.error("Fail in get_market_mins for empty db_obj")
        sqlstr = self._get_sql_query(startdate, enddate, sec_codes, filter, orderby,
                                     groupby, table_name)
        logger.debug('query sql string is:{0}'.format(sqlstr))
        rows, desc = self.db_obj.execute_query(sqlstr)
        return rows, desc
