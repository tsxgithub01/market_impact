# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : preprocessing.py

import datetime
from collections import defaultdict
from ..utils.utils import get_config
from ..utils.date_utils import datetime_delta
from ..logger import Logger

logger = Logger('log.txt', 'INFO', __name__).get_log()
config = get_config()
source_cache = {}


def grouped_trade_schedule(trade_schedules=[], frequency=5):
    '''

    :param trade_schedules: [(start_datetime, end_datetime, volume, price)]
    :param frequency:
    :return:
    '''
    grouped_trades = defaultdict(list)
    ungrouped_trades = []
    for execute_trade in trade_schedules:
        try:
            start_datetime, end_date_time = execute_trade[0], execute_trade[1]
            h1, m1, s1 = start_datetime.split(' ')[1].split(':')
            h2, m2, s2 = end_date_time.split(' ')[1].split(':')
            if int(h1) == int(h2) and int(m1) // frequency == int(m2) // frequency:
                grouped_trades['{0}_{1}'.format(h1, int(m1) // frequency)].append(execute_trade)
            else:
                ungrouped_trades.append(execute_trade)
        except Exception as ex:
            ungrouped_trades.append(execute_trade)
            logger.debug('grouped trade with error:{0}'.format(ex))
    for k, v in grouped_trades.items():
        tmp_start_time, tmp_end_time, total_vol, total_price = None, None, None, None
        cnt = 0
        for sub_order in v:
            cnt += 1
            tmp_start_time = tmp_start_time if tmp_start_time and sub_order[0] > tmp_start_time else sub_order[0]
            tmp_end_time = tmp_end_time if tmp_end_time and sub_order[1] <= tmp_end_time else sub_order[1]
            total_vol = total_vol + sub_order[2] if total_vol else sub_order[2]
            total_price = total_price + sub_order[2] * sub_order[-1] if total_price else sub_order[2] * sub_order[-1]
        ungrouped_trades.append((tmp_start_time, tmp_end_time, total_vol, total_price / total_vol))
    return ungrouped_trades


def _is_valid_interval(start_datetime='', end_datetime='', mkt=None, filtered=True):
    if not filtered:
        return True
    d1, t1 = start_datetime.split(' ')
    d2, t2 = end_datetime.split(' ')
    if not mkt:
        return False
    q = mkt.get_intraday_bs_volume(t1, t2).get(d1)
    start_p, end_p = mkt.get_start_end_price(start_datetime, end_datetime)
    avg_price = mkt.get_avg_price(start_datetime, end_datetime)
    if q and start_p and avg_price and q > 0 and avg_price - start_p > 0:
        return True
    return False


def mock_datetimes(dates=[], interval_mins=5, mkt=None, filtered=True):
    ret = []
    for d in dates:
        # TODO hardcode start/end time
        start_datetime = '{0} {1}'.format(d, '09:30:00')
        end_datetime = '{0} {1}'.format(d, '15:00:00')
        curr_datetime = start_datetime
        curr_end_datetime = start_datetime
        skip = False
        while curr_datetime < end_datetime and curr_end_datetime < end_datetime:
            if curr_datetime >= '{0} {1}'.format(d, '11:00:00') and not skip:
                curr_datetime = '{0} {1}'.format(d, '13:00:00')
                skip = True
            curr_end_datetime = datetime_delta(dt=curr_datetime, format=config['constants']['no_dash_datetime_format'],
                                               minutes=interval_mins, )
            cnt = 1
            while curr_end_datetime < end_datetime and (not _is_valid_interval(curr_datetime, curr_end_datetime, mkt,
                                                                               filtered)):
                curr_end_datetime = datetime_delta(dt=curr_end_datetime,
                                                   format=config['constants']['no_dash_datetime_format'],
                                                   minutes=interval_mins)
                cnt += 1
            if _is_valid_interval(curr_datetime, curr_end_datetime, mkt, filtered):
                ret.append([curr_datetime, curr_end_datetime])
            curr_datetime = curr_end_datetime
    return ret


def pre_cache_market_data(startdate=None, enddate=None, sec_codes=[], exchangecd='XSHG'):
    from core.src.data_processing.sqlite_opt import insert_rows
    from core.src.data_processing.oracle_db_opt import fetch_data
    def _format_datetime(val):
        return datetime.datetime.strftime(datetime.datetime.strptime(val, '%Y%m%d %H:%M'), '%Y%m%d %H:%M:%S')

    rows, cols = fetch_data(startdate=startdate, enddate=enddate, sec_codes=sec_codes)
    inesrt_values = [(_format_datetime('{0} {1}'.format(item[0], item[5])), item[1], item[2], item[6], item[7],
                      item[8], item[9], item[10], item[11], item[12]) for item in rows if item[2] == exchangecd]
    insert_rows('MarketMin', 'DATETIME,TICKER,EXCHANGECD,CLOSEPRICE,OPENPRICE,HIGHPRICE,LOWPRICE,VOLUME,VALUE,VWAP',
                inesrt_values)
