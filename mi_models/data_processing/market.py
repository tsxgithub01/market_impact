# -*- coding: utf-8 -*-
# @time      : 2018/10/18 13:36
# @author    : yuxiaoqi
# @file      : market.py

import numpy as np
import math
from functools import reduce
from datetime import datetime
from ..utils.date_utils import datetime_delta
from ..utils.utils import get_config
from ..utils.utils import adjusted_sma
from ..data_processing.oracle_data_fetcher import DataFetcher
import pandas as pd
from ..logger import Logger

config = get_config()
logger = Logger('log.txt', 'INFO', __name__).get_log()


def _format_datetime(val):
    return datetime.strftime(datetime.strptime(val, '%Y%m%d %H:%M'), '%Y%m%d %H:%M:%S')


class Market(object):
    def __init__(self):
        self._sec_code = None
        self._eod_cache = {}
        self._intraday_cache = None
        self._daily_log_return = None
        self._eod_volume = None
        self.default_side = 1  # 1 for buy, 2 fo r sell
        self._exchange = 'XSHG'
        self._data_fetcher = None

    @property
    def sec_code(self):
        return self._sec_code

    @sec_code.setter
    def sec_code(self, value):
        self._sec_code = value

    @property
    def exchange(self):
        return self._exchange

    @exchange.setter
    def exchange(self, value):
        self._exchange = value

    def _get_price(self, idx):
        row = self._intraday_cache.iloc[idx, [3, 4]]  # get close and open price
        return sum(row) / 2

    def get_price(self, datetime):
        filtered = self._intraday_cache[self._intraday_cache['DATETIME'] == datetime]

    def initialize(self, start_date='', end_date='', db_obj=None):
        if not self._sec_code:
            logger.error("sec_code is empty, set the sec_code before call initialize")
            return
        logger.info("Start initialize market for sec_code:{0} from {1} to {2}".format(self._sec_code,
                                                                                      start_date,
                                                                                      end_date))
        self._data_fetcher = DataFetcher(db_obj) if not self._data_fetcher else self._data_fetcher
        rows, cols = self._data_fetcher.get_market_mins(startdate=start_date, enddate=end_date,
                                                        sec_codes=[self._sec_code])
        _format_datetime = lambda val: datetime.strftime(datetime.strptime(val, '%Y%m%d %H:%M'), '%Y%m%d %H:%M:%S')
        rows = [(_format_datetime('{0} {1}'.format(item[0], item[5])), item[1], item[2], item[6], item[7],
                 item[8], item[9], item[10], item[11], item[12]) for item in rows if
                item[2] == self._exchange]
        colume_name = ['DATETIME', 'TICKER', 'EXCHANGECD', 'CLOSEPRICE', 'OPENPRICE', 'HIGHPRICE', 'LOWPRICE',
                       'VOLUME', 'VALUE', 'VWAP']
        self._intraday_cache = pd.DataFrame(rows, columns=colume_name)
        self._intraday_cache['DATE'] = [item.split(' ')[0] for item in self._intraday_cache['DATETIME']]
        self._intraday_cache['TIME'] = [item.split(' ')[1] for item in self._intraday_cache['DATETIME']]
        self._eod_cache = self._intraday_cache.groupby('DATE').agg(
            {'VOLUME': 'sum', 'CLOSEPRICE': 'last', 'OPENPRICE': 'first'})
        self._eod_cache = self._intraday_cache.sort_values(by=['DATETIME']).groupby('DATE').agg(
            {'VOLUME': 'sum', 'CLOSEPRICE': 'last', 'OPENPRICE': 'first'})

        self._intraday_cache = self._intraday_cache.sort_values(by=['DATETIME']).drop_duplicates()
        BS_TAGS = []
        BS_TAGS.append(self.default_side)
        row_num = self._intraday_cache.shape[0]
        for i in range(1, row_num):
            prev_row = self._intraday_cache.iloc[i - 1]
            curr_row = self._intraday_cache.iloc[i]
            if prev_row[-2] == curr_row[-2]:  # same date
                if curr_row[3] >= prev_row[3]:
                    BS_TAGS.append(1)
                else:
                    BS_TAGS.append(2)
            else:
                BS_TAGS.append(self.default_side)
        self._intraday_cache['BS'] = BS_TAGS
        logger.info("Done initialize market for sec_code:{0} from {1} to {2}".format(self._sec_code,
                                                                                     start_date,
                                                                                     end_date))

    def get_ma_volume(self, date_period=10):
        ret = {}
        try:
            ret = adjusted_sma(list(self._eod_cache['VOLUME']), date_period)
        except Exception as ex:
            logger.debug('Fail in get_ma_volume for date_period:{0} with error {1}'.format(date_period, ex))
            return ret
        return dict(zip(list(self._eod_cache.index), ret))

    def get_daily_log_returns(self):
        close_prices = self._eod_cache['CLOSEPRICE']
        dates = list(self._eod_cache.index)
        ret = [1.0]
        total_len = len(close_prices)
        for i in range(1, total_len):
            ret.append(math.log(close_prices[i]) / close_prices[i - 1])
        self._daily_log_return = dict(zip(dates, ret))
        return self._daily_log_return

    def get_daily_sigma(self, date_period=30, date=''):
        log_ret = self._daily_log_return or self.get_daily_log_returns()
        tmp = []
        for d, v in log_ret.items():
            if d <= date:
                tmp.append(v)
            else:
                break
        ret = tmp[-date_period:]
        avg_ret = sum(ret) / len(ret)
        ret[0] = avg_ret  # fill the star of the period return of the avg log return
        return math.sqrt(reduce(lambda x, y: x + math.pow((x - avg_ret), 2), ret) / (len(ret) - 1)) if len(
            ret) > 1 else ret[0]

    def get_sigma(self, start_time='', end_time='', adjusted=False):
        idx = np.where(self._intraday_cache.DATETIME >= start_time)
        if not len(idx[0]):
            return
        else:
            start_idx = idx[0][0]
        idx = np.where(self._intraday_cache.DATETIME <= end_time)
        if not len(idx[0]):
            return
        else:
            end_idx = idx[0][-1]
        ret = 0.0
        source_price = []
        for i in range(start_idx, end_idx + 1):
            source_price.append(self._get_price(i))
        for i in range(1, end_idx - start_idx + 1):
            ret += (source_price[i] - source_price[i - 1]) ** 2
        while adjusted and not ret:
            logger.debug('Start idx:{0}, end idx:{1}')
            if start_idx > 0:
                ret += (self._get_price(start_idx) - self._get_price(start_idx - 1)) ** 2
                start_idx -= 1
                if ret:
                    break
            if end_idx < self._intraday_cache.shape[0] - 1:
                ret += (self._get_price(end_idx + 1) - self._get_price(end_idx - 1)) ** 2
                end_idx += 1
                if ret:
                    break
        return math.sqrt(ret)

    def get_ma_intraday_volume(self, start_datetime='', end_datetime='', date_period=10):
        df1 = self._intraday_cache[
            (self._intraday_cache.TIME >= start_datetime) & (self._intraday_cache.TIME <= end_datetime)]
        if df1.size == 0.0:
            logger.error(
                "Market data missing in get_ma_intraday_volume from {0} to {1}".format(start_datetime,
                                                                                       end_datetime,
                                                                                       ))
            return {}
        df1 = df1.groupby('DATE').agg({'VOLUME': 'sum'})
        try:
            ret = adjusted_sma(list(df1['VOLUME']), date_period)
        except Exception as ex:
            logger.debug('Ma intraday volume fail with error {0}'.format(ex))
            return {}
        return dict(zip(list(df1.index), ret))

    def get_ma_intraday_bs_volume(self, start_datetime='', end_datetime='', date_period=10):
        try:
            df1 = self._intraday_cache[
                (self._intraday_cache.TIME >= start_datetime) & (self._intraday_cache.TIME <= end_datetime)]
            if df1.size == 0.0:
                logger.error(
                    "Market data missing in get_intraday_bs_volume from {0} to {1} ".format(
                        start_datetime,
                        end_datetime,
                    ))
                return np.nan
            else:
                df_buy = df1[df1['BS'] == 1].groupby('DATE').agg({'VOLUME': 'sum'})
                df_sell = df1[df1['BS'] == 2].groupby('DATE').agg({'VOLUME': 'sum'})
                if df_buy.size == 0:
                    df_bs = df_sell
                elif df_sell.size == 0:
                    df_bs = df_buy
                else:
                    df_bs = df_buy - df_sell
                ret = adjusted_sma(list(df_bs['VOLUME']), date_period)

                return dict(zip(list(df_bs.index), ret))
        except Exception as ex:
            logger.debug(
                'Fail in get_intraday_bs_volume from {0} to {1} with error'.format(start_datetime, end_datetime, ex))

    def get_intraday_bs_volume(self, start_datetime='', end_datetime=''):
        try:
            df1 = self._intraday_cache[
                (self._intraday_cache.TIME >= start_datetime) & (self._intraday_cache.TIME <= end_datetime)]
            if df1.size == 0.0:
                logger.error(
                    "Market data missing in get_intraday_bs_volume from {0} to {1} ".format(
                        start_datetime,
                        end_datetime,
                    ))
                return np.nan
            else:
                df_buy = df1[df1['BS'] == 1].groupby('DATE').agg({'VOLUME': 'sum'})
                df_sell = df1[df1['BS'] == 2].groupby('DATE').agg({'VOLUME': 'sum'})
                return (df_buy - df_sell).to_dict().get('VOLUME')
        except Exception as ex:
            logger.debug(
                'Fail in get_intraday_bs_volume from {0} to {1} with error'.format(start_datetime, end_datetime, ex))

    def get_eod_volume(self):
        vols = self._eod_cache['VOLUME']
        dates = list(self._eod_cache.index)
        self._eod_volume = dict(zip(dates, vols))
        return self._eod_volume

    def get_sod_price(self):
        prices = self._eod_cache['OPENPRICE']
        dates = list(self._eod_cache.index)
        self._sod_price = dict(zip(dates, prices))
        return self._sod_price

    def get_eod_price(self):
        prices = self._eod_cache['CLOSEPRICE']
        dates = list(self._eod_cache.index)
        self._eod_price = dict(zip(dates, prices))
        return self._eod_price

    def get_avg_price(self, start_datetime='', end_datetime=''):
        df1 = self._intraday_cache[
            (self._intraday_cache.DATETIME >= start_datetime) & (self._intraday_cache.DATETIME <= end_datetime)]
        if df1.size == 0.0:
            logger.error(
                "Market data missing in get_avg_price from {0} to {1}".format(start_datetime,
                                                                              end_datetime,
                                                                              ))
            return np.nan
        total_lst = df1['CLOSEPRICE'] * df1['VOLUME']
        try:
            sum_divided = sum(df1['VOLUME'])
            sum_divider = sum(total_lst)
            avg_p = sum_divider / sum_divided
        except Exception as ex:
            logger.debug(
                'Fail in get_avg_price for {0} divide {1} with error :{2}'.format(sum_divider, sum_divided, ex))
            return np.nan
        return avg_p

    def get_start_end_price(self, start_datetime='', end_datetime=''):
        df1 = self._intraday_cache[
            (self._intraday_cache.DATETIME >= start_datetime) & (self._intraday_cache.DATETIME <= end_datetime)]
        if df1.size == 0.0:
            logger.error(
                "Market data missing in get_start_end_price from {0} to {1}".format(start_datetime,
                                                                                    end_datetime,
                                                                                    ))
            return np.nan, np.nan
        avg_prices = list(df1['CLOSEPRICE'] + df1['OPENPRICE'])
        try:
            start_p, end_p = avg_prices[0] / 2, avg_prices[-1] / 2
        except Exception as ex:
            logger.debug('Fail in get_start_end_price from datetime:{0} to {1} with error:{1}'.format(start_datetime,
                                                                                                      end_datetime, ex))
            start_p, end_p = np.nan, np.nan
        return start_p, end_p

    def get_intraday_volume_time(self, start_datetime='', end_datetime=''):
        d1, t1 = start_datetime.split(' ')
        d2, t2 = end_datetime.split(' ')
        intraday_vol = self.get_ma_intraday_volume(t1, t2) or 0.0
        eod_vol = self.get_ma_volume().get(d1)
        return intraday_vol / eod_vol

    def get_acc_mkt_vol(self, start_datetime='', end_datetime=''):
        try:
            mkt = self._intraday_cache[self._intraday_cache.DATETIME >= start_datetime]
        except Exception as ex:
            logger.debug(
                "Fail in get_acc_mkt_vol from {0} to {1} with error:{2}".format(start_datetime, end_datetime, ex))
            return {}
        if mkt.size == 0.0:
            logger.error(
                "Market data missing in get_start_end_price from {0} to {1}".format(start_datetime,
                                                                                    end_datetime,
                                                                                    ))
            return {}
        mkt = mkt[mkt.DATETIME < end_datetime]
        mkt_vol_mins = list(mkt['VOLUME'])
        times_str = list(mkt['DATETIME'])
        if not mkt_vol_mins:
            return {}
        tmp_len = len(mkt_vol_mins)
        # total_vol = reduce(lambda x, y: x + y, mkt_vol_mins)
        total_vol = 0.0
        for item in mkt_vol_mins:
            if item != item:  # skip float nan
                continue
            total_vol += float(item)
        acc_vol = [mkt_vol_mins[0] / total_vol]
        curr_vol = 0.0
        for i in range(0, tmp_len):
            acc_vol.append((mkt_vol_mins[i] + curr_vol) / total_vol)
            curr_vol += mkt_vol_mins[i]
        # total_len = len(acc_vol)
        new_ret = {}
        for idx, time_key in enumerate(times_str):
            new_ret.update({time_key: acc_vol[idx]})
        return new_ret

    @classmethod
    def get_tao(cls, mkt_vol_acc={}, input_time=None):
        # try prev and next minute too
        ret = mkt_vol_acc.get(input_time)
        if ret:
            return ret
        min_delta = 1
        while not ret:
            prev_time = datetime_delta(dt=input_time, format=config['constants']['no_dash_datetime_format'],
                                       minutes=-min_delta)
            ret = mkt_vol_acc.get(prev_time)
            if ret:
                return ret
            else:
                next_time = datetime_delta(dt=input_time, format=config['constants']['no_dash_datetime_format'],
                                           minutes=min_delta)
                ret = mkt_vol_acc.get(next_time)
                if ret:
                    return ret
                else:
                    min_delta += 1
        return 0.0
