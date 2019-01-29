# -*- coding: utf-8 -*-
# @time      : 2018/10/18 13:36
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : feature_calculation.py

import numpy as np
import math
import os
from collections import defaultdict
from ..utils.utils import list_files
from ..utils.utils import get_config
from ..data_processing.preprocessing import mock_datetimes
from ..data_processing.preprocessing import grouped_trade_schedule
from ..utils.date_utils import get_all_trading_dates
from ..utils.date_utils import get_prev_trading_date
from ..utils.utils import get_parent_dir
from ..utils.decorators import timeit
from ..data_processing.market import Market
from ..data_processing.order import Order
from ..logger import Logger

numerical_default = 0.0
config = get_config()

logger = Logger('log.txt', 'INFO', __name__).get_log()

_sgn = lambda val: 1 if val >= 0 else -1


def cal_trading_rate(row={}):
    T, X, V = row.get('volume_time'), row.get('volume'), row.get('total_volume')
    if not V or not T:
        raise ValueError('Divide by Zero')
    return abs(float(X) / (V * T))


def cal_permenent_cost_factor(row={}):
    sigma, T, X, V = row.get('sigma'), row.get('volume_time'), row.get('volume'), row.get('total_volume')
    alpha = float(config['market_impact']['alpha'])
    rate = cal_trading_rate(row=row)
    val = sigma * T * _sgn(X) * math.pow(rate, alpha)
    # 永久冲击成本的时间累计以及异方差处理，使用log因子
    return 0.5 * val, math.log2(0.5 * val)


def get_nyu_perm_factor(row={}):
    sigma, VT, ADV, X = row.get('SIGMA'), row.get('VT'), row.get('ADV'), row.get('X')
    if VT != VT or ADV != ADV:
        raise ValueError('Divide by Zero')
    rate = abs(float(X) / VT)
    alpha = float(config['market_impact']['alpha'])
    return sigma * (float(VT) / ADV) * _sgn(X) * math.pow(rate, alpha)


def get_nyu_tmp_factor(row={}):
    sigma, VT, ADV, X = row.get('SIGMA'), row.get('VT'), row.get('ADV'), row.get('X')
    if VT != VT:
        raise ValueError('Divide by Zero')
    rate = abs(float(X) / VT)
    beta = float(config['market_impact']['beta'])
    return sigma * _sgn(X) * math.pow(rate, beta)


def cal_tempaory_cost_factor(row={}):
    sigma, T, X, V = row.get('sigma'), row.get('volume_time'), row.get('volume'), row.get('total_volume')
    if not V or not T:
        raise ValueError('Divide by Zero')
    beta = float(config['market_impact']['beta'])
    rate = cal_trading_rate(row=row)
    val = sigma * _sgn(X) * math.pow(rate, beta)
    # 异方差处理，使用log因子
    return val, math.log2(val)


def get_perm_and_tmp_cost(row={}):
    sigma, T, X, V = row.get('sigma'), row.get('volume_time'), row.get('volume'), row.get('total_volume')
    alpha, beta, Sig, phi = float(config['market_impact']['alpha']), float(config['market_impact']['beta']), float(
        config['market_impact']['sigma']), float(config['market_impact']['phi'])
    rate = cal_trading_rate(row=row)
    perm = Sig * sigma * T * _sgn(X) * math.pow(rate, alpha)
    tmp = phi * sigma * _sgn(X) * math.pow(rate, beta)
    return perm, tmp


def get_market_impact_label(start_price=np.nan, avg_price=np.nan):
    if start_price == np.nan or avg_price == np.nan:
        logger.error('Fail to calculate label with start_price:{0}, and avg_price:{1}'.format(start_price, avg_price))
        return np.nan
    return float(avg_price - start_price) / start_price if start_price else np.nan


def get_istart_label(start_price=np.nan, avg_price=np.nan, pov=np.nan, sec_code='', b1=None, a4=None, file_name=''):
    if start_price == np.nan or avg_price == np.nan or pov == np.nan:
        logger.error('Fail to generate label, Price value nan')
        return np.nan
    mi = get_market_impact_label(start_price, avg_price)
    params = get_config(overwrite_config_path=file_name)
    b1 = b1 or float(params[sec_code]['b1'])
    a4 = a4 or float(params[sec_code]['a4'])
    star = (1 - b1) + math.pow(abs(pov), a4) * b1
    if pov < 0:
        return np.nan
    try:
        ret = math.log1p(mi / star - 1.0)
    except Exception as ex:
        logger.error('Fail to calculate i_star label for mi{0}, and star:{1}, with error:{2}'.format(mi, star, ex))
        return np.nan
    return ret


def get_log_sigma(row={}):
    return math.log1p(row.get('SIGMA') - 1.0)


def get_log_q_adv(row={}):
    q, adv = row.get('Q'), row.get('ADV')
    return float(math.log1p(abs(q)) / adv - 1.0)


def get_source_features(inputs=[], mkt=None):
    if not inputs or not mkt:
        return []
    if len(inputs) == 4:
        start_datetime, end_datetime, vol, price = inputs
    else:
        start_datetime, end_datetime = inputs[:2]
        vol, price = np.nan, np.nan
    date = start_datetime.split(' ')[0]
    try:
        vt = mkt.get_ma_intraday_volume(start_datetime.split(' ')[1], end_datetime.split(' ')[1]).get(date) or np.nan
    except Exception as ex:
        logger.debug('Fail to get vt from {0} to {1}, with error:{2}'.format(start_datetime, end_datetime, ex))
        vt = np.nan
    try:
        q = mkt.get_intraday_bs_volume(start_datetime.split(' ')[1], end_datetime.split(' ')[1]).get(date) or np.nan
    except Exception as ex:
        logger.debug('Fail to get Q from {0} to {1} with error:{2}'.format(start_datetime, end_datetime, ex))
        q = np.nan
    try:
        adv_mapping = mkt.get_ma_volume()
        adv = adv_mapping.get(date) or np.nan
    except Exception as ex:
        logger.debug('Fail to get adv for date {0} with error:{1}'.format(date, ex))
        adv = np.nan
    try:
        sigma = mkt.get_daily_sigma(date=date)
    except Exception as ex:
        logger.debug('Fail to get sigma for date{0} with error:{1}'.format(date, ex))
        sigma = np.nan
    if sigma == 1.0:
        logger.debug("Sigma is 1.0 for data:{0}".format(date))
    try:
        avg_price = mkt.get_avg_price(start_datetime, end_datetime)
    except Exception as ex:
        logger.debug(
            'Fail to get average price from {0} to {1} with error:{2}'.format(start_datetime, end_datetime, ex))
        avg_price = np.nan
    try:
        start_price, end_price = mkt.get_start_end_price(start_datetime, end_datetime)
    except Exception as ex:
        logger.debug(
            'Fail to get start and end price from {0} to {1} with error:{2}'.format(start_datetime, end_datetime, ex))
        start_price, end_price = np.nan, np.nan
    return {'ADV': adv, 'SIGMA': sigma, 'X': vol, 'Q': q / 2, 'VWAP': avg_price, 'P0': start_price, 'VT': vt,
            'start_time': start_datetime, 'end_time': end_datetime}


def get_cal_features(row={}, features=[]):
    feature_to_executors = {
        'log_q_adv': get_log_q_adv,
        'log_sigma': get_log_sigma,
        'nyu_perm': get_nyu_perm_factor,
        'nyu_tmp': get_nyu_tmp_factor,
        'pov': lambda val: float(val.get('Q')) / val.get('VT'),
        'volume_time': lambda val: float(val.get('VT')) / val.get('ADV'),
        'q_adv': lambda val: float(abs(val.get('Q')) / val.get('ADV')),
    }
    for f in features:
        try:
            row.update({f: feature_to_executors.get(f.lower())(row)})
        except Exception as ex:
            logger.error('Fail to get feature {0} with error {1}'.format(f, ex))
            row.update({f: np.nan})
        logger.debug("features updated after calculating feature:{0},{1}".format(f, row))
    return row


def _resolve_start_end_dates(start_date=None, end_date=None, datetime_intervals=[]):
    if start_date and end_date:
        return start_date, end_date
    dates = []
    for item in datetime_intervals:
        try:
            sd, ed = item[0].split(' ')[0], item[1].split(' ')[0]
            dates.extend([sd, ed])
        except Exception as ex:
            logger.error('Datetime format not valid for {0} with error'.format(item, ex))
    if dates:
        dates = list(set(dates))
        dates = sorted(dates)
        return dates[0], dates[-1]


@timeit
def get_market_impact_features(features=[], sec_codes=[], sec_code_to_order_ids={}, trade_direction=1, start_date=None,
                               end_date=None, interval_mins=30, datetime_intervals=[], order_qty=[],
                               trade_schedules=[], exchange='XSHG', db_obj=None, order_price=[]):
    features_by_sec_code = {}
    features = features or config['market_impact']['features'].split(',')
    # TODO double check
    if 'LOG_Q_ADV' in features:
        filtered = True
    else:
        filtered = False
    # features.extend(['start_time', 'end_time'])
    cal_features = [f for f in features if
                    f not in ['ADV', 'SIGMA', 'X', 'Q', 'VWAP', 'P0', 'VT', 'start_time', 'end_time']]
    cnt = 0
    start_date, end_date = _resolve_start_end_dates(start_date, end_date, datetime_intervals)
    logger.info("Start get_market_impact_features for sec_code:{0},exchange:{1} from {2} to {3} with interval:{4},"
                " datetime_invervals:{5}, order_qty:{6}, features: {7}, db_obj:{8}".format(sec_codes, exchange,
                                                                                           start_date, end_date,
                                                                                           interval_mins,
                                                                                           datetime_intervals,
                                                                                           order_qty, features, db_obj))
    for sec_code in sec_codes:
        cnt += 1
        ret_features = defaultdict(list)
        order_ids = sec_code_to_order_ids.get(sec_code)
        mkt = Market()
        mkt.sec_code = sec_code
        mkt.exchange = exchange
        mkt.initialize(start_date=get_prev_trading_date(start_date)[0], end_date=end_date, db_obj=db_obj)
        input_lst = []
        if order_ids:
            for order_id in order_ids:
                order = Order(order_id)
                order.sec_code = sec_code
                order.order_direction = trade_direction
                order.trade_schedule = trade_schedules
                input_lst.extend(grouped_trade_schedule(order.trade_schedule))
        elif datetime_intervals:
            input_lst = datetime_intervals
        else:
            all_trading_dates = get_all_trading_dates(start_date=start_date, end_date=end_date)
            input_lst = mock_datetimes(all_trading_dates[:-1], interval_mins=interval_mins, mkt=mkt, filtered=filtered)
        logger.debug(
            'Processing dates from {0} to {1} for sec code:{2} with input list size:{3}'.format(start_date, end_date,
                                                                                                sec_code,
                                                                                                len(input_lst)))
        total_len = len(input_lst)
        logger.info('Return {0} mock orders'.format(total_len))
        cnt = 0
        if order_qty and order_price:
            for item in input_lst:
                item.extend([order_qty[0], order_price[0]])
        for idx, inputs in enumerate(input_lst):
            cnt += 1
            logger.debug(
                "Processsing {0} th mock order: {1} out of {2} features_calculation".format(cnt, total_len, inputs))
            curr_row = get_source_features(inputs, mkt)
            logger.debug("source features returned before adjusted quantity:{0}".format(curr_row))
            if order_qty:
                logger.debug('Order quantity is set as:{0}'.format(order_qty[0]))
                curr_row.update({'Q': order_qty[0]})
            logger.debug("source features returned after adjusted quantity:{0}".format(curr_row))
            flag = 0
            # remove all nan row, may hv better way
            for f, fv in curr_row.items():
                if not (fv == np.nan):
                    flag = 1
            if not flag:
                continue
            curr_row = get_cal_features(curr_row, cal_features)
            logger.debug("calculate features returned :{0}".format(curr_row))
            start_time = curr_row.get('start_time')
            sub_path = start_time.split(' ')[0][:6]
            if 'P0' not in features:
                features.append('P0')
            if 'VWAP' not in features:
                features.append('VWAP')
            ret_features[sub_path].append([curr_row.get(f.strip('\n')) for f in features])
            # ret_features.append([curr_row.get(f.strip('\n')) for f in features])
            logger.debug("calculate features returned after formatting:{0}".format(curr_row))
        features_by_sec_code.update({sec_code: ret_features})
    logger.info("Done get_market_impact_features for sec_code:{0},exchange:{1} from {2} to {3} with interval:{4},"
                " datetime_invervals:{5}, order_qty:{6}, results: {7}, db_obj:{8}".format(sec_codes, exchange,
                                                                                          start_date, end_date,
                                                                                          interval_mins,
                                                                                          datetime_intervals,
                                                                                          order_qty,
                                                                                          features_by_sec_code, db_obj))
    logger.debug("features returned in get_market_impact_features before return:{0}".format(features_by_sec_code))
    return features_by_sec_code


def save_features(payload=[], path='', key_type=''):
    # feature_path = path
    base_dir, file_name = os.path.split(path)
    if not os.path.isdir(base_dir):
        logger.debug('Feature directory does not exist, create a new directory:{0}'.format(base_dir))
        os.mkdir(base_dir)
    logger.info('save_features to file:{0}'.format(path))
    with open(path, 'w') as f:
        for row in payload:
            if isinstance(row, dict):
                key = row.pop(key_type)
                lst = row.values()
                line = key
            elif isinstance(row, list) or isinstance(row, np.ndarray):
                lst = row
                line = ''
            else:
                lst = []
                line = ''
            for item in lst:
                line = '{0}\t{1}'.format(line, item) if line else '{0}'.format(item)
            line = line + '\n'
            f.write(line)
    f.close()


def read_features(feature_name=None):
    feature_path = os.path.join(get_parent_dir(), 'data', 'features')
    feature_name = [feature_name] if isinstance(feature_name, str) else feature_name
    for item in feature_name:
        feature_path = os.path.join(feature_path,item)
    # feature_path = os.path.join(get_parent_dir(), 'data', 'features', feature_name)
    logger.info("read features for sec_code:{0}".format(feature_name))
    files = list_files(abs_path=feature_path)
    ret_lines = []
    n_files = len(files)
    cnt = 0
    for f in files:
        cnt += 1
        with open(f, 'r') as fr:
            logger.info("read features from file:{0}, completed:{1} out of {2}".format(f, cnt, n_files))
            lines = fr.readlines()
            row = [line.strip('\n').split('\t') for line in lines]
            flag = False
            for i in row:
                if i is not 'nan':
                    flag = True
            if flag:
                ret_lines.extend(row)
    return ret_lines
