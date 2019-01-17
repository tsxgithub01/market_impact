# -*- coding: utf-8 -*-
# @time      : 2018/10/29 15:52
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : testing.py

from mi_models.api.mi_model import MIModel
from mi_models.utils.utils import get_parent_dir
# from mi_models.utils.oracle_helper import OracleHelper
from logger import Logger
import os
from mi_models.data_processing.features_calculation import get_market_impact_features
import matplotlib.pyplot as plt
from copy import deepcopy
from WindPy import *

logger = Logger('result.txt', 'INFO', __name__).get_log()


# def test_oracle():
#     db_config = {"user": "cust", "pwd": "admin123", "host": "10.200.40.170", "dbname": "dbcenter",
#                  "mincached": 0, "maxcached": 1}
#     obj = OracleHelper(db_config)
#     print(obj)
#     # sql_str = ('''select *
#     #               from mkt_equw
#     #               where SECURITY_ID in (300182.XSHE)''')
#     # ret = obj.execute_query(sql_str)
#     # print(ret)

def logger_testing():
    logger.info("info")
    logger.error('error')
    logger.debug("debug")
    logger.warn('warn')
    logger.exception("exception")


# def get_market_impacts():
#     ret_features = get_market_impact_features(sec_code_to_order_ids={'603612': ['111111']}, features=[],
#                                               trade_direction=2,d


def get_mkt_data(symbol='', fields='', start_time='', end_time='', intraday_day=True):
    if intraday_day:
        data = w.wsi(symbol, fields, start_time, end_time)
    else:
        data = w.wsd(symbol, fields, start_time, end_time)
    return data.Data, data.Times, data.ErrorCode


def adjust_sum(val=[]):
    sum, cnt = 0.0, 0.0
    for item in val:
        if item != item:
            continue
        else:
            sum += item
            cnt += 1
    return sum, cnt


def suotong_mi(start_date='20180718', end_date='20181018'):
    db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                 "mincached": 0, "maxcached": 1}
    file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')
    mi = MIModel(db_config, file_name)
    train_codes = [('300182', 'XSHE'), ('002001', 'XSHE'), ('603608', 'XSHG')]
    train_codes = [('000001', 'XSHG')]

    # sub_order_num = 5396
    sub_order_num = 7962
    total_order_num = 3400000
    # order_num = 37778
    # order_num = 55737
    order_num = 55737
    total_hr = 3.0 if order_num == 55737 else 1.0
    # gen_features(self, sec_code, exchange, start_date=None, end_date=None, features=[], interval_mins=30):
    ret = mi.gen_features('603612', 'XSHG', start_date=start_date, end_date=end_date,
                          # sec_code_to_order_ids={'603612': ['111111']},
                          # trade_schedules=[('20180709 09:30:00', '20180709 10:00:00', 1000, 18),],
                          order_qty=[order_num], order_price=[17.67], interval_mins=60)
    ret_impacts = []
    start_dates = []
    for sec_code, item in ret.items():
        for sub_item in item:
            ret_impacts.append([sub_item[2], sub_item[3]])
            start_dates.append(sub_item[10].split(' ')[0])
    # open, dates, error_code = get_mkt_data(symbol='603612.SH', fields='open',
    #                                        start_time='2018-07-18', end_time='2018-08-31',
    #                                        intraday_day=False)
    #
    # close, dates, error_code = get_mkt_data(symbol='603612.SH', fields='close',
    #                                         start_time='2018-07-18', end_time='2018-08-31',
    #                                         intraday_day=False)

    from mi_models.data_processing.market import Market
    from utils.oracle_helper import OracleHelper
    OracleHelper(db_config)

    mkt = Market()
    mkt.sec_code = '603612'
    mkt.exchange = 'XSHG'
    mkt.initialize(start_date=start_date, end_date=end_date, db_obj=OracleHelper(db_config))
    opens = mkt.get_sod_price()

    total_tmp_rate = 0.0
    tmp_trading_cost = []
    initial_price = 17.67
    daily_vol = 103097

    perm_impact = []
    tmp_impact = []
    perm_cost = {}
    tmp_cost = {}
    total_cost = {}
    for idx, item in enumerate(ret_impacts):
        perm, tmp = item
        perm=perm/total_hr
        tmp=tmp/total_hr
        perm_impact.append(perm)
        d = start_dates[idx]
        try:
            open_price = opens.get(d)
            # print('open price for date:{0} is :{1}'.format(d, open_price))
        except Exception as ex:
            open_price = initial_price
        if not open_price:
            open_price = initial_price
        try:
            curr_perm = perm_cost.get(d) or 0.0
            updated_perm = curr_perm + open_price * abs(perm) * order_num
        except Exception as ex:
            updated_perm = 0.0
        # print(curr_perm, updated_perm)
        perm_cost.update({d: updated_perm})
        curr_tmp = tmp_cost.get(d) or 0.0
        updated_tmp = curr_tmp + open_price * abs(tmp) * order_num
        tmp_cost.update({d: updated_tmp})
        # print(tmp_cost)
        total_cost.update({d: perm_cost.get(d) + tmp_cost.get(d)})
        tmp_impact.append(abs(tmp))

    sorted_dates = sorted(list(set(start_dates)))
    perm_cost_vals = [perm_cost.get(d) for d in sorted_dates]
    for idx, item in enumerate(perm_cost_vals):
        if item != item:
            sum, cnt = adjust_sum(perm_cost_vals[idx - 3:idx + 3])
            perm_cost_vals[idx] = sum / cnt
    tmp_cost_vals = [tmp_cost.get(d) for d in sorted_dates]
    for idx, item in enumerate(tmp_cost_vals):
        if item != item:
            sum, cnt = adjust_sum(tmp_cost_vals[idx - 3:idx + 3])
            tmp_cost_vals[idx] = sum / cnt
    all_cost_vals = [total_cost.get(d) for d in sorted_dates]
    for idx, item in enumerate(all_cost_vals):
        if item != item:
            sum, cnt = adjust_sum(all_cost_vals[idx - 3:idx + 3])
            all_cost_vals[idx] = sum / cnt
    total_perm_val, total_tmp_val, total_val = 0.0, 0.0, 0.0
    for item in perm_cost_vals:
        total_perm_val += item
    for item in tmp_cost_vals:
        total_tmp_val += item

    total_val = total_perm_val + total_tmp_val

    init_val = initial_price * 3400000
    logger.info("total perm cost:{0}".format( total_perm_val))
    logger.info('total tmp cost:{0}'.format( total_tmp_val))
    logger.info('total cost:{0}'.format(total_val))
    logger.info('perm_rate:{0}'.format(total_perm_val / init_val))
    logger.info('tmp_rate:{0}'.format(total_tmp_val / init_val))

    import  pandas as pd
    import datetime
    now=datetime.datetime.now().strftime('hhmmss')
    df = pd.DataFrame({'perm':perm_cost_vals, 'all': all_cost_vals})
    df.to_csv('second2.csv')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.bar(sorted_dates, perm_cost_vals, alpha=0.65, color='r', label='permenent')
    # plt.plot(dates, trading_cost, label='compared', linestyle='-', c='r')
    plt.title(u'算法交易每日市场永久冲击成本')
    plt.xlabel(u'日期')
    # plt.gca().xaxis.set_major_formatter('%Y%M%d')
    plt.gcf().autofmt_xdate()
    plt.ylabel(u'市场永久冲击成本（元）')
    # xticks(range(10),['a','b'])
    plt.legend([u'每日市场永久冲击成本（元）'])
    xticks = list(range(0, len(sorted_dates), 10))
    xlabels = [sorted_dates[idx] for idx in xticks]
    # plt.set_xticks(xticks)
    plt.xticks(xticks, xlabels)
    # plt.set_xticklabels(xlabels, rotation=40)
    # print(sum(daily_cost))
    plt.show()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.bar(sorted_dates, all_cost_vals, alpha=0.65, color='r', label='permenent')
    # plt.plot(dates, trading_cost, label='compared', linestyle='-', c='r')
    plt.title(u'算法交易每日市场市场总冲击成本')
    plt.xlabel(u'日期')
    # plt.gca().xaxis.set_major_formatter('%Y%M%d')
    plt.gcf().autofmt_xdate()
    plt.ylabel(u'市场总冲击成本（元）')
    # xticks(range(10),['a','b'])
    plt.xticks(xticks, xlabels)
    plt.legend([u'每日市场总冲击成本（元）'])
    # print(sum(daily_cost))
    plt.show()

    #
    #
    #
    # # TWAP order
    # for idx, item in enumerate(all_impacts):
    #     for k, v in item.items():
    #         # print('perment rate', v[0],'temp rate', v[1])
    #         tmp_perm, tmp_tmp = v[0] * impact_open_prices[idx], v[1] * impact_open_prices[idx]
    #         total_tmp_rate += abs(v[1])
    #         tmp_perm = abs(tmp_perm)
    #         tmp_tmp = abs(tmp_tmp)
    #         trading_cost.append((tmp_perm + tmp_tmp) * daily_vol)
    #         perm_cost_vals.append(tmp_perm)
    #
    #         tmp_cost_vals.append(tmp_tmp * daily_vol)
    #
    # print('perm cost', perm_cost_vals)
    # print('trading cost', trading_cost)
    # print('result for twap')
    # print('accumulate market effect is:', sum(perm_cost_vals)/impact_open_prices[0])
    # print('trading cost is:', sum(trading_cost))
    # print('avg tmp rate:', total_tmp_rate/len(perm_cost_vals))
    # print('total tmp cost', sum(tmp_cost_vals))
    #
    #
    # # direct ordering
    # accu_market_impact = []
    # accu_trading_cost = []
    # total_tmp_rate = 0.0
    # for idx, all_impact_0816 in enumerate(all_impact_total):
    #     avg_daily_perm, avg_daily_tmp = 0.0, 0.0
    #     for dd, v in all_impact_0816.items():
    #         tmp_perm, tmp_tmp = v[0] * impact_open_prices[idx], v[1] * impact_open_prices[idx]
    #         tmp_perm = abs(tmp_perm)
    #         tmp_tmp = abs(tmp_tmp)
    #         accu_trading_cost.append((tmp_perm + tmp_tmp) * 3402201)
    #         accu_market_impact.append(tmp_perm)
    #         total_tmp_rate += abs(v[1])
    #     # accu_market_impact.append((avg_daily_perm / cnt1))
    #     # accu_trading_cost.append((avg_daily_tmp / cnt1 + avg_daily_perm / cnt1) * initial_price * 3402201)
    # print('result for direct order')
    # print('market impact: ', sum(accu_market_impact) / len(accu_market_impact))
    # print('market impact rate',sum(accu_market_impact) / (len(accu_market_impact)*impact_open_prices[0]) )
    # print('trading cost', sum(accu_trading_cost) / len(accu_trading_cost))
    # print('avg tmp rate', total_tmp_rate/len(accu_trading_cost))
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # from datetime import datetime
    # fig = plt.figure(dpi=128, figsize=(10, 6))
    # fig.autofmt_xdate()
    #
    # for idx, tt in enumerate(perm_cost_vals):
    #     print(trading_cost[idx]/tt)
    #
    # # perm_cost_vals = [item*100 for item in perm_cost_vals]
    # # plt.plot(dates, perm_cost_vals, label='compared', linestyle='-', c='b')
    # plt.bar(dates, perm_cost_vals,alpha=0.65, color='r', label='permenent')
    # # plt.plot(dates, trading_cost, label='compared', linestyle='-', c='r')
    # plt.title(u'拆单情况每日市场永久冲击成本')
    # plt.xlabel(u'日期')
    # # plt.gca().xaxis.set_major_formatter('%Y%M%d')
    # plt.gcf().autofmt_xdate()
    # plt.ylabel(u'市场永久冲击成本（元）')
    # # xticks(range(10),['a','b'])
    # plt.legend([u'每日市场永久冲击成本（元）'])
    # # print(sum(daily_cost))
    # # plt.show()
    #
    # # plt.plot(dates, trading_cost, label='compared', linestyle='-', c='r')
    # plt.bar(dates, trading_cost, alpha=0.65, color='b', label='permenent')
    # plt.title(u'拆单情况每日交易总成本')
    # plt.xlabel(u'日期')
    # # plt.gca().xaxis.set_major_formatter('%Y%M%d')
    # plt.gcf().autofmt_xdate()
    # plt.ylabel(u'每日交易总成本（元）')
    # # xticks(range(10),['a','b'])
    # plt.legend([u'拆单情况每日交易总成本（元）'])
    # # plt.show()


def mi_test():
    # db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
    #              "mincached": 0, "maxcached": 1}
    # file_name = os.path.join(get_parent_dir(), 'conf', 'istar_params')
    # mi = MIModel(db_config, file_name)
    # train_codes = [('300182', 'XSHE'), ('002001', 'XSHE'), ('603608', 'XSHG')]
    # train_codes = [('000001', 'XSHG')]
    # # gen_features(self, sec_code, exchange, start_date=None, end_date=None, features=[], interval_mins=30):
    # ret = mi.gen_features('603612', 'XSHG', start_date='20180719', end_date='20180721',
    #                       # sec_code_to_order_ids={'603612': ['111111']},
    #                       # trade_schedules=[('20180709 09:30:00', '20180709 10:00:00', 1000, 18),],
    #                       order_qty=[1000], order_price=[18])
    # print(ret)
    # mkt = Market()
    # mkt.sec_code = sec_code
    # mkt.exchange = exchange
    # mkt.initialize(start_date=get_prev_trading_date(start_date)[0], end_date=end_date, db_obj=db_obj)

    # for sec_code, exchange in train_codes:
    #     mi.train(sec_code=sec_code, exchange=exchange, start_date='20180103', end_date='20181018',
    #              trained_intervals=[60, 90, 120])
    # print(mi.load_model(file_name))
    # mi.load_model(file_name)
    #
    # for sec_code, exchange in train_codes:
    #     ret = mi.predict(sec_code=sec_code, exchange=exchange, quantity=3000, begin_time='20180709 09:30:41',
    #                      end_time='20180709 10:00:13')
    #     print(ret)
    print('mi model testing')


def get_mkt_test(start_date='20180718', end_date='20181110'):
    db_config = {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter",
                 "mincached": 0, "maxcached": 1}

    # from mi_models.data_processing.market import Market
    # from utils.oracle_helper import OracleHelper
    # OracleHelper(db_config)
    #
    # mkt = Market()
    # db_obj = OracleHelper(db_config)
    #
    # mkt.sec_code = '000001'
    # mkt.exchange = 'XSHG'
    # mkt.initialize(start_date=start_date, end_date=end_date, db_obj=db_obj)
    # closes = mkt.get_eod_price()
    # dates1 = closes.keys()
    # prices1 = closes.values()
    # print(closes)
    import cx_Oracle
    conn = cx_Oracle.connect('cust/admin123@10.200.40.170/clouddb', encoding='utf-8')
    cursor = conn.cursor()
    sql_str = "SELECT trade_date,close_price FROM MKT_EQUD WHERE ticker_symbol=603612 and trade_date>TO_DATE({0}, 'yyyymmdd') and  trade_date<TO_DATE({1}, 'yyyymmdd')".format(start_date, end_date)

    cursor.execute(sql_str)
    results = cursor.fetchall()
    tmp_dict={}
    for item in results:
        tmp_dict.update({item[0].strftime('%Y%m%d'):item[1]})
    dates1=sorted(list(tmp_dict.keys()))
    prices1=[tmp_dict.get(d) for d in dates1]

    # mkt.sec_code = '603612'
    # mkt.exchange = 'XSHG'
    # mkt.initialize(start_date=start_date, end_date=end_date, db_obj=db_obj)
    # closes = mkt.get_eod_price()
    # print(closes)
    # dates2 = closes.keys()
    # prices2 = closes.values()

    sql_str = "SELECT trade_date,close_index FROM MKT_IDXD WHERE ticker_symbol='000001' and exchange_cd='XSHG' and trade_date>TO_DATE({0}, 'yyyymmdd') and  trade_date<TO_DATE({1}, 'yyyymmdd')".format(start_date, end_date)

    cursor.execute(sql_str)
    results = cursor.fetchall()

    for item in results:
        tmp_dict.update({item[0].strftime('%Y%m%d'):item[1]})
    dates2=sorted(list(tmp_dict.keys()))
    prices2=[tmp_dict.get(d) for d in dates2]


    xticks = list(range(0, len(dates1), 5))
    xlabels = [dates1[idx] for idx in xticks]

    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax1 = fig.add_subplot(111)
    # ax1.rcParams['font.sans-serif'] = ['SimHei']
    ax1.plot(dates2, prices2, label='compared', linestyle='-', c='b')
    # ax1.plot(ret_avg.keys(), ret_sum)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel(u'上证市场价格走势(元）', fontdict={'fontproperties': 'SimHei'})
    ax1.set_title(u"索通解禁后市场走势与上证对比", fontdict={'fontproperties': 'SimHei'})
    ax1.legend(labels=[u'上证综指'])
    # ax1.gcf().autofmt_xdate()

    ax2 = ax1.twinx()  # this is the important function
    # ax2.rcParams['font.sans-serif'] = ['SimHei']
    ax2.plot(dates1, prices1, label='compared', linestyle='-', c='r')
    # ax2.plot(ret_avg.keys(), ret_sum_vol)
    # ax2.set_xlim([0, np.e])
    ax2.set_ylabel(u'索通解禁后市场价格走势（元）', fontdict={'fontproperties': 'SimHei'})
    # ax2.set_xlabel(u'上证市场价格走势', fontdict={'fontproperties': 'SimHei'})
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels)
    ax2.legend(labels=[u'索通发展'])
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    # suotong_mi('20180718', '20181018')
    # from utils.date_utils import get_all_trading_dates
    # print(len(get_all_trading_dates('20180718', '20181018')))
    get_mkt_test(start_date='20180718', end_date='20181110')
