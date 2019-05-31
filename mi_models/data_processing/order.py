# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi
# @file      : order.py

class Order(object):
    def __init__(self, order_id):
        self._order_id = order_id
        self._order_type = None
        self._sec_code = None
        self._trade_schedule = []
        self._real_done_info = []
        self._trade_direction = 1
        self._execute_msg = None
        self._ticker = None
        self._order_time = None

    @property
    def order_time(self):
        return self._order_time

    @order_time.setter
    def order_time(self, value):
        self._order_time = value

    @property
    def order_id(self):
        return self._order_id

    @order_id.setter
    def order_id(self, value):
        self._order_id = value

    @property
    def ticker(self):
        return self._ticker

    @ticker.setter
    def ticker(self, value):
        self._ticker = value

    @property
    def order_type(self):
        return self._order_type

    @order_type.setter
    def order_type(self, value):
        self._order_type = value

    @property
    def order_direction(self):
        return self._trade_direction

    @order_direction.setter
    def order_direction(self, value):
        self._trade_direction = value

    @property
    def sec_code(self):
        return self._sec_code

    @sec_code.setter
    def sec_code(self, value):
        self._sec_code = value

    @property
    def trade_schedule(self):
        '''
        columns: [(start_datetime, end_datetime, volume, price)]
        :return:
        '''
        return self._trade_schedule

    @trade_schedule.setter
    def trade_schedule(self, value):
        self._trade_schedule = value

    @property
    def execute_msg(self):
        return self._execute_msg

    @execute_msg.setter
    def execute_msg(self, value):
        self._execute_msg = value

    @property
    def real_done(self):
        '''
        columns:[(start_datetime, end_datetime, volume, price)]
        :return:
        '''
        return self._real_done_info

    @real_done.setter
    def real_done(self, value):
        self._real_done_info = value

    def get_order_volum(self):
        return sum([item[-2] for item in self._trade_schedule])

    def get_avg_price(self):
        total_amt, total_vol = 0.0, 0.0
        for item in self._real_done_info:
            price, vol = item[4], item[1]
            total_amt += price * vol
            total_vol += vol
        if not total_amt or not total_vol:
            return 0.0
        return total_amt / total_vol
