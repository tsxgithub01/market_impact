# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi
# @file      : decorators.py

import time
from ..logger import Logger

logger = Logger('log.txt', 'INFO', __name__).get_log()


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        logger.info('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        return result

    return timed


@timeit
def sample_usage():
    time.sleep(1)
