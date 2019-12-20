import configparser
import numpy as np
import os
import json
import sys
import hashlib
import talib as ta
from ..utils.logger import Logger
# from logger import Logger
logger = Logger('log.txt', 'INFO', __name__).get_log()


def get_config(overwrite_config_path=None):
    '''
    apply configparser to parse the config file, by default type is str
    :param overwrite_config_path: str, path of the config file to overwritten the default path
    :return:
    '''
    config = configparser.ConfigParser()
    config_path = overwrite_config_path or os.path.join(get_parent_dir(), 'conf', 'conf')
    config.read(config_path)
    logger.debug('Reading config file from:{0}'.format(config_path))
    return config


def set_config(overwrite_config_path='', section='', key='', val=''):
    config = configparser.ConfigParser()
    config_path = overwrite_config_path or os.path.join(get_parent_dir(), 'conf', 'conf')
    config.read(config_path)
    logger.info('Reading config file from:{0}'.format(config_path))
    sections = config.sections()
    if section not in sections:
        config.add_section(section)
    if key:
        config.set(section, key, val)
    config.write((open(config_path, 'w')))


def format_float(val=None):
    return np.nan if not val else float(val)


def handle_none(df):
    df.fillna(value=np.nan, inplace=True)
    df.replace(to_replace=np.NaN, value=0, inplace=True)
    df.replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)


def trans_none_2_nan(tmpdata):
    if (tmpdata == None):
        tmpdata = np.nan
    return tmpdata


def clear_none(tmpFactors):
    tmpFactors.fillna(value=np.nan, inplace=True)
    return tmpFactors


def clear_nan_inf(tmpfactors):
    tmpfactors.replace(to_replace=np.NaN, value=0, inplace=True)
    tmpfactors.replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)
    return tmpfactors


def df_to_payload(df=None):
    index_lst = df.index
    list_of_dicts = df.to_dict('record')
    record_num = len(index_lst)
    for i in range(record_num):
        list_of_dicts[i].update({'index': index_lst[i]})
    return list_of_dicts


def get_hash_key(key=None):
    if not key:
        return key

    if isinstance(key, list):
        key = ','.join(key)
    m1 = hashlib.md5()
    m1.update(key.encode('utf-8'))
    return m1.hexdigest()


def format_sec_code(val=None):
    sec_code = val if isinstance(val, str) else str(val)
    if len(sec_code) == 6:
        return sec_code
    else:
        return '0' * (6 - len(sec_code)) + sec_code


def list_files(abs_path=None, ref_path=None):
    if not (abs_path or ref_path):
        return []
    path = abs_path or os.path.join(get_parent_dir(), ref_path)
    if os.path.exists(path):
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            all_files = os.listdir(path)
            return ['{0}/{1}'.format(path, item) for item in all_files]
    return []


def adjusted_sma(inputs=[], period=10):
    ret = list(ta.SMA(np.array(list(inputs), dtype=float), timeperiod=period))
    fixed_len = period - 1 if period < len(inputs) else len(inputs)
    for i in range(fixed_len):
        ret[i] = sum(inputs[:i + 1]) / (i + 1)
    return ret


def get_parent_dir(file=None):
    _file = file or __file__
    curr_path = os.path.abspath(_file)
    parent_path = os.path.abspath(os.path.dirname(curr_path) + os.path.sep)
    return os.path.dirname(parent_path)


def write_json_file(file_path='', data=None):
    if not data:
        return
    with open(file_path, 'w') as outfile:
        j_data = json.dumps(data)
        outfile.write(j_data)


def load_json_file(filepath=''):
    with open(filepath) as infile:
        contents = infile.read()
        return json.loads(contents)


if __name__ == '__main__':
    print(get_parent_dir())