# -*- coding: utf-8 -*-
# @time    : 2018/10/18 11:08
# @author  : huangyu
# @file    : MIModel.py

import math
import copy
from mi_models.utils.logger import Logger
import numpy as np
import gc
import os
from ..utils.oracle_helper import OracleHelper
from ..utils.utils import get_config
from ..data_processing.features_calculation import get_market_impact_features
from ..data_processing.features_calculation import get_market_impact_label
from ..data_processing.features_calculation import get_istar_label
from ..data_processing.features_calculation import save_features
from ..data_processing.features_calculation import read_features
from ..data_processing.dataset import split_train_val_dataset
from ..data_processing.preprocessing import feature_preprocessing
from ..utils.utils import set_config
from ..utils.utils import format_float
from ..utils.utils import get_hash_key
from ..utils.decorators import timeit
from ..utils.utils import get_parent_dir
from ..utils.date_utils import get_all_month_start_end_dates
from ..model_processing.pipeline import num_pipeline
from ..model_processing.ml_reg_models import MlRegModel
from ..model_processing.sci_optimize_models import OptimizeModel
from ..model_processing.pipeline import get_pipeline
from ..model_processing.pipeline import save_pipeline
from ..model_processing.tf_models import TFRegModel

config = get_config()

logger = Logger('log.txt', 'INFO', __name__).get_log()


class MIModel(object):
    def __init__(self, db_config={}, file_name=None):
        """
        市场冲击模型API对象初始化
        :param db_config: 数据库相关参数，格式如下：
        {"user": "cust", "pwd": "admin123", "host": "172.253.32.132", "port": 1521, "dbname": "dbcenter", "mincached": 0, "maxcached": 1}
        :param file_name:模型文件路径，需要加载或生成的文件路径
        """
        self._oracle = OracleHelper(db_config)
        self.features = config['market_impact']['all_features'].split(",")
        self._parent_dir = get_parent_dir()
        self.file_name = file_name
        self._params = dict()

    def train(self, sec_code='', exchange='', start_date='', end_date='', model_name='',
              trained_intervals=[60, 90, 120]):
        '''
        训练指定个股的市场冲击模型
        :param sec_code:证券代码
        :param exchange:市场代码
        :param start_date:训练开始日期，如：'20180103'
        :param end_date: 训练结束日期， 如：'20180901'
        :param model_name:训练的模型，如： 'istar'
        :return: 不同分钟级别的训练分数结果，如： {60:{'mse':0.1, 'r2_score': 0.5}}
        '''

        ret = {}
        searched_params = {'b1': list(np.linspace(0.5, 0.99, 10)), 'a4': list(np.linspace(0.5, 0.99, 10))}
        start_date = start_date or '20180103'
        end_date = end_date or '20180926'
        trained_intervals = trained_intervals or [60, 90, 120]
        logger.info(
            'Start train for sec_code: {0} with intervals:{1} from {2} to {3}'.format(sec_code, trained_intervals,
                                                                                      start_date, end_date))
        for min in trained_intervals:
            best_score = -np.inf
            best_mse = np.inf
            split_months = get_all_month_start_end_dates(start_date, end_date)
            for sd, ed in split_months:
                hash_key = get_hash_key([sec_code, exchange, sd, ed, ','.join(self.features), str(min)])
                files = []
                try:
                    files = os.listdir(
                        os.path.join(get_parent_dir(), 'data', 'features', '{0}'.format(sec_code), '{0}'.format(min)))
                except Exception as ex:
                    logger.info('Fail to list the files with error:{0}'.format(ex))
                if hash_key not in files:
                    self.gen_features(sec_code=sec_code, exchange=exchange, start_date=sd, end_date=ed,
                                      features=self.features,
                                      interval_mins=min, file_name=hash_key)
                else:
                    logger.info('feature exist, skip')
            if model_name == 'istar':
                ret_score, ret_params = self._train_with_grid_search(sec_code=sec_code,
                                                                     features=['LOG_SIGMA', 'LOG_Q_ADV'],
                                                                     ajusted_mi=True,
                                                                     search_params=searched_params,
                                                                     model_name='linear_nnls',
                                                                     file_name=self.file_name,
                                                                     inverval_mins=min)
            elif model_name == 'istar_opt':
                ret_score, ret_params = self._train_with_istar_opt(sec_code=sec_code, model_name=model_name,
                                                                   features=['Q', 'ADV', 'POV', 'SIGMA'],
                                                                   file_name=self.file_name, opt_method='trf', lb=0.0,
                                                                   ub=np.inf)
            elif model_name == 'tf_dnn':
                features = config['market_impact']['tf_dnn_features'].split(',')
                ret_score, ret_params = self._train_with_dnn(sec_code=sec_code, model_name=model_name,
                                                             features=features,
                                                             interval_mins=min)

            elif model_name == 'nyu_opt':
                ret_score, ret_params = self._train_with_nyu_opt(sec_code=sec_code, model_name=model_name,
                                                                 features=['X', 'VT', 'SIGMA', 'T'],
                                                                 file_name=self.file_name, opt_method='trf', lb=0.0,
                                                                 ub=np.inf, interval_mins=min)

            logger.debug(
                'train result for sec_code {0} for interval_mins {1} from {2} to {3}: {4} '.format(sec_code, min,
                                                                                                   start_date,
                                                                                                   end_date, ret_score))
            # params = get_config(overwrite_config_path=self.file_name)
            # logger.debug('** a1: {0}, a2: {1},a3: {2}, a4: {3}, b1: {4}**'.format(params[sec_code]['a1'],
            #                                                                       params[sec_code]['a2'],
            #                                                                       params[sec_code]['a3'],
            #                                                                       params[sec_code]['a4'],
            #                                                                       params[sec_code]['b1']))
            if ret_score.get('r2_score') > best_score:
                self.save_model(file_name=self.file_name, sec_code=sec_code, params=ret_params)
                best_score = ret_score.get('r2_score')
                best_mse = ret_score.get('mse')
            ret.update({min: (best_score, best_mse)})
            gc.collect()
            logger.info('Done train for sec_code: {0} with intervals:{1} from {2} to {3}'.format(sec_code,
                                                                                                 trained_intervals,
                                                                                                 start_date,
                                                                                                 end_date))
        return ret

    def save_model(self, file_name='', sec_code='', params={}):
        """
        指定文件路径，将模型训练的结果(a、b参数)保存到文件中，增量写，如果参数已存在，则更新
        :param file_name:模型参数路径
        :param sec_code:证券代码
        :param params:要更新的模型参数，如：{'a4':0.1, 'b1':0.1, }
        :return:
        """
        logger.info('save_model for sec_code:{0} with params: {1} to file:{2}'.format(sec_code, params, file_name))
        for k, v in params.items():
            set_config(overwrite_config_path=file_name, section=sec_code, key=k, val=str(v))

    def load_model(self, file_name=None):
        """
        加载模型文件，读取所有个股的模型参数，保存到内存字典映射
        :param file_name:模型参数路径
        :return:
        """
        logger.info("load_model from file:{0}".format(file_name))
        file_name = self.file_name if self.file_name else file_name
        if not file_name:
            logger.error("File name is empty for loading model")
        ret = get_config(file_name)
        d = dict()
        sess = ret.sections()
        for s in sess:
            d.update({s: dict(ret[s])})
        self._params = d
        return d

    def predict(self, sec_code='', exchange='', quantity=None, begin_time='', end_time='', features=[],
                model_name='istar', use_default=False):
        """
        预测指定交易行为的冲击成本
        :param sec_code: 证券代码
        :param exchange: 市场代码
        :param quantity: 购买数量
        :param begin_time: 执行开始时间，如：'20180719 10:00:00'
        :param end_time: 执行结束时间，如：'20180719 10:30:00'
        :param features: 模型因子，默认为 ['Q_ADV', 'POV', 'SIGMA']
        :return: 冲击成本，tuple类型，包含4个元素，格式如：(临时冲击、永久冲击、市场冲击、瞬间冲击)
        """
        features = features or config['market_impact']['all_features'].split(',')
        exchange = exchange or 'XSHG'
        logger.info(
            "Start predict market impact for sec_code {0}, exchange:{1}, quantity:{2} from {3} to {4} with features".format(
                sec_code, exchange, quantity, begin_time, end_time, features))
        target_idx = [idx for idx, val in enumerate(self.features) if val in features]

        ret_features = get_market_impact_features(features=[self.features[idx] for idx in target_idx],
                                                  sec_codes=[sec_code],
                                                  datetime_intervals=[[begin_time, end_time]],
                                                  order_qty=[quantity], exchange=exchange, db_obj=self._oracle)

        if not ret_features.get(sec_code):
            logger.exception("No feature returns for sec_code:{0}".format(sec_code))
            return
        logger.info('Return features:{0} in predict for sec_code:{1}, begin_time:{2}, end_time:{3}'.format(ret_features,
                                                                                                           sec_code,
                                                                                                           begin_time,
                                                                                                           end_time))
        ret = []
        _sgn = lambda x: -1 if x < 0 else 1
        if not self._params:
            self.load_model(self.file_name)
        test_rows = []
        if model_name == 'istar':
            logger.debug('Predicting the results with model: {0}'.format(model_name))
            b1, a1, a2, a3, a4 = float(self._params.get(sec_code).get('b1')), \
                                 float(self._params.get(sec_code).get('a1')), \
                                 float(self._params.get(sec_code).get('a2')), \
                                 float(self._params.get(sec_code).get('a3')), \
                                 float(self._params.get(sec_code).get('a4'))
            for sec_code, val in ret_features.items():
                for d, feature_lst in val.items():
                    for item in feature_lst:
                        q_adv, pov, sigma = item[:3]
                        i_star = a1 * math.pow(abs(q_adv), a2) * math.pow(sigma, a3)
                        # tmp_impact = b1 * i_star * math.pow(abs(pov), a4)* _sgn(pov)
                        # perm_impact = (1 - b1) * i_star * _sgn(pov)
                        tmp_impact = b1 * i_star * math.pow(abs(pov), a4)
                        perm_impact = (1 - b1) * i_star
                        logger.debug(
                            'q_adv:{0},pov:{1}, sigma:{2},i_star:{3},tmp:{4},perm:{5}'.format(q_adv, pov, sigma, i_star,
                                                                                              tmp_impact, perm_impact))
                        logger.info(
                            'Return temp impact:{0}, Perm impact:{1} for sec_code:{2}'.format(tmp_impact, perm_impact,
                                                                                              sec_code))
                        ret.append((tmp_impact, perm_impact, tmp_impact + perm_impact, i_star))
            logger.info("Done predict with results:{0}".format(ret))
        elif model_name == 'nyu_opt':
            logger.debug('Predicting the results with model: {0}'.format(model_name))
            test_rows = []
            param_key = '0' if use_default else sec_code
            alpha, beta, perm_param, tmp_param = float(self._params.get(param_key).get('alpha')), \
                                                 float(self._params.get(param_key).get('beta')), \
                                                 float(self._params.get(param_key).get('perm_param')), \
                                                 float(self._params.get(param_key).get('tmp_param'))
            _sgn = lambda x: 1 if x > 0 else -1
            for sec_code, val in ret_features.items():
                for d, feature_lst in val.items():
                    for item in feature_lst:
                        # FIXME remove the hardcode index
                        sigma, x, vt, t = item[:4]
                        perm_impact = sigma * t * _sgn(x) * perm_param * math.pow(x / vt, alpha)
                        tmp_impact = sigma * _sgn(x) * tmp_param * math.pow(x / vt, beta)
                        logger.info(
                            'Return temp impact:{0}, Perm impact:{1} for sec_code:{2}'.format(tmp_impact, perm_impact,
                                                                                              sec_code))
                        ret.append((tmp_impact, perm_impact, tmp_impact + perm_impact, tmp_impact + perm_impact))
                        test_rows.append([x, vt, sigma, t, alpha, beta, perm_param, tmp_param, perm_impact, tmp_impact])
            # df = pd.DataFrame(test_rows, columns=['x', 'vt', 'sigma', 't', 'alpha', 'beta', 'perm_param', 'tmp_param',
            #                                       'perm_impact', 'tmp_impact'])
            # df.to_csv("test_rows.csv")
            logger.info("Done predict with results:{0}".format(ret))
        elif model_name == 'tf_dnn':
            acc_features = config['market_impact']['acc_features'].split(',')
            acc_idx = [idx for idx, val in enumerate(self.features) if val in acc_features]
            x, acc = [], []
            acc_idx = [idx for idx, val in enumerate(self.features) if val in acc_features]
            for sec_code, val in ret_features.items():
                for d, feature_lst in val.items():
                    for item in feature_lst:
                        _x = item[:-2]
                        _a = [item[idx] for idx in acc_idx]
                        # _x.extend(item[8:-2])
                    x.append(_x)
                    acc.append(_a)
            x = np.array(x)
            acc = np.array(acc)
            if not self._build_model:
                # TODO check the parameter settings
                self._m = TFRegModel() + 2
                self._m.build_model(x_shape=x.shape, acc_shape=acc.shape)
                self._build_model = True
            self._m.load_model(model_name)
            _x_pipeline_key = get_hash_key([self.features[idx] for idx in target_idx] + [sec_code])
            _acc_pipeline_key = get_hash_key(acc_features + [sec_code])
            _x_pipeline = get_pipeline(f_name='feature_pipeline_{0}'.format(_x_pipeline_key), f_type='num')
            _acc_pipeline = get_pipeline(f_name='feature_pipeline_{0}'.format(_acc_pipeline_key), f_type='num')
            std_x = _x_pipeline.transform(x)
            std_acc = _acc_pipeline.transform(acc)
            y = self._m.predict(std_x, std_acc)
            print(y)
            ret.append([item[0][0] for item in y])
        # TODO remove test_rows, this test_rows are only for the mi report testing to check the params for nyu_opt
        # models calculation
        return ret, test_rows

    @timeit
    def gen_features(self, sec_code, exchange, start_date=None, end_date=None, features=[], interval_mins=30, **kwargs):
        '''
        生成因子，存储在data/features/'%s'%sec_code
        :param sec_code: 证券代码
        :param exchange: 市场代码
        :param start_date: 开始时间
        :param end_date: 结束时间
        :param features: 训练因子
        :param interval_mins: 间隔分钟数
        :return:
        '''
        logger.info(
            'Start gen_features for sec_code:{0} and exchange:{1} from {2} to {3} with features {4} and interval {5}'.format(
                sec_code, exchange, start_date, end_date, features, interval_mins))
        sec_code_to_order_ids = kwargs.get('sec_code_to_order_ids') or {}
        order_qty = kwargs.get('order_qty') or None
        order_price = kwargs.get('order_price') or None
        ret_features = get_market_impact_features(features=features or self.features, sec_codes=[sec_code],
                                                  start_date=start_date, end_date=end_date, interval_mins=interval_mins,
                                                  exchange=exchange, db_obj=self._oracle,
                                                  sec_code_to_order_ids=sec_code_to_order_ids, order_qty=order_qty,
                                                  order_price=order_price)

        for sec_code, features in ret_features.items():
            for yymm, rows in features.items():
                name = kwargs.get('file_name') or '{0}'.format(yymm)
                path = os.path.join(self._parent_dir, 'data', 'features',
                                    '{0}'.format(sec_code), '{0}'.format(interval_mins), '{0}'.format(name))
                save_features(rows, path=path)
        logger.info(
            'Done gen_features for sec_code:{0} and exchange: {1} from {2} to {3} for  interval {4}'.format(
                sec_code, exchange, start_date, end_date, interval_mins))
        return ret_features

    def set_default_params(self, sec_code='', trained_params='b'):
        '''
        用理论值初始化模型参数
        :param sec_code: 证券代码
        :param trained_params: 初始化的参数类别
        :return:
        '''
        saved_params = {}
        if trained_params == 'b':
            saved_params.update({'b1': 0.84, 'a4': 1.00})
        elif trained_params == 'a':
            saved_params.update({'a1': 2431.9, 'a2': 0.52, 'a3': 0.92})
        elif trained_params == 'all':
            saved_params.update({'a1': 2431.9, 'a2': 0.52, 'a3': 0.92, 'b1': 0.84, 'a4': 1.00})
        else:
            logger.debug('No init default params')
        if saved_params:
            self.save_model(file_name=self.file_name, sec_code=sec_code, params=saved_params)

    @timeit
    def train_models(self, sec_code='', model_name='linear', **kwargs):
        '''
        训练一个基本模型
        :param sec_code: 证券代码
        :param model_name: 模型类型选择
        :param file_name: 模型参数路径
        :param kwargs:
        :return:
        '''
        if not self._params:
            self.load_model(self.file_name)
        istar_params = kwargs.get('istar_params') or 'b'
        logger.info(
            "Start train_models with sec_code:{0}, mode_name:{1}, and parameter:{2}".format(sec_code, model_name,
                                                                                            kwargs))
        if istar_params == 'all':
            if sec_code not in self._params:
                self.set_default_params(sec_code=sec_code, trained_params=istar_params)
        else:
            self.set_default_params(sec_code=sec_code, trained_params=istar_params)

        ajusted_mi = kwargs.get('ajusted_mi') or False
        fine_grained = kwargs.get('fine_grained') or False
        features = kwargs.get('features') or self.features
        target_idx = [idx for idx, val in enumerate(self.features) if val in features]
        ret_features = read_features(feature_name=[sec_code, str(kwargs.get('inverval_mins'))])
        init_a4 = kwargs.get('a4') or self._params[sec_code]['a4'] or 1.0
        init_b1 = kwargs.get('b1')
        train_rate = kwargs.get('train_rate') or 0.8

        train_X = []
        backup_features = copy.deepcopy(ret_features)
        for item in ret_features:
            flag = 0
            for sub_item in item:
                if not ('nan' in sub_item):
                    flag = 1
            if flag == 0:
                backup_features.remove(item)
                continue
            train_X.append([format_float(item[idx]) for idx in target_idx])

        train_X = num_pipeline.fit_transform(train_X)
        if ajusted_mi:
            train_Y = [get_istar_label(format_float(item[-2]), format_float(item[-1]),
                                       format_float(item[self.features.index('POV')]),
                                       sec_code=sec_code, b1=init_b1, a4=init_a4, file_name=self.file_name) for item in
                       backup_features]
        else:
            train_Y = [get_market_impact_label(format_float(item[-2]), format_float(item[-1])) for item in
                       backup_features]
        _sgn = lambda x: -1 if x < 0 else 1
        if istar_params == 'b':
            train_X = [[_sgn(item[0]) * math.pow(abs(item[0]), float(init_a4))] for item in train_X]

        model = MlRegModel(model_name=model_name)
        model.build_model()
        model_path = 'models_{0}_{1}_{2}'.format(model_name, sec_code, get_hash_key(features))
        if fine_grained:
            model.best_estimate(train_X, train_Y)
            model.build_model()
            logger.debug(model._best_estimate)
        train_Y = np.array(train_Y)
        train_Y = np.nan_to_num(train_Y)

        train_X, train_Y, val_X, val_Y = split_train_val_dataset(train_X, train_Y, train_rate)
        model.train_model(train_X, train_Y)
        model.save_model(model_path)
        model_param = model.output_model()
        y_predict = model.predict(val_X)
        eval_model = model.eval_model(val_Y, y_predict, ['mse', 'r2_score'])
        logger.debug(
            'results with train_sample:{0}, a4:{1}, b1:{2},ret:{3}'.format(train_rate, init_a4, init_b1, eval_model))
        ret_params = {}
        try:
            if istar_params == 'b':
                a0, a1 = model_param[1], model_param[0][0]
                b = a1 / (a0 + a1)
                ret_params.update({'b1': b, 'a4': init_a4})
            elif istar_params == 'a':
                a2, a3 = model_param[0]
                a1 = model_param[1]
                ret_params.update({'a1': math.exp(a1), 'a2': a2, 'a3': a3})
        except Exception as ex:
            logger.exception('Fail to retrieve model parameters with error:{0}'.format(ex))
        logger.info(
            "Done train_models with sec_code:{0}, mode_name:{1}, and parameter:{2}".format(sec_code, model_name,
                                                                                           kwargs))
        return eval_model, ret_params

    def _train_with_grid_search(self, sec_code='', search_params={}, model_name='linear',
                                features=['LOG_SIGMA', 'LOG_Q_ADV'],
                                ajusted_mi=True, file_name='', inverval_mins=30):
        b1_params = search_params.get('b1') or [0.86]
        a4_params = search_params.get('a4') or [0.50]
        max_score = 0.0
        best_b1, best_a4 = 0.86, 0.50
        logger.info(
            'Start train_with_grid_search for sec_code:{0}, search_parms:{1}, model_name:{2}, features:{3}'.format(
                sec_code, search_params, model_name, features))
        for b1 in b1_params:
            for a4 in a4_params:
                logger.debug('Evaluate b1:{0} and a4:{1}'.format(b1, a4))
                ret_score, ret_param = self.train_models(sec_code=sec_code, model_name=model_name, features=features,
                                                         ajusted_mi=ajusted_mi,
                                                         istar_params='all',
                                                         a4=a4, b1=b1, file_name=file_name, inverval_mins=inverval_mins)
                if ret_score.get('r2_score') > max_score:
                    best_b1 = b1
                    best_a4 = a4
        logger.debug('Best b1:{0} and best a4:{1}'.format(best_b1, best_a4))
        ret_score, ret_param = self.train_models(sec_code=sec_code, model_name=model_name, features=features,
                                                 ajusted_mi=ajusted_mi, istar_params='a',
                                                 a4=best_a4, b1=best_b1, file_name=file_name,
                                                 inverval_mins=inverval_mins)
        ret_param.update({'b1': best_b1, 'a4': best_a4})
        logger.debug(
            'Done train_with_grid_search for sec_code:{0}, search_parms:{1}, model_name:{2}, features:{3}'.format(
                sec_code, search_params, model_name, features))
        return ret_score, ret_param

    def _train_with_istar_opt(self, sec_code='', model_name='istar_opt', features=['LOG_SIGMA', 'LOG_Q_ADV'],
                              file_name='', opt_method='trf', lb=0.0, ub=np.inf, inverval_mins=30):
        logger.info(
            'Start train_with_grid_search for sec_code:{0}, model_name:{1}, features:{2}, opt_method:{3}, lower_bound:{4}, upper_bound:{5}'.format(
                sec_code, model_name, features, opt_method, lb, ub))
        features = features or self.features
        target_idx = [idx for idx, val in enumerate(self.features) if val in features]
        ret_features = read_features(feature_name=[sec_code, str(inverval_mins)])
        train_X = []
        backup_features = copy.deepcopy(ret_features)
        for item in ret_features:
            flag = 0
            for sub_item in item:
                if not ('nan' in sub_item):
                    flag = 1
            if flag == 0:
                backup_features.remove(item)
                continue
            train_X.append([format_float(item[idx]) for idx in target_idx])

        train_X = num_pipeline.fit_transform(train_X)
        train_Y = [get_market_impact_label(format_float(item[-2]), format_float(item[-1])) for item in
                   backup_features]
        model = OptimizeModel(model_name=model_name)
        model.build_model()
        model_path = 'models_{0}_{1}_{2}'.format(model_name, sec_code, get_hash_key(features))
        tmp_Y = [0.0 if item != item else item for item in train_Y]
        mean_y = sum(tmp_Y) / len(tmp_Y)
        train_Y = [mean_y if item != item else item for item in train_Y]
        ret = model.train_model(train_X, train_Y, lb=lb, up=ub, method=opt_method)
        model.save_model(model_path)
        popt, pcov = model.output_model()
        y_predict = model.predict(train_X)
        eval_model = model.eval_model(train_Y, y_predict, ['mse', 'r2_score'])
        logger.debug(eval_model)
        param_keys = ['a1', 'a2', 'a3', 'a4', 'b1']
        ret_params = dict(zip(param_keys, popt))
        return eval_model, ret_params

    def _train_with_nyu_opt(self, sec_code='', model_name='istar_opt', features=[],
                            file_name='', opt_method='trf', lb=0.0, ub=np.inf, interval_mins=30):
        logger.info(
            'Start train_with_grid_search for sec_code:{0}, model_name:{1}, features:{2}, opt_method:{3}, lower_bound:{4}, upper_bound:{5}'.format(
                sec_code, model_name, features, opt_method, lb, ub))
        features = features or self.features
        target_idx = [idx for idx, val in enumerate(self.features) if val in features]
        # ret_features = read_features(feature_name=[sec_code, str(inverval_mins)])
        ret_features = []
        if isinstance(interval_mins, (list, tuple)):
            for min in interval_mins:
                ret_features.extend(read_features(feature_name=[sec_code, str(min)]))
        else:
            ret_features = read_features(feature_name=[sec_code, str(interval_mins)])
        train_X = []
        backup_features = copy.deepcopy(ret_features)
        for item in ret_features:
            flag = 0
            for sub_item in item:
                if not ('nan' in sub_item):
                    flag = 1
            if flag == 0:
                backup_features.remove(item)
                continue
            train_X.append([format_float(item[idx]) for idx in target_idx])
        _num_pipeline = get_pipeline(f_type='num')
        _pipeline_key = get_hash_key(features + [sec_code])
        train_X = _num_pipeline.fit_transform(train_X)
        save_pipeline(_num_pipeline, 'feature_pipeline_{0}'.format(_pipeline_key))
        train_Y = [get_market_impact_label(format_float(item[-2]), format_float(item[-1])) for item in
                   backup_features]
        model = OptimizeModel(model_name=model_name)
        model.build_model()
        model_path = 'models_{0}_{1}_{2}'.format(model_name, sec_code, get_hash_key(features))
        tmp_Y = [0.0 if item != item else item for item in train_Y]
        mean_y = sum(tmp_Y) / len(tmp_Y)
        train_Y = [mean_y if item != item else item for item in train_Y]
        ret = model.train_model(train_X, train_Y, lb=lb, up=ub, method=opt_method)
        model.save_model(model_path)
        popt, pcov = model.output_model()
        y_predict = model.predict(train_X)
        eval_model = model.eval_model(train_Y, y_predict, ['mse', 'r2_score'])
        logger.debug(eval_model)
        param_keys = ['alpha', 'beta', 'perm_param', 'tmp_param']
        ret_params = dict(zip(param_keys, popt))
        return eval_model, ret_params

    def _train_with_dnn(self, sec_code='', model_name='tf_dnn', features=[],
                        file_name='', interval_mins=30):
        features = features or self.features
        if not features:
            raise ValueError('empty feature is not allowed for training with dnn')
        acc_features = config['market_impact']['acc_features'].split(',')
        target_idx = [idx for idx, val in enumerate(self.features) if val in features]
        acc_idx = [idx for idx, val in enumerate(self.features) if val in acc_features]
        ret_features = []
        if isinstance(interval_mins, (list, tuple)):
            for min in interval_mins:
                ret_features.extend(read_features(feature_name=[sec_code, str(min)]))
        else:
            ret_features = read_features(feature_name=[sec_code, str(interval_mins)])
        train_X = []
        acc_X = []
        backup_features = copy.deepcopy(ret_features)
        for item in ret_features:
            flag = 0
            for sub_item in item:
                if not ('nan' in sub_item):
                    flag = 1
            if flag == 0:
                backup_features.remove(item)
                continue
            train_X.append([format_float(item[idx]) for idx in target_idx])
            acc_X.append([format_float(item[idx]) for idx in acc_idx])
        train_X = feature_preprocessing(train_X)
        acc_X = feature_preprocessing(acc_X)
        train_Y = np.array([[get_market_impact_label(format_float(item[-2]), format_float(item[-1]))] for item in
                            backup_features])
        model = TFRegModel()
        model.build_model(x_shape=train_X.shape, acc_shape=acc_X.shape,
                          n_hidden_layers=int(config['tf_models']['n_hidden_lays']), save_prefix=sec_code)
        model.train_model(np.array(train_X), np.array(train_Y), np.array(acc_X), int(config['tf_models']['epoch']),
                          int(config['tf_models']['batch']), model_name)
        model.load_model(model_name)
        output = model.predict(train_X, acc_X)
        y_predict = [item[0] for item in output[0]]
        eval_model = model.eval_model([item[0] for item in train_Y], y_predict, ['mse', 'r2_score'])
        logger.debug(eval_model)
        return eval_model, {}
