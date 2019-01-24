# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : models.py

import os
import numpy as np
from abc import abstractmethod
from sklearn.externals import joblib
from utils.utils import get_config
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from logger import Logger
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.base import _rescale_data
from sklearn.utils.validation import check_X_y
from scipy import linalg
import scipy.sparse as sp
from sklearn.externals.joblib.parallel import Parallel
from sklearn.externals.joblib.parallel import delayed
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
# from quant_models.utils.utils import get_parent_dir
from ..utils.utils import get_parent_dir

# from scipy.sparse.linalg.isolve.lsqr import sparse_lsqr

logger = Logger('log.txt', 'INFO', __name__).get_log()

config = get_config()


class Model(object):
    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self._best_estimate = None

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    def eval_model(self, y_true, y_pred, metrics):
        '''

        :param y_true:
        :param y_pred:
        :param metrics: []
        :return:
        '''
        metric_map = {
            'mse': mean_squared_error,
            'r2_score': r2_score,
            'precision': precision_score
        }
        ret = [metric_map.get(metric)(y_true, y_pred) for metric in metrics]
        return dict(zip(metrics, ret))

    @abstractmethod
    def output_model(self, path=None):
        pass

    @abstractmethod
    def train_model(self, train_X=[], train_Y=[], **kwargs):
        pass

    def save_model(self, model_name):
        model_path = os.path.join(get_parent_dir(), 'data','models', model_name)
        # p =  '{0}/data/models/{1}.m'.format(config['constants']['root'], model_path)
        #TODO remove hardcode
        # model_path = 'E:/pycharm/algo_trading/quant_models/quant_models/data/models/{0}'.format(model_name)
        joblib.dump(self.model, model_path)

    def load_model(self, model_name):
        #TODO remove hardcode
        model_path = os.path.join(get_parent_dir(), 'data', 'models', '{0}'.format(model_name))
        # p = '{0}models/{1}.m'.format(config['constants']['root'], model_name)
        # model_path = 'E:/pycharm/algo_trading/quant_models/quant_models/data/models/{0}'.format(model_name)
        self.model = joblib.load(model_path)

    def train_features(self, train_X, train_Y, predict=True, threshold=0.15):
        pass

    def predict(self, input_X):
        return self.model.predict(input_X)

    def fit_linear_nnls(self, X, y, sample_weight=None):
        if not isinstance(self.model, LinearRegression):
            raise ValueError('Model is not linearRegression, could not call fit for linear nnls')
        n_jobs_ = self.model.n_jobs
        self.model.coef_ = []
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)

        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self.model._preprocess_data(
            X, y, fit_intercept=self.model.fit_intercept, normalize=self.model.normalize,
            copy=self.model.copy_X, sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if sp.issparse(X):
            if y.ndim < 2:
                # out = sparse_lsqr(X, y)
                out = lsq_linear(X, y, bounds=(0, np.Inf))
                self.model.coef_ = out[0]
                self.model._residues = out[3]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(lsq_linear)(X, y[:, j].ravel())
                    for j in range(y.shape[1]))
                self.model.coef_ = np.vstack(out[0] for out in outs)
                self.model._residues = np.vstack(out[3] for out in outs)
        else:
            # self.model.coef_, self.model.cost_, self.model.fun_, self.model.optimality_, self.model.active_mask_,
            # self.model.nit_, self.model.status_, self.model.message_, self.model.success_\
            out = lsq_linear(X, y, bounds=(0, np.Inf))
            self.model.coef_ = out.x
            self.model.coef_ = self.model.coef_.T

        if y.ndim == 1:
            self.model.coef_ = np.ravel(self.model.coef_)
        self.model._set_intercept(X_offset, y_offset, X_scale)
        return self.model

    def fine_grained(self, param_grids=[], cv=5, scoring='neg_mean_squared_error', train_X=[], train_Y=[]):
        logger.info('Start fine_grained search')
        grid_search = GridSearchCV(self.model, param_grids, cv=cv, scoring=scoring)
        logger.info('Done fine_frained search')
        best_model = grid_search.fit(train_X, train_Y)
        self.model = best_model
        self._best_estimate = grid_search.best_estimator_
        return grid_search.best_estimator_
