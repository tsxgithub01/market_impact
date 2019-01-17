# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : ml_reg_models.py


from utils.utils import get_config
from sklearn import linear_model
from model_processing.models import Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


config = get_config()


class Ml_Reg_Model(Model):
    def __init__(self, model_name=None):
        self.model_name = model_name
        self._best_estimate = {}

    def build_model(self, **kwargs):
        regs = {
            'linear': linear_model.LinearRegression(),
            'gbdt': GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, min_samples_split=2,
                                              min_samples_leaf=1, init=None, random_state=None,
                                              max_features=None, max_depth=None,
                                              alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False),
            # max_features: 划分时候考虑的最大特征数，默认是‘None’，考虑所有的特征数；浮点数表示特征的百分比，
            # 或者'log2''sqrt'/'auto'
            'ridge': Ridge(alpha=.5),  # 正则化的线性模型，可以解决过拟合情况，L2范数惩罚项
            'lasso': Lasso(alpha=.5),  # 正则化的线性模型，可以解决过拟合情况，L1范数惩罚项
            'linear_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
                              kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            'poly_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=0.5, epsilon=0.2, gamma='auto',
                            kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            'rbf_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
                           kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            'sigmoid_svr': SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
                               kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
            # kernel values: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
            'decision_tree': DecisionTreeRegressor(max_depth=5),
            'random_forest': RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42),
            'linear_nnls': linear_model.LinearRegression(),
        }
        self.features = kwargs.get('features')
        param_grids = kwargs.get('super_params') or self._best_estimate
        self.model = regs.get(self.model_name)

    def train_model(self, train_X=[], train_Y=[], **kwargs):
        sample_weights = kwargs.get('sample_weights')
        if self.model_name == 'linear_nnls':
            self.fit_linear_nnls(train_X, train_Y, sample_weight=sample_weights)
            scores = 0.0
        else:
            self.model.fit(train_X, train_Y, sample_weights)
            scores = cross_val_score(self.model, train_X, train_Y, cv=int(config['ml_reg_model']['cv']),
                                     scoring=make_scorer(mean_squared_error))
            print("Mean squared error 67%%: %0.5f -  %0.5f" % (
                scores.mean() - scores.std() * 3, scores.mean() + scores.std() * 3))
        return scores

    def output_model(self, path=None):
        if self.model_name and 'linear' in self.model_name.lower():
            coef, intercept = self.model.coef_, self.model.intercept_
            return coef, intercept

    def best_estimate(self, train_X, train_Y):
        if self.model_name == 'gbdt':
            # TODO add logics for n_estimators and learning_rate
            param_grids = [
                {'n_estimators': range(98, 105),
                 'subsample': [0.5, 0.6, 0.7],
                 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                 }
            ]
            cv = int(config['ml_reg_model']['cv'])
            self.fine_grained(param_grids, cv, 'neg_mean_squared_error', train_X, train_Y)

