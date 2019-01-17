# -*- coding: utf-8 -*-
# @time      : 2019/1/17 14:11
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : sci_optimize_models.py


from model_processing.models import Model
from utils.utils import get_config
from scipy.optimize import curve_fit
import numpy as np

config = get_config()


class Optimize_Model(Model):
    def __init__(self, model_name='istar'):
        self.model_name = model_name
        self._popt = None
        self._pcov = None

    def _istar_cal(self, x_data, a1, a2, a3, a4, b1):
        '''
        x_data:[s,adv,pov, sigma]
        '''
        s = x_data[:, 0]
        adv = x_data[:, 1]
        pov = x_data[:, 2]
        sigma = x_data[:, 3]
        total_len = s.shape[0]
        ra2 = np.array([float(a2)] * total_len)
        ra3 = np.array([float(a3)] * total_len)
        ra4 = np.array([float(a4)] * total_len)
        istar = np.exp((s / adv), ra2) * a1 * np.exp(sigma, ra3)
        adj_pov = np.exp(pov, ra4)
        mi = b1 * istar * adj_pov + (1 - b1) * istar
        return mi

    def build_model(self, **kwargs):
        pass

    def train_model(self, train_X=[], train_Y=[], **kwargs):
        '''
        method notes ref to:https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        :param train_X:
        :param train_Y:
        :param kwargs:
        :return:
        '''
        lb = kwargs.get('lb') or -np.inf
        up = kwargs.get('up') or np.inf
        method = kwargs.get('method') or 'lm'
        popt, pcov = curve_fit(self._istar_cal, train_X, train_Y, check_finite=kwargs.get('check_finite'),
                               bounds=[lb, up], method=method)
        self._popt = popt
        self._pcov = pcov

    def eval_model(self, y_true, y_pred, metrics):
        '''

        :param y_true:
        :param y_pred:
        :param metrics: []
        :return:
        '''
        pass

    def output_model(self):
        pass

if __name__ == '__main__':
    m = Optimize_Model()
    import numpy as np
    x_data = np.random.random(size=(10,4))
    y_data = m._istar_cal(x_data, 0.5, 0.5, 0.5, 0.5,0.8)
    m.train_model(x_data, y_data, lb=0.0, up=np.inf, method='trf')
    print(m._pcov)
    print(m._popt)