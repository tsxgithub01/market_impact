# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : ml_clfs_models.py

import os
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from utils.utils import get_config
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from model_processing.models import Model
from utils.utils import get_parent_dir


config = get_config()


class Ml_Clf_Model(Model):
    def __init__(self, model_name='decision_tree'):
        self.model_name = model_name

    def build_model(self, **kwargs):
        clfs = {
            'svm': svm.SVC(),
            'linear_svc': LinearSVC(C=0.01, penalty="l1", dual=False, class_weight='balanced'),
            'decision_tree': tree.DecisionTreeClassifier(max_depth=int(config['ml_clf_model']['maxdepth']),
                                                         min_samples_leaf=float(
                                                             config['ml_clf_model']['min_samples_leaf']),
                                                         class_weight=config['ml_clf_model']['class_weight'],
                                                         criterion=config['ml_clf_model']['criterion']),
            'naive_gaussian': naive_bayes.GaussianNB(),
            # 'naive_mul':naive_bayes.MultinomialNB(),
            'K_neighbor': neighbors.KNeighborsClassifier(),
            'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(),
                                             max_samples=config['ml_clf_model']['max_samples'],
                                             max_features=config['ml_clf_model']['max_features']),
            'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(),
                                              max_samples=config['ml_clf_model']['max_samples'],
                                              max_features=config['ml_clf_model']['max_features']),
            'random_forest': RandomForestClassifier(n_estimators=int(config['ml_clf_model']['n_estimators'])),
            'adaboost': AdaBoostClassifier(n_estimators=int(config['ml_clf_model']['n_estimators'])),
            'gradient_boost': GradientBoostingClassifier(n_estimators=config['ml_clf_model']['n_estimators'],
                                                         learning_rate=config['ml_clf_model']['learning_rate'],
                                                         max_depth=config['ml_clf_model']['max_depth'],
                                                         random_state=config['ml_clf_model']['random_state'])
        }
        self.model = clfs.get(self.model_name)

    def train_model(self, train_X=[], train_Y=[], **kwargs):
        self.model.fit(train_X, train_Y)

    def train_features(self, train_X, train_Y, predict=True, threshold=0.15):
        train_model = self.model.fit(train_X, train_Y)
        feature_model = SelectFromModel(train_model, prefit=predict, threshold=threshold)
        return feature_model.transform(train_X)

    def output_model(self):
        if self.model_name == 'decision_tree':
            out_file = os.path.join(get_parent_dir(), 'data', 'tree.dot')
            tree.export_graphviz(self.model, out_file=out_file)
