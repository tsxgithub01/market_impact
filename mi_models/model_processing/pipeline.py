# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi
# @file      : pipeline.py

import pickle
import os
from ..utils.utils import get_parent_dir
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion

num_pipeline = Pipeline([
    ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('label_binarizer', LabelBinarizer())
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])


def get_pipeline(f_name=None, f_type='num'):
    if f_name:
        feature_trained_path = os.path.join(get_parent_dir(), 'data', 'models', f_name)
        with open(feature_trained_path, 'rb') as in_file:
            return pickle.load(in_file)
    else:
        return {
            'num': num_pipeline,
            'cat': cat_pipeline
        }.get(f_type.lower())


def save_pipeline(p_obj=None, f_name=None):
    if not p_obj or not f_name:
        raise ValueError('pipeline object and save file name should be nonempty')
    feature_trained_path = os.path.join(get_parent_dir(), 'data', 'models', f_name)
    with open(feature_trained_path, 'wb') as out_file:
        pickle.dump(p_obj, out_file)