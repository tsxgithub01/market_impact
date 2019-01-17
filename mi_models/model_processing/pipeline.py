# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : pipeline.py

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
