#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\UCIDataset\Abalone\abaline_classifier.py
# Project: d:\CodeWareHouse\UCIDataset\Abalone
# Created Date: Sunday, August 12th 2018, 2:58:55 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Sunday, 12th August 2018 4:13:59 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder


abalone_data = pd.read_csv(
    'D:\\CodeWareHouse\\UCIDataset\\Abalone\\dataset\\abalone.data', header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'])

abalone_data.to_csv(
    'D:\\CodeWareHouse\\UCIDataset\\Abalone\\dataset\\abalone.csv')

print(abalone_data)

Y = LabelEncoder().fit_transform(abalone_data['Rings'])
print(Y)

abalone_train = abalone_data.drop('Rings', axis=1)

feature_num = abalone_train.drop('Sex', axis=1)
feature_cate = abalone_train['Sex']

feature_cate = pd.get_dummies(feature_cate)

train_data = pd.concat([feature_cate, feature_num], axis=1)
print(train_data)


X_train, X_test, Y_train, Y_test = train_test_split(
    train_data, Y, test_size=0.2, random_state=0)


def get_train_score(model):
    train_score = cross_val_score(
        model, X_train, Y_train, cv=10, scoring='accuracy').mean()

    return train_score


def get_test_score(model):
    test_score = cross_val_score(
        model, X_test, Y_test, cv=10, scoring='accuracy').mean()

    return test_score


# Ada_model = AdaBoostClassifier(n_estimators=150)

# print(get_train_score(Ada_model))
# print(get_test_score(Ada_model))

# rf_model = RandomForestClassifier(n_estimators=150)

# print(get_train_score(rf_model))
# print(get_test_score(rf_model))


# gbdt_model = GradientBoostingClassifier(n_estimators=120)

# print(get_train_score(gbdt_model))
# print(get_test_score(gbdt_model))

svc_model = LinearSVC(class_weight='balanced')
print(get_train_score(svc_model))
print(get_test_score(svc_model))
