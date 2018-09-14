#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\UCIDataset\CarEvaluation\carEvaluation.py
# Project: d:\CodeWareHouse\UCIDataset\CarEvaluation
# Created Date: Wednesday, August 8th 2018, 6:44:28 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Thursday, 9th August 2018 7:52:36 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


carData = pd.read_csv('D:\\CodeWareHouse\\UCIDataset\\CarEvaluation\\dataset\\car.data',
                      header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'values'])

print("数据的维度：{0}".format(carData.shape))

Y = carData['values']

carData_train = carData.drop("values", axis=1)

# ! 对categorical类型的数据进行编码
carData_train["buying"] = carData_train.buying.replace(
    {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4})

carData_train["maint"] = carData_train.maint.replace(
    {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4})

carData_train["doors"] = carData_train.doors.replace(
    {2: 1, 3: 2, 4: 3, '5more': 4})

carData_train["persons"] = carData_train.persons.replace(
    {2: 1, 4: 2, 'more': 3})

carData_train["safety"] = carData_train.safety.replace(
    {'low': 1, 'med': 2, 'high': 3})

carData_train["lug_boot"] = carData_train.lug_boot.replace(
    {'small': 1, 'med': 2, 'big': 3})


Y = LabelEncoder().fit_transform(Y)

# carData_train = pd.get_dummies(carData_train)  # ! 进行one-hot编码
# print(carData_train)

X_train, X_test, Y_train, Y_test = train_test_split(
    carData_train, Y, test_size=0.2, random_state=44)


def get_train_score(model):
    train_score = cross_val_score(
        model, X_train, Y_train, cv=10, scoring='accuracy').mean()

    return train_score


def get_test_score(model):
    test_score = cross_val_score(
        model, X_test, Y_test, cv=10, scoring='accuracy').mean()

    return test_score


Ada_model = AdaBoostClassifier(n_estimators=200)

print(get_train_score(Ada_model))
print(get_test_score(Ada_model))


gbdt_model = GradientBoostingClassifier(n_estimators=120)

print(get_train_score(gbdt_model))
print(get_test_score(gbdt_model))

gbdt_model.fit(X_train, Y_train)

predictions = gbdt_model.predict(X_test)

print(predictions)
print(Y_test)

print(np.sum((predictions == Y_test)) / len(Y_test))  # ! 计算test的准确率
