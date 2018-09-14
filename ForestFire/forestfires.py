#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\UCIDataset\ForestFire\forestfires.py
# Project: d:\CodeWareHouse\UCIDataset\ForestFire
# Created Date: Thursday, August 9th 2018, 7:07:31 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Sunday, 12th August 2018 3:22:03 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from scipy.stats import skew
from scipy.stats import norm
from scipy import stats


firesData = pd.read_csv(
    'D:\\CodeWareHouse\\UCIDataset\\ForestFire\\dataset\\forestfires.csv')

print('数据集的维度: {0}'.format(firesData.shape))


fires_train = firesData.drop('area', axis=1)
Y = firesData['area']

print(firesData.describe())

# ! 判断area值是否为正态分布
sns.distplot(Y, fit=norm)
fig = plt.figure()
res = stats.probplot(Y, plot=plt)
plt.show()

Y = np.log1p(Y)

fires_train['month'] = fires_train.month.replace(
    {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})

fires_train['day'] = fires_train.day.replace(
    {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7})

print("找出与'area'项最为的相关的特征项: ")
corr = firesData.corr()
# ! 按照'area'相关系数的大小从大到小来排列
corr.sort_values(["area"], ascending=False, inplace=True)
print(corr.area)

fires_train["temp-s2"] = fires_train["temp"] ** 2
fires_train["temp-s3"] = fires_train["temp"] ** 3
fires_train["temp-Sq"] = np.sqrt(fires_train["temp"])
fires_train["DMC-2"] = fires_train["DMC"] ** 2
fires_train["DMC-3"] = fires_train["DMC"] ** 3
fires_train["DMC-Sq"] = np.sqrt(fires_train["DMC"])
fires_train["X-2"] = fires_train["X"] ** 2
fires_train["X-3"] = fires_train["X"] ** 3
fires_train["X-Sq"] = np.sqrt(fires_train["X"])
fires_train["Y-2"] = fires_train["Y"] ** 2
fires_train["Y-3"] = fires_train["Y"] ** 3
fires_train["Y-Sq"] = np.sqrt(fires_train["Y"])
fires_train["DC-2"] = fires_train["DC"] ** 2
fires_train["DC-3"] = fires_train["DC"] ** 3
fires_train["DC-Sq"] = np.sqrt(fires_train["DC"])
# ! 将feature分成两类进行处理
# fires_cate = pd.concat([fires_train['month'], fires_train['day']], axis=1)
fires_num = fires_train.drop(['month', 'day', 'rain', 'RH'], axis=1)

# fires_cate = pd.get_dummies(fires_cate)  # ! 进行one-hot编码
# print(fires_cate)

skewness = fires_num.apply(lambda x: skew(x))
print(skewness)
skewness = skewness[abs(skewness) > 0.5]  # ! 偏态值大于0.5，全部是正偏态
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
# ! 将被定为正偏态的值进行数据转换
fires_num[skewed_features] = np.log1p(fires_num[skewed_features])

# train_data = pd.concat([fires_cate, fires_num], axis=1)
# print(train_data)

X_train, X_test, Y_train, Y_test = train_test_split(
    fires_num, Y, test_size=0.3, random_state=44)

print(fires_num.info())
numerical_features = fires_num.select_dtypes(exclude=["object"]).columns
# ! 数值型feature进行归一化
min_max_sc = MinMaxScaler()
X_train.loc[:, numerical_features] = min_max_sc.fit_transform(
    X_train.loc[:, numerical_features])
X_test.loc[:, numerical_features] = min_max_sc.transform(
    X_test.loc[:, numerical_features])


def get_train_scores(model):  # ! 计算训练集上的得分

    mse = - cross_val_score(model, X_train, Y_train, cv=10,
                            scoring='neg_mean_squared_error').mean()

    return mse


def get_test_scores(model):  # ! 计算验证集上的得分

    mse = - cross_val_score(model, X_test, Y_test, cv=10,
                            scoring='neg_mean_squared_error').mean()

    return mse


# ! alpha和estimators可调，其它参数视情况而定
Ada_model = AdaBoostRegressor(n_estimators=200)

print(get_train_scores(Ada_model))
print(get_test_scores(Ada_model))


# gbdt_model = GradientBoostingRegressor(n_estimators=200)

# print(get_train_scores(gbdt_model))
# print(get_test_scores(gbdt_model))


Rf_model = RandomForestRegressor(n_estimators=200)

print(get_train_scores(Rf_model))
print(get_test_scores(Rf_model))


ridge_model = Ridge(alpha=0.5)

print(get_train_scores(ridge_model))
print(get_test_scores(ridge_model))
