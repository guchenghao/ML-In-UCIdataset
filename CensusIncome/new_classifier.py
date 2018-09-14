#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/CodeWarehouse/UCIDataset/CensusIncome/new_classifier.py
# Project: /Users/guchenghao/CodeWarehouse/UCIDataset/CensusIncome
# Created Date: Monday, September 10th 2018, 11:48:44 am
# Author: Richard Gu
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2018 University
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Dropout
from keras import backend as K
from keras.models import Model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score


train_data = pd.read_csv('/Users/guchenghao/CodeWarehouse/UCIDataset/CensusIncome/dataset/adult.data', names=[
                         'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'])

test_data = pd.read_csv(
    '/Users/guchenghao/CodeWarehouse/UCIDataset/CensusIncome/dataset/adult.test', names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'])

print('训练数据集的维度: {0}'.format(train_data.shape))
print('测试数据集的维度: {0}'.format(test_data.shape))


print('训练数据的描述统计量: {0}'.format(train_data.describe()))


print('训练数据的属性信息: {0}'.format(train_data.info()))

if ' ?' in list(train_data['income']):
    train_data['income'] = train_data['income'].replace({' ?': None})
    train_data.dropna(inplace=True)

y = LabelEncoder().fit_transform(train_data['income'])
train_X = train_data.drop('income', axis=1)

# ! 划分数值型特征与类别型特征
numerical_features = train_X.select_dtypes(exclude=["object"]).columns
categorical_features = train_X.select_dtypes(include=["object"]).columns
notNull_features = []
hasNull_features = []

for item in list(numerical_features):
    if ' ?' in list(train_X[item]):
        hasNull_features.append(item)
        # ! train_X[item][train_X[item] == ' ?'] = None
        train_X[item] = train_X[item].replace({' ?': None})
        # train_X[item].fillna(train_X[item].mean(), inplace=True)
    else:
        notNull_features.append(item)


for item in list(categorical_features):
    if ' ?' in list(train_X[item]):
        hasNull_features.append(item)
        train_X[item] = train_X[item].replace({' ?': None})
        # train_X[item].fillna(train_X[item].mode().iloc[0], inplace=True)
    else:
        notNull_features.append(item)

print("无缺失值的属性列表: {0}".format(notNull_features))
print('\n')
print("有缺失值的属性列表: {0}".format(hasNull_features))


# train_X = pd.get_dummies(train_X)
print(train_X['workclass'])
isnulldata = train_X[pd.isnull(train_X['workclass'])]
notnulldata = train_X[pd.notnull(train_X['workclass'])]
workclass_classifier = GradientBoostingClassifier()
workclass_classifier.fit(pd.get_dummies(notnulldata[notNull_features]).values, notnulldata['workclass'].values)
predictions = workclass_classifier.predict(
    pd.get_dummies(isnulldata[notNull_features]).values)
train_X.workclass[pd.isnull(train_X['workclass'])] = predictions
print(train_X['workclass'])


print(train_X['occupation'])
isnulldata = train_X[pd.isnull(train_X['occupation'])]
notnulldata = train_X[pd.notnull(train_X['occupation'])]
occupation_classifier = GradientBoostingClassifier()
occupation_classifier.fit(pd.get_dummies(
    notnulldata[notNull_features]).values, notnulldata['occupation'].values)
predictions = occupation_classifier.predict(
    pd.get_dummies(isnulldata[notNull_features]).values)
train_X.occupation[pd.isnull(train_X['occupation'])] = predictions
print(train_X['occupation'])

# print(train_X['native_country'])
# isnulldata = train_X[pd.isnull(train_X['native_country'])]
# notnulldata = train_X[pd.notnull(train_X['native_country'])]
# classifier = GradientBoostingClassifier()
# classifier.fit(pd.get_dummies(
#     notnulldata[notNull_features]).values, notnulldata['native_country'].values)
# predictions = classifier.predict(
#     pd.get_dummies(isnulldata[notNull_features]).values)
# train_X.native_country[pd.isnull(train_X['native_country'])] = predictions
# print(train_X['native_country'])


print('数据清洗后的数据: {0}'.format(train_X))
# train_X.to_csv(
#     '/Users/guchenghao/CodeWarehouse/UCIDataset/CensusIncome/dataset/adult_new.csv', index=False)


categorical_train_data = pd.get_dummies(train_X[categorical_features])
print(train_X[numerical_features].describe())
# numerical_train_data = pd.DataFrame(MinMaxScaler().fit_transform(
#     train_X[numerical_features]), columns=numerical_features)
# print(numerical_train_data)
# print(list(categorical_train_data.columns))

new_train_X = pd.concat(
    [categorical_train_data, train_X[numerical_features]], axis=1)
print(new_train_X)

X_train, X_test, Y_train, Y_test = train_test_split(
    new_train_X, y, test_size=0.2, random_state=66)


def get_train_scores(model):  # ! 计算训练集上的得分

    acc = cross_val_score(model, X_train, Y_train, cv=10,
                          scoring='accuracy').mean()

    return acc


def get_test_scores(model):  # ! 计算训练集上的得分

    acc = cross_val_score(model, X_test, Y_test, cv=10,
                          scoring='accuracy').mean()

    return acc


gbdt_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2)

print(get_train_scores(gbdt_model))
print(get_test_scores(gbdt_model))
