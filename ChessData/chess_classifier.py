#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/CodeWarehouse/UCIDataset/ChessData/chess_classifier.py
# Project: /Users/guchenghao/CodeWarehouse/UCIDataset/ChessData
# Created Date: Thursday, August 30th 2018, 12:51:29 pm
# Author: Richard Gu
# -----
# Last Modified: Fri Aug 31 2018
# Modified By: Richard Gu
# -----
# Copyright (c) 2018 University
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ds_train = pd.read_csv(
    '/Users/guchenghao/CodeWarehouse/UCIDataset/ChessData/dataset/krkopt.data', header=None, names=['WKf', 'WKr', 'WRf', 'WRr', 'BKf', 'BKr','depth-of-win'])

print('数据集的维度为: {0}'.format(ds_train.shape))

print(ds_train.info())

Y = LabelEncoder().fit_transform(ds_train['depth-of-win']) # ! sklearn对Y值进行标签编码处理
Y = np_utils.to_categorical(Y, num_classes=18)  # ! 将Y值转化为神经网络可处理的格式
# print(Y)

train_data = ds_train.drop('depth-of-win', axis=1)
numerical_features = train_data.select_dtypes(exclude=["object"]).columns
categorical_features = train_data.select_dtypes(include=["object"]).columns

cate_data = pd.get_dummies(train_data[categorical_features])

new_train_data = pd.concat([cate_data, train_data[numerical_features]], axis=1)
print('编码后的训练数据: {0}'.format(new_train_data.shape))
print(new_train_data)

X_train, X_test, Y_train, Y_test = train_test_split(
    new_train_data, Y, test_size=0.2, random_state=56)


# def get_train_scores(model):  # ! 计算训练集上的得分

#     acc = cross_val_score(model, X_train, Y_train, cv=10,
#                           scoring='accuracy').mean()

#     return acc


# def get_test_scores(model):  # ! 计算训练集上的得分

#     acc = cross_val_score(model, X_test, Y_test, cv=10,
#                           scoring='accuracy').mean()

#     return acc


# gbdt_model = GradientBoostingClassifier(n_estimators=200)

# print(get_train_scores(gbdt_model))
# print(get_test_scores(gbdt_model))

# Rf_model = RandomForestClassifier(n_estimators=200)

# print(get_train_scores(Rf_model))
# print(get_test_scores(Rf_model))


# svc_model = LinearSVC(class_weight='balanced')
# print(get_train_scores(svc_model))
# print(get_test_scores(svc_model))


epochs = 30

inputs = Input(shape=(23, ))

dense_1 = Dense(64, activation='relu')(inputs)
# dense_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(128, activation='relu')(dense_1)
# dense_2 = Dropout(0.2)(dense_2)
dense_3 = Dense(256, activation='relu')(dense_2)
dense_3 = Dropout(0.2)(dense_3)
outputs = Dense(18, activation='softmax')(dense_3)

chess_model = Model(inputs=inputs, outputs=outputs)
chess_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
chess_model.summary()

history = chess_model.fit(X_train, Y_train, batch_size=64, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))


X_epoch = np.arange(0, epochs, 1)
plt.plot(X_epoch, history.history['acc'], label='acc')
plt.plot(X_epoch, history.history['val_acc'], label='val_acc')
plt.legend(loc='best')
plt.show()
