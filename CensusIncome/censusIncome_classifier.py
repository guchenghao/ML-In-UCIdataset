import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Dropout
from keras import backend as K
from keras.models import Model
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score


train_data = pd.read_csv('/Users/guchenghao/CodeWarehouse/UCIDataset/CensusIncome/dataset/adult.data', names=[
                         'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

test_data = pd.read_csv(
    '/Users/guchenghao/CodeWarehouse/UCIDataset/CensusIncome/dataset/adult.test', names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

print('训练数据集的维度: {0}'.format(train_data.shape))
print('测试数据集的维度: {0}'.format(test_data.shape))


print(train_data.describe())


print(train_data.info())

if ' ?' in list(train_data['income']):
    train_data['income'] = train_data['income'].replace({' ?': None})
    train_data['income'].fillna(train_data['income'].mode().iloc[0], inplace=True)

y = LabelEncoder().fit_transform(train_data['income'])
train_X = train_data.drop('income', axis=1)

# ! 划分数值型特征与类别型特征
numerical_features = train_X.select_dtypes(exclude=["object"]).columns
categorical_features = train_X.select_dtypes(include=["object"]).columns
print(list(categorical_features))

for item in list(numerical_features):
    if ' ?' in list(train_X[item]):
        # ! train_X[item][train_X[item] == ' ?'] = None
        train_X[item] = train_X[item].replace({' ?': None})
        train_X[item].fillna(train_X[item].mean(), inplace=True)

for item in list(categorical_features):
    if ' ?' in list(train_X[item]):
        train_X[item] = train_X[item].replace({' ?': None})
        train_X[item].fillna(train_X[item].mode().iloc[0], inplace=True)
print(train_X['workclass'])


# plt.scatter(train_X['hours-per-week'], train_X['capital-gain'], c=y)
# plt.title('test')
# plt.show()

categorical_train_data = pd.get_dummies(train_X[categorical_features])
print(train_X[numerical_features].describe())
numerical_train_data = pd.DataFrame(MinMaxScaler().fit_transform(
    train_X[numerical_features]), columns=numerical_features)
print(numerical_train_data)
# print(list(categorical_train_data.columns))

new_train_X = pd.concat([categorical_train_data, numerical_train_data], axis=1)
print(new_train_X)

X_train, X_test, Y_train, Y_test = train_test_split(
    new_train_X, y, test_size=0.3, random_state=44)


def get_train_scores(model):  # ! 计算训练集上的得分

    acc = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy').mean()

    return acc


def get_test_scores(model):  # ! 计算训练集上的得分

    acc = cross_val_score(model, X_test, Y_test, cv=10,
                          scoring='accuracy').mean()

    return acc


gbdt_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2)

print(get_train_scores(gbdt_model))
print(get_test_scores(gbdt_model))


# Ada_model = AdaBoostClassifier(n_estimators=200)

# print(get_train_scores(Ada_model))
# print(get_test_scores(Ada_model))

# Lg_model = LogisticRegression(solver='saga', max_iter=300)

# print(get_train_scores(Lg_model))
# print(get_test_scores(Lg_model))

epochs = 50

inputs= Input(shape=(105, ))

dense_1 = Dense(64, activation='relu')(inputs)
dense_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(128, activation='relu')(dense_1)
dense_2 = Dropout(0.2)(dense_2)
dense_3 = Dense(256, activation='relu')(dense_2)
dense_3 = Dropout(0.2)(dense_3)
outputs = Dense(1, activation='sigmoid')(dense_3)

censusIncome_model = Model(inputs=inputs, outputs=outputs)
censusIncome_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = censusIncome_model.fit(
    X_train, Y_train, epochs=epochs, batch_size=128, verbose=1, validation_data=(X_test, Y_test))

censusIncome_model.summary()

X_epoch = np.arange(0, epochs, 1)
plt.plot(X_epoch, history.history['acc'], label='acc')
plt.plot(X_epoch, history.history['val_acc'], label='val_acc')
plt.legend(loc='best')
plt.show()
