# -*- coding: utf-8 -*-
"""
通常遇到缺值的情况，我们会有几种常见的处理方式
1.缺省样本比例极高，可舍弃，可能作为特征反而加入noise
2.缺省样本比例适中，且非连续值特征属性，那就把NaN作为一个新类别，加到类别特征中
3.如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，
  我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import pandas as pd
import numpy as np


# 使用RandomForestClassifier 填补缺失年龄属性
def set_missingAge(data, label="train", rfr=[]):
	# 我想查看Age为nan是不是Cabin一定为nan
	# data_train.Cabin[pd.isnull(data_train.Age)].count()
	# 把已知的数据类型特征取出丢进Random Forest Regressor
	# data = data.copy()
	age_df = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
	# 乘客分为已知年龄和未知年龄两部分
	unknow_age = age_df[age_df.Age.isnull()].values
	if label == "train":
		know_age = age_df[age_df.Age.notnull()].values
		# y - 目标年龄
		y = know_age[:, 0]
		# x - 特征属性值
		X = know_age[:, 1:]
		# fit到RandomForestRegressor之中
		rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
		rfr.fit(X, y)
	# 用得到的模型进行未知年龄结果预测
	predictedAge = rfr.predict(unknow_age[:, 1:])
	# 用得到的预测结果填补原缺失数据 df.loc(行标签, 列标签)
	data.loc[(data.Age.isnull()), 'Age'] = predictedAge
	if label == "train":
		return data, rfr
	else:
		return data


# 随机森林方法可能一般化，考虑到生存概率可能会偏向老人与小孩，所以make some changes


def set_Cabin_type(data):
	# data = data.copy()
	data.loc[data.Cabin.notnull(), 'Cabin'] = 'Yes'
	data.loc[data.Cabin.isnull(), 'Cabin'] = 'no'
	return data


# 因为网络输入数据为数值化数据 所以要对dataframe特征因子化
# 注意Cabin Embarked 都非数值类型
def data_dummies(data):
	# ?? Cabin或许直接一个属性
	dummies_Cabin = pd.get_dummies(data.Cabin, prefix='Cabin')
	dummies_Embarked = pd.get_dummies(data.Embarked, prefix='Embarked')
	dummies_Sex = pd.get_dummies(data.Sex, prefix='Sex')
	# ??我在想要不要进行这一步 Pclass本身就是int型而且1,2,3本身代表等级
	dummies_Pclass = pd.get_dummies(data.Pclass, prefix='Pclass')
	data = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
	data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
	return data


# 归一化数据
def data_scaling(data):
	scaler = preprocessing.StandardScaler()
	# data.drop(['Age', 'Fare'], axis = 1, inplace = True)
	# fit/fit_tra 需要的是二维np
	data['Age'] = scaler.fit_transform(data.Age[:, np.newaxis])
	data['Fare'] = scaler.fit_transform(data.Fare[:, np.newaxis])
	return data


"""
测试代码
def data_scaling(data, label = "train", age_scale_param = "", fare_scale_param = ""):  
    # data.drop(['Age', 'Fare'], axis = 1, inplace = True)
    scaler = preprocessing.StandardScaler()
    # fit/fit_tra 需要的是二维np 
    if label == "train":
        age_scale_param = scaler.fit(data.Age[:, np.newaxis])
        fare_scale_param = scaler.fit(data.Fare[:, np.newaxis])
    data['Age'] = scaler.fit_transform(data.Age[:, np.newaxis], age_scale_param)
    data['Fare'] = scaler.fit_transform(data.Fare[:, np.newaxis], fare_scale_param)
    if label == "train":
        return data, age_scale_param, fare_scale_param
    else:
        return data
"""


# def test_data_processing(data, y, rfr, age_scale_param, fare_scale_param):
def test_data_processing(data, y, rfr):
	data.loc[data.Fare.isnull(), 'Fare'] = 0
	data = set_missingAge(data, "test", rfr)
	data = set_Cabin_type(data)
	data = data_dummies(data)
	# data = data_scaling(data, "test", age_scale_param, fare_scale_param)
	data = data_scaling(data)
	y_true = y.values[:, 1]
	data = data.values[:, 1:]
	return data, y_true
