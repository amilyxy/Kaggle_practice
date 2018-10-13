"""
Created on Wed Sep 26 15:01:47 2018

@author: amilyxy
describe: the main function to predict Titanic survival
"""
# -*- coding:utf-8 -*-
'''
Pandas是python的一个基于Numpy的数据处理包，有Series、Time-Series、DataFrames、Panel数据结构
'''
import pandas as pd
from pandas import Series, DataFrame 
import numpy as np 
from sklearn import linear_model

# import matplotlib.pyplot as plt
import data_properties
import data_processing

data_train = pd.read_csv("./data/train.csv")
data_test = pd.read_csv("./data/test.csv")
y_ = pd.read_csv("./data/gender_submission.csv")

# data_train.info()/describe() view data properties
# get data properties
# data_properties.Single_attributes(data_train)
# data_properties.Multiple_attributes(data_train)

# 拟合Age数据
data_train_pocessing, rfr = data_processing.set_missingAge(data_train)
data_train_pocessing = data_processing.set_Cabin_type(data_train_pocessing)

# 特征因子化 此处设置了inplace = True
data_train_pocessing = data_processing.data_dummies(data_train_pocessing)
data_train_pocessing = data_processing.data_scaling(data_train_pocessing)

'''
因为是生存问题所以可以使用逻辑回归
'''
# 用正则取出属性值 因为不知道具体的列值
# train_input = data_train_pocessing.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# A DataFrame with mixed type columns will return Object types
train_input = data_train_pocessing.iloc[:, 1:].values
y = train_input[:, 1:]
X = train_input[:, 0]
logr = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)
logr.fit(X, y)

# 归一化测试集数据
test_input, y_true= data_train_pocessing.test_data_procesing(test_data, y_, rfr)

'''
def main(argv = None):
    data_train_pocessing = data_process(data_train)
    sys.exit()

if __name__ == "__main__":
    sys.exit(main())
'''
