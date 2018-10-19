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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
import data_properties
import data_processing

data_train = pd.read_csv("./data/train.csv")
data_test = pd.read_csv("./data/test.csv")
y_ = pd.read_csv("./data/gender_submission.csv")

'''
关于数据属性的一些可视化
'''
# data_train.info()/describe() view data properties
# get data properties
# data_properties.Single_attributes(data_train)
# data_properties.Multiple_attributes(data_train)

# 拟合Age数据
data_train_pocessing, rfr = data_processing.set_missingAge(data_train)
data_train_pocessing = data_processing.set_Cabin_type(data_train_pocessing)

# 特征因子化 此处设置了inplace = True
data_train_pocessing = data_processing.data_dummies(data_train_pocessing)
# data_train_pocessing, age_scale_param, fare_scale_param = data_processing.data_scaling(data_train_pocessing)
data_train_pocessing = data_processing.data_scaling(data_train_pocessing)

'''
因为是生存问题所以可以使用逻辑回归(交叉验证)
'''
# 用正则取出属性值 因为不知道具体的列值
# train_input = data_train_pocessing.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# df.values A DataFrame with mixed type columns will return Object types
# 选择使用特征
regex='Survived|Age|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*'
train_data, cv_data  = train_test_split(data_train_pocessing, test_size = 0.3, random_state = 0)
train_input = train_data.filter(regex = regex).values
# train_input = data_train_pocessing.filter(regex = regex).values

X_train = train_input[:, 1:]
y_train = train_input[:, 0]
# 逻辑回归模型      
logr = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)
logr.fit(X_train, y_train)
# 逻辑回归查看每个sample的feature weight， 正相关 or 负相关
# logr_weight = pd.DataFrame({"columns":list(train_data.columns)[2:], "coef":list(logr.coef_.T)})

# 输出5次交叉验证score
print (cross_val_score(logr, X_train, y_train, cv = 5))

'''
对cross validation数据进行预测
'''
# origin_data = pd.read_csv("./data/train.csv")
# cv_input = cv_data.filter(regex = regex).values
# y_cvpred = logr.predict(cv_input[:, 1:])
# # 预测错误示例信息
# bad_cases = origin_data.loc[origin_data.PassengerId.isin(cv_data[y_cvpred != cv_input[:, 0]].PassengerId.values)]

'''
测试集验证
'''
# 归一化测试集数据
test_input, y_true= data_processing.test_data_processing(data_test, y_, rfr)
# test_input, y_true= data_processing.test_data_processing(data_test, y_, rfr, age_scale_param, fare_scale_param)

y_pred = logr.predict(test_input).astype(int)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': y_pred})
result.to_csv("./data/test_pred.csv", index = False)
'''
def main(argv = None):
    data_train_pocessing = data_process(data_train)
    sys.exit()

if __name__ == "__main__":
    sys.exit(main())
'''
