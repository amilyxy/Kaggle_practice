"""
Created on Wed Sep 26 15:01:47 2018

@author: amilyxy
describe: the main function to predict Titanic survival
"""
# -*- coding:utf-8 -*-
import pandas as pd
'''
Pandas是python的一个基于Numpy的数据处理包，有Series、Time-Series、DataFrames、Panel数据结构
'''
import numpy as np 
from pandas import Series, DataFrame 
# import matplotlib.pyplot as plt
import data_properties
import data_processing

# data_train.info()/describe() view data properties
data_train = pd.read_csv("./data/train.csv")
# get data properties
# data_properties.Single_attributes(data_train)
# data_properties.Multiple_attributes(data_train)

# 拟合Age数据
data_train_pocessing, rfr = data_processing.set_missingAge(data_train)
data_train_pocessing = data_processing.set_Cabin_type(data_train_pocessing)

# 特征因子化 此处设置了inplace = True
data_train_pocessing = data_processing.data_dummies(data_train_pocessing)
data_train_pocessing = data_processing.data_scaling(data_train_pocessing)

