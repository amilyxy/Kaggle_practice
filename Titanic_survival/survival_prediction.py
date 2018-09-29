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

#data_train.info()/describe() view data properties
data_train = pd.read_csv("./data/train.csv")
#get data figures
data_properties.Multiple_attributes(data_train)
