#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:55:20 2018

@author: amilyxy
describe: Construct some figure to display the data properties
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def Single_attributes(data):
    fig = plt.figure()
    fig.set(alpha = 0.2)
    #the rescued number of pople
    plt.subplot2grid((2, 4), (0, 0))
    data.Survived.value_counts().plot(kind = "bar", color = 'lightskyblue') #柱状图
    plt.title("1: survived people")
    plt.ylabel("num")
    #the distribution of passenger class 
    plt.subplot2grid((2,4), (0, 1))
    data.Pclass.value_counts().plot(kind = "bar", color = 'lightskyblue')
    plt.title("the class of passengers")
    plt.ylabel("num")
    #the gender of passengers
    plt.subplot2grid((2,4), (0, 2))
    data.Sex.value_counts().plot(kind = "bar", color = 'lightskyblue')
    plt.title("the gender of passengers")
    plt.ylabel("num")    
    #the distribution of passengers'age
    plt.subplot2grid((2, 4), (0, 3))
    plt.scatter(data.Survived, data.Age)
    plt.ylabel("age")
    plt.grid(b = True, which = "major", axis = 'y')
    plt.title("the distribution of passengers'age")
    #各个船仓的乘客年龄分布
    plt.subplot2grid((2, 4), (1, 0), colspan = 3)
    data.Age[data.Pclass == 1].plot(kind = 'kde')
    data.Age[data.Pclass == 2].plot(kind = 'kde')
    data.Age[data.Pclass == 3].plot(kind = 'kde')
    plt.xlabel("age")
    plt.ylabel("dedity")
    plt.title("the pass_age of each Pclass")
    plt.legend(('一等舱', '二等舱', '三等舱'), loc = 'best')
    #the passengers'age of each port of embarkation
    plt.subplot2grid((2, 4), (1, 3))
    data.Embarked.value_counts().plot(kind = "bar", color = "lightskyblue")
    plt.xlabel("Port")
    plt.ylabel("num")
    plt.title("the passengers number of each port") 

def Multiple_attributes(data):
    #各船舱获救与未获救人数
    # plt.subplot2grid((2, 3), (0, 0))
    Survived = data.Pclass[data.Survived == 1].value_counts()
    Unsurvived = data.Pclass[data.Survived == 0].value_counts()
    #构建层叠图(Survived和Unsurvived) 
    Pclassp = pd.DataFrame({'survived':Survived, 'unsurvived':Unsurvived})
    Pclassp.plot(kind = 'bar', stacked = True)
    plt.grid(b = True, axis = 'y')
    plt.xlabel("Pclass")
    plt.ylabel("num")
    plt.title("Pclass and passengers")
    plt.show()
    #性别与获救的关系
    # plt.subplot2grid((2, 3), (0, 1))
    male = data.Survived[data.Sex == 'male'].value_counts()
    female = data.Survived[data.Sex == 'female'].value_counts()
    genderp = pd.DataFrame({'male': male, 'female': female})
    genderp.plot(kind = 'bar', stacked = True)
    plt.grid(b = True, axis = 'y')
    plt.xlabel("survival")
    plt.ylabel("num")
    plt.title("gender and passengers")
    plt.show()
    #堂兄妹和父母人数与获救的关系
    '''
    SibSp_group = data.groupby(['SibSp', 'Survived'])
    # 变量SibSp_group是一个GroupBy对象 只是含有data的分组键的中间数据
    df = pd.DataFrame(SibSp_group.count()['PassengerId'])
    print(df)
    parch_group = data_train.groupby(['Parch','Survived'])
    df = pd.DataFrame(parch_group.count()['PassengerId'])
    print(df)
    '''
    #ticket和船舱与获救的关系 
    # Cabin = data.Cabin.value_counts()  #count() 只有204个已知船舱 第一步可先忽略
    # data.Survived[pd.notnull(data.Cabin)] 其中[]里面为index
    Survived_cabin = data.Survived[pd.notnull(data.Cabin)].value_counts()
    Survived_nocabin = data.Survived[pd.isnull(data.Cabin)].value_counts()
    Cabin = pd.DataFrame({'Survived_cabin': Survived_cabin, 'Survived_nocabin': Survived_nocabin})  #.transpose()
    Cabin.plot(kind = 'bar', stacked = True)
    plt.grid(b = True, axis = 'y')
    plt.xlabel("survival")
    plt.ylabel("num")
    plt.title("Cabin and passengers")
    plt.show() 

'''
通常遇到缺值的情况，我们会有几种常见的处理方式
1.缺省样本比例极高，可舍弃，可能作为特征反而加入noise
2.缺省样本比例适中，且非连续值特征属性，那就把NaN作为一个新类别，加到类别特征中
3.如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
'''
#使用RandomForestClassifier 填补缺失年龄属性
def set_missingAge(data):
    #如我想查看Age为nan是不是Cabin一定为nan
    #data_train.Cabin[pd.isnull(data_train.Age)].count()
    # 把已知的数据类型特征取出丢进Random Forest Regressor
    age_df = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    #乘客分为已知年龄和未知年龄两部分
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.istnull()].as_matrix()
    # y - 目标年龄
    y = know_age[:, 0]
    # x - 特征属性值
    X = know_age[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedAge = rfr.predict(unknow_age[:, 1::])
    # 用得到的预测结果填补原缺失数据
    data.loc[(data.Age.isnull()), 'Age'] = predictedAges  
    return data, rfr

def set_Cabin_type(data):
    

