#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:55:20 2018

@author: amilyxy
describe: Construct some figure to display the data properties
"""
import matplotlib.pyplot as plt
import pandas as pd

def Single_attributes(data):
    fig = plt.figure()
    fig.set(alpha = 0.2)
    #the rescued number of people
    plt.subplot2grid((2, 2), (0, 0))
    data.Survived.value_counts().plot(kind = "bar", color = 'lightskyblue') #柱状图
    plt.title("1: survived people")
    plt.ylabel("num")
    #the distribution of passenger class 
    plt.subplot2grid((2,2), (0, 1))
    data.Pclass.value_counts().plot(kind = "bar", color = 'lightskyblue')
    plt.title("the class of passengers")
    plt.ylabel("num")
    #the gender of passengers
    plt.subplot2grid((2,2), (1, 0))
    data.Sex.value_counts().plot(kind = "bar", color = 'lightskyblue')
    plt.title("the gender of passengers")
    plt.ylabel("num")    
    #the distribution of passengers'age
    plt.subplot2grid((2, 2), (1, 1))
    plt.scatter(data.Survived, data.Age)
    plt.ylabel("age")
    plt.grid(b = True, which = "major", axis = 'y')
    plt.title("the distribution of passengers'age")

    plt.figure()
    #各个船仓的乘客年龄分布
    plt.subplot2grid((2, 1), (0, 0), colspan = 2)
    data.Age[data.Pclass == 1].plot(kind = 'kde')
    data.Age[data.Pclass == 2].plot(kind = 'kde')
    data.Age[data.Pclass == 3].plot(kind = 'kde')
    plt.xlabel("age")
    plt.ylabel("dedity")
    plt.title("the pass_age of each Pclass")
    plt.legend(('一等舱', '二等舱', '三等舱'), loc = 'best')
    #the passengers'age of each port of embarkation
    plt.subplot2grid((2, 2), (1, 0))
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
    Survived = data.Sex[data.Survived == 1].value_counts()
    Unsurvived = data.Sex[data.Survived == 0].value_counts()  
    genderp = pd.DataFrame({'Survived': Survived, 'Unsurvived': Unsurvived})
    genderp.plot(kind = 'bar', stacked = True)
    plt.grid(b = True, axis = 'y')
    plt.xlabel("Sex")
    plt.ylabel("num")
    plt.title("gender and passengers")
    plt.show()
    #堂兄妹和父母人数与获救的关系
    '''
    SibSp_group = data.groupby(['SibSp', 'Survived'])
    # 变量SibSp_group是一个GroupBy对象 只是含有data的分组键['SibSp', 'Survived']的中间数据
    df = pd.DataFrame(SibSp_group.count()['PassengerId'])
    print(df)
    parch_group = data_train.groupby(['Parch','Survived'])
    df = pd.DataFrame(parch_group.count()['PassengerId'])
    print(df)
    '''
    #ticket和船舱与获救的关系 
    # Cabin = data.Cabin.value_counts()  #count() 只有204个已知船舱 第一步可先忽略
    # data.Survived[pd.notnull(data.Cabin)] 其中[]里面为index
    Survived = data.Cabin.notnull()[data.Survived == 1].value_counts()
    Unsurvived = data.Cabin.notnull()[data.Survived == 0].value_counts()
    Cabin = pd.DataFrame({'Survived': Survived, 'Unsurvived': Unsurvived})  #.transpose()
    Cabin.plot(kind = 'bar', stacked = True)
    plt.grid(b = True, axis = 'y')
    plt.xlabel("Cabin")
    plt.ylabel("num")
    plt.title("Cabin and passengers")
    plt.show() 

    #登港港口与获救的关系
    Survived = data.Embarked[data.Survived == 1].value_counts()
    Unsurvived = data.Embarked[data.Survived == 0].value_counts()
    Embarked = pd.DataFrame({'Survived': Survived, 'Unsurvived': Unsurvived})
    Embarked.plot(kind = 'bar', stacked = True)
    plt.grid(b = True, axis = 'y')
    plt.xlabel("Embarked")
    plt.ylabel("num")
    plt.title("Embarked and passengers")
    plt.show() 
    
