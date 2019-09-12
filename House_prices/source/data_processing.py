# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       data_processing
   Description :
   Author :         amilyxy-pc
   date：           2019/6/5
-------------------------------------------------
"""
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm, skew  # for some statistics

'''
func: 画图显示属性
'''
def single_property(data):
    sns.set_style('dark')
    sns.pairplot(x_vars=['OverallQual', 'GrLivArea','YearBuilt', 'TotalBsmtSF'], y_vars = ['SalePrice'], data = data, dropna = True)
    plt.show()

    sns.distplot(data["SalePrice"], kde="False")
    plt.show()

'''
func: 处理离群点与缺失数据
'''
def deal_outlier(train_data, test_data):
    # 离群点只针对训练数据
    train_data.drop(train_data[(train_data['OverallQual'] < 5) & (train_data['SalePrice'] > 200000)].index, inplace=True)
    train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 200000)].index, inplace=True)
    train_data.drop(train_data[(train_data['YearBuilt'] < 1900) & (train_data['SalePrice'] > 400000)].index, inplace=True)
    train_data.drop(train_data[(train_data['TotalBsmtSF'] > 6000) & (train_data['SalePrice'] < 200000)].index, inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    # 缺失点可以考虑所有数据
    train_index = train_data.index
    final_data = pd.concat([train_data, test_data], axis=0, sort=False)
    final_data.reset_index(drop=True, inplace=True)
    test_index = list(set(final_data.index).difference(set(train_index)))

    '''
    # 统计缺失数据
    count=final_data.isnull().sum().sort_values(ascending=False)
    ratio=count/len(final_data)
    final_nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
    '''

    '''
    # 观察变量之间的相关性
    corrmat = train_data.corr()
    plt.subplots(figsize=(12, 9))
    # annot=True 在每个方块上显示值
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()   
    '''
    final_data = fill_missing(final_data)
    return final_data, test_index

def fill_missing(data):
    data['Alley'] = data['Alley'].fillna('missing')
    data['PoolQC'] = data['PoolQC'].fillna(data['PoolQC'].mode()[0])
    # 最多的类型为None
    data['MasVnrType'] = data['MasVnrType'].fillna('None')
    # data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])
    # 但我觉得训练集 测试集 GT和TA分布差不多 应该随机选取
    data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].value_counts().sort_values(ascending=False).index[np.random.randint(0, 2)])
    data['BsmtCond'] = data['BsmtCond'].fillna(data['BsmtCond'].mode()[0])
    data['FireplaceQu'] = data['FireplaceQu'].fillna(data['FireplaceQu'].mode()[0])
    data['GarageType'] = data['GarageType'].fillna('missing')
    data['GarageFinish'] = data['GarageFinish'].fillna('missing')
    data['GarageQual'] = data['GarageQual'].fillna(data['GarageQual'].mode()[0])
    data['GarageCond'] = data['GarageCond'].fillna(data['GarageCond'].mode()[0])
    data['Fence'] = data['Fence'].fillna('missing')
    data['Street'] = data['Street'].fillna(data['Street'].mode()[0])
    data['LotShape'] = data['LotShape'].fillna(data['LotShape'].value_counts().sort_values(ascending=False).index[np.random.randint(0, 2)])
    data['LandContour'] = data['LandContour'].fillna(data['LandContour'].mode()[0])
    data['BsmtExposure'] = data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0])
    data['BsmtFinType1'] = data['BsmtFinType1'].fillna('missing')
    data['BsmtFinType2'] = data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0])
    data['CentralAir'] = data['CentralAir'].fillna('missing')
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['MiscFeature'] = data['MiscFeature'].fillna('missing')
    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])    
    data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode()[0])
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])    
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].value_counts().sort_values(ascending=False).index[np.random.randint(0, 2)])
    data["Functional"] = data["Functional"].fillna(data['Functional'].mode()[0])
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
    # 数值型变量的空值先用0值替换
    flist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
             'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
             'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
             'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    for fl in flist:
        data[fl] = data[fl].fillna(0)
    # 0值替换
    data['TotalBsmtSF'] = data['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    data['2ndFlrSF'] = data['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    data['GarageArea'] = data['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    data['GarageCars'] = data['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    data['LotFrontage'] = data['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    data['MasVnrArea'] = data['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    data['BsmtFinSF1'] = data['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    return data

def QualToInt(x):
    if(x=='Ex'):
        r = 0
    elif(x=='Gd'):
        r = 1
    elif(x=='TA'):
        r = 2
    elif(x=='Fa'):
        r = 3
    elif(x=='missing'):
        r = 4
    else:
        r = 5
    return r

def deal_feature(data):
    # 一些特征其被表示成数值特征缺乏意义，例如年份还有类别，这里将其转换为字符串，即类别型变量
    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)
    data['OverallCond'] =data['OverallCond'].astype(str)
    #
    data['ExterQual'] = data['ExterQual'].apply(QualToInt)
    data['ExterCond'] = data['ExterCond'].apply(QualToInt)
    data['KitchenQual'] = data['KitchenQual'].apply(QualToInt)
    data['HeatingQC'] = data['HeatingQC'].apply(QualToInt)
    data['BsmtQual'] = data['BsmtQual'].apply(QualToInt)
    data['BsmtCond'] = data['BsmtCond'].apply(QualToInt)
    data['FireplaceQu'] = data['FireplaceQu'].apply(QualToInt)
    data['GarageQual'] = data['GarageQual'].apply(QualToInt)
    data['PoolQC'] = data['PoolQC'].apply(QualToInt)
    data['BsmtFinType1'] = data['BsmtFinType1'].apply(QualToInt)
    data['MasVnrType'] = data['MasVnrType'].apply(QualToInt)
    data['Foundation'] = data['Foundation'].apply(QualToInt)
    data['HouseStyle'] = data['HouseStyle'].apply(QualToInt)
    data['Functional'] = data['Functional'].apply(QualToInt)
    data['BsmtExposure'] = data['BsmtExposure'].apply(QualToInt)
    data['GarageFinish'] = data['GarageFinish'].apply(QualToInt)
    data['PavedDrive'] = data['PavedDrive'].apply(QualToInt)
    data['Street'] = data['Street'].apply(QualToInt)

    # 区域相关特征对于确定房价非常重要，我们还增加了一个特征，即每个房屋的地下室，一楼和二楼的总面积
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    data['HasWoodDeck'] = (data['WoodDeckSF'] == 0) * 1
    data['HasOpenPorch'] = (data['OpenPorchSF'] == 0) * 1
    data['HasEnclosedPorch'] = (data['EnclosedPorch'] == 0) * 1
    data['Has3SsnPorch'] = (data['3SsnPorch'] == 0) * 1
    data['HasScreenPorch'] = (data['ScreenPorch'] == 0) * 1
    # 房屋改造时间（YearsSinceRemodel）与房屋出售时间（YrSold）间隔时间的长短通常也会影响房价
    data['YearsSinceRemodel'] = data['YrSold'].astype(int) - data['YearRemodAdd'].astype(int)
    # 房屋的整体质量也是影响房价的重要因素
    data['Total_Home_Quality'] = data['OverallQual'] + data['OverallCond']


def data_change(data):
    quantitative = [f for f in data.columns if data.dtypes[f] != 'object' and data.dtypes[f] != 'str']
    quantitative.remove('SalePrice')
    f = pd.melt(data, value_vars=quantitative)
    g = sns.FacetGrid(f, col="variable", col_wrap=5, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    plt.show()

def cal_skew(data):
    quantitative = [f for f in data.columns if data.dtypes[f] != 'object' and data.dtypes[f] != 'str']
    quantitative.remove('SalePrice')
    skewed_feats = data[quantitative].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print(skewness.head(20))


# 我觉得教程有点乱  很抱歉这个题目暂时只能进行到这里 以后有机会再来看看吧~









