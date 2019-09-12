# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       hourceprice_predic
   Description :
   Author :         amilyxy-pc
   date：           2019/6/5
-------------------------------------------------
"""
import pandas as pd
import data_processing as data_pr

if __name__ == "__main__":
    '''
    func: 读取数据
    data_train.columns可查看特征名称
    '''
    train_data = pd.read_csv("../data/train.csv")
    test_data = pd.read_csv("../data/test.csv")

    '''
    func: 了解数据特征
    '''
    # data_pr.data_change(train_data)
    # data_pr.single_property(train_data)
    final_data, _ = data_pr.deal_outlier(train_data, test_data)
    final_data = data_pr.deal_feature(final_data)
    # data_pr.cal_skew(final_data)
