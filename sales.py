#!/usr/bin/python3
# __*__ coding: utf-8 __*__

'''
@Author: simonKing
@License: (C) Copyright 2013-2019, Best Wonder Corporation Limited.
@Os：Windows 10 x64
@Contact: bw_wangxiaomeng@whty.com.cn
@Software: PY PyCharm 
@File: sales.py
@Time: 2019/8/17 12:01
@Desc: define your function
'''

from model.arimaModel import *
from settings import fname

def loadData(fname):
    '''
    导入数据
    :return:
    '''
    data = pd.read_excel(fname, index_col = '日期',header = 0)
    return data

def roundResult(result):
    '''
    默认预测6个点，即为两个月的数据，否则就不合并
    :param result:
    :return:
    '''
    if len(result) ==6:
        salesArr = [round(sum(result[0:3])),round(sum(result[3:6]))]
    else:
        salesArr = [round(r) for r in result]
    # 对预测结果进行业务判断，小于等于0就预测为1
    sales = []
    for s in  salesArr:
        if s<= 0:
            s = 1
        sales.append(s)
    return sales

def predictSales(fname,n=6,isVisiable=False):
    '''
    程序执行的入口
    :param fname:输入文件
    :param n:预测的点个数
    :param isVisiable:是否可视化
    :return:6个点就是两个月，每月分上中下旬三个点
    '''
    # 加载数据
    data = loadData(fname)
    # 对序列差分处理
    D_data = diffData(data)
    if isVisiable:
        # 画出差分后的时序图
        sequencePlot(D_data)
        # 画出自相关图
        selfRelatedPlot(D_data)
        # 画出偏相关图
        partialRelatedPlot(D_data)
    # 对差分序列平稳性检测
    D_result = stableCheck(D_data)
    print('差分序列的ADF 检验结果为：', D_result)
    # 对模型进行定阶
    p,q = selectArgsForModel(D_data)
    # 建立模型
    model = bulidModel(data,p,q)
    # 进行销量预测
    result = predict(model,n).tolist()
    # 对结果进行取整处理
    result = roundResult(result)
    print('预测未来n个点的销量为：',result)
    return result

if __name__ == '__main__':
    # isVisiable可视化按钮
    result = predictSales(fname,6,isVisiable=True)