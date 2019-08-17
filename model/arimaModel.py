#!/usr/bin/python3
# __*__ coding: utf-8 __*__

'''
@Author: simonKing
@License: (C) Copyright 2013-2019, Best Wonder Corporation Limited.
@Os：Windows 10 x64
@Contact: bw_wangxiaomeng@whty.com.cn
@Software: PY PyCharm 
@File: arimaModel.py
@Time: 2019/8/1 18:53
@Desc: define your function
'''

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
# 定义使其正常显示中文字体黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示表示负号
plt.rcParams['axes.unicode_minus'] = False

def dataConversion(v):
    '''
    转换数据格式为dataFrame
    :param v:销量
    :return:
    '''
    new_v = pd.Series(v)
    # data = pd.DataFrame({"日期":ids,"销量":new_v},)
    data = pd.DataFrame({"销量":new_v},)
    return data

def sequencePlot(data):
    '''
    画出时序图
    :param data:输入数据
    :return:
    '''
    data.plot()
    plt.title("销量时序图")
    plt.show()

def selfRelatedPlot(data):
    '''
    画出自相关性图，看看是否具有周期性、淡旺季等
    :param data:输入数据
    :return:
    '''
    plot_acf(data)
    plt.title("序列自相关情况")
    plt.show()

def partialRelatedPlot(data):
    '''
    画出偏相关图，序列受前后销量的走势的影响情况
    :param data:差分序列
    :return:
    '''
    plot_pacf(data)
    plt.title("序列偏相关情况")
    plt.show()

def stableCheck(data):
    '''
    平稳性检测
    :param data:
    :return:返回值依次为：adf, pvalue p值， usedlag, nobs, critical values临界值
    # icbest, regresults, resstore
    #adf 分别大于3中不同检验水平的3个临界值，单位检测统计量对应的p 值显著大于 0.05 ，
    #说明序列可以判定为 非平稳序列
    '''
    result = adfuller(data['销量'])
    print('原始序列的检验结果为：',adfuller(data['销量']))
    return result

def diffData(data):
    '''
    对数据进行差分
    :param data:
    :return:
    '''
    D_data = data.diff().dropna()
    return D_data

def whiteNoiseCheck(data):
    '''
    对n阶差分后的序列做白噪声检验,
    差分序列的白噪声检验结果： (array([*]), array([*]))
    p值为第二项， 远小于 0.05
    :param data:n阶差分序列
    :return:
    '''
    result = acorr_ljungbox(data, lags= 1)
    #返回统计量和 p 值
    print('差分序列的白噪声检验结果：',result)
    return result

def selectArgsForModel(D_data):
    '''
    对模型进行定阶
    :param D_data:差分数列
    :return:p,q
    '''
    #一般阶数不超过 length /10
    pmax = int(len(D_data) / 10)
    qmax = int(len(D_data) / 10)
    bic_matrix = []
    for p in range(pmax +1):
        temp = []
        for q in range(qmax+1):
            try:
                value = ARIMA(D_data, (p, 1, q)).fit().bic
                temp.append(value)
            except:
                temp.append(None)
            bic_matrix.append(temp)
    #将其转换成Dataframe 数据结构
    bic_matrix = pd.DataFrame(bic_matrix)
    #先使用stack 展平， 然后使用 idxmin 找出最小值的位置,
    p,q = bic_matrix.stack().astype('float64').idxmin()
    #  BIC 最小的p值 和 q 值：0,1
    print('BIC 最小的p值 和 q 值：%s,%s' %(p,q))
    return p,q

def bulidModel(data,p,q):
    '''
    建立ARIMA 模型，修复平稳性检测不通过的情况
    :param data:
    :param p:
    :param q:
    :return:
    '''
    try:
        model = ARIMA(data, (p,1,q)).fit()
    except:
        # 平稳性检测不通过，参考：https://github.com/statsmodels/statsmodels/issues/1155/
        model = ARIMA(data, (4,1,1)).fit()
    try:
        # 检测模型是否可用
        model.summary2()
    except:
        # 模型平衡性查，就固定p,d,q固定为4，1，1
        model = ARIMA(data, (4,1,1)).fit()
        model.summary2()
    # 保存模型
    # model.save('model.pkl')
    return model

def predict(model,n=6):
    '''
    进行预测
    :param model: 模型
    :param n:一般3个点是一个月
    :return:预测结果
    '''
    if isinstance(model,str):
        # 模型加载
        loaded = ARIMAResults.load('model.pkl')
        # 预测未来3个单位,即为1个月
        predictions=loaded.forecast(n)
        # 预测结果为：
        pre_result = predictions[0]
        print('预测结果为：',pre_result)
        # 标准误差为：
        error = predictions[1]
        print('标准误差为：',error)
        # 置信区间为：
        confidence = predictions[2]
        print('置信区间为：',confidence)
    else:
        # 预测未来3个单位,即为1个月
        predictions=model.forecast(n)
        # 预测结果为：
        pre_result = predictions[0]
        print('预测结果为：',pre_result)
        # 标准误差为：
        error = predictions[1]
        print('标准误差为：',error)
        # 置信区间为：
        confidence = predictions[2]
        print('置信区间为：',confidence)
    return pre_result