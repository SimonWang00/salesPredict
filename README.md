# salesPredict

### ARIMA模型简介
ARIMA模型提供了基于时间序列理论，对数据进行平稳化处理（AR和MA过程）、模型定阶（自动差分过程）、参数估计，建立模型，并对模型进行检验。
在Python中statsmodel提供了全套的解决方案，包括窗口选择、自动定阶和平稳性检测等等算法。

### 预测策略
每月分上中下旬三个点预测，每月预测三次当月销量。这么做的好处是，月上旬和中旬的实际销量可以作为先验知识，提高模型预测的准确率。

### 环境
- Windows 10
- Python 3.6.5

### 依赖包
```
pip install -r requirements.txt
```

### 程序执行
```
python sales.py
```

### 建模过程
<img src="./pictures/销量时序图.png?raw=true"/> 
<img src="./pictures/一阶差分后，序列自相关情况.png?raw=true"/> 
<img src="./pictures/一阶差分后，序列偏相关情况.png?raw=true"/>
</br>

### 预测效果测试
<img src="./pictures/销量预测测试情况.png?raw=true"/>
</br>

### 线上预测效果
截止到8月15日累计销量486，预测8月份销量为889
<img src="./pictures/上线效果.png?raw=true"/>
</br>
