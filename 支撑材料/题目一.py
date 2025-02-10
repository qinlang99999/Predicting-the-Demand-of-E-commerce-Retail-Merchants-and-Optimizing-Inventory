import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
# 读取历史出货量数据
input_file_path = r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件1预处理.xlsx'
hi_data = pd.read_excel(input_file_path)
#historical_data =historical_data[1:167]

    
    
    
def forecast_dem(historical_data ,a): 
    historical_data =historical_data[1+a*166:167+a*166]
    # 假设历史数据按日期升序排列
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    
    # 定义检测时间序列稳定性的函数
    
    def test_stationarity(timeseries):
        # 进行滚动统计检验（Rolling Statistics）
        #每三十个数取一个均值
        #检测均值和方差是否发生明显变化    
        rolmean = timeseries.rolling(window=30).mean()  # 选择合适的窗口大小
        rolstd = timeseries.rolling(window=30).std()
    
        # 绘制滚动统计检验结果
        orig = plt.plot(timeseries, color='blue', label='原始数据')
        mean = plt.plot(rolmean, color='red', label='滚动均值')
        std = plt.plot(rolstd, color='black', label='滚动标准差')
        plt.legend(loc='best')
        plt.title('滚动均值和滚动标准差')
        plt.show()
    
        # 进行Dicky-Fuller检验
        #此处用于检验平稳性
        print('Dicky-Fuller检验结果：')
        dftest = adfuller(timeseries, autolag='AIC')
        #timeseries唯一待检验一维时间序列数据，autolag滞后阶数的选择标准
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)
        #白噪声检测   
        print("白噪声检验")
        #返回统计量和p值  lags为检验的延迟数
        print(u'白噪声检验结果：',acorr_ljungbox(timeseries, lags=1))


        
      
        
    if a<=5:
    # 做时间序列稳定性检验
        test_stationarity(historical_data['qty'])
        
        # 进行时间序列差分以达到稳定性
        # 通常需要多次差分，直到时间序列变得稳定
        #第一列数据p-value为约0.0046916，不需要再次差分
        differenced_data = historical_data['qty'].diff().dropna()
        
        # 再次进行稳定性检验
        test_stationarity(differenced_data)
        #参数估计
        train_results = sm.tsa.arma_order_select_ic(differenced_data, ic=['aic', 'bic'], max_ar=4, max_ma=4)
        print('AIC', train_results.aic_min_order)
        print('BIC', train_results.bic_min_order)
    #处理时间数据
    
        
    #转换为浮点数便于程序计算
    historical_data['qty']=np.array(historical_data['qty'],dtype=np.float64)
    # 使用ARIMA模型进行预测
    model = ARIMA(historical_data['qty'], order=(1,1,1))  # 选择合适的ARIMA参数
    results = model.fit()
    
    # 输出模型的统计摘要
    #print(results.summary())
    
    
    # 预测未来需求量
    forecast_periods = 15 # 假设预测未来15天
    forecast = results.forecast(steps=forecast_periods)
    
    # 创建日期范围
    forecast_dates = pd.date_range(start=historical_data['date'].max() + pd.Timedelta(days=1), periods=forecast_periods)
    
    # 将预测结果与日期合并
    forecast_df = pd.DataFrame({'date': forecast_dates, 'predicted_demand': forecast})
    forecast_df.reset_index(inplace=True,drop=True)
    forecast_df['seller_no']=(historical_data['seller_no'][1:16]).reset_index(inplace=False,drop=True)
    forecast_df['product_no']=(historical_data['product_no'][1:16]).reset_index(inplace=False,drop=True)
    forecast_df['warehouse_no']=(historical_data['warehouse_no'][1:16]).reset_index(inplace=False,drop=True)
    
    
    forecast_df['new_index']=[i for i in range(0+15*a,15+15*a)]
    forecast_df=forecast_df.set_index('new_index')

    # 输出预测结果
    return forecast_df

predict_df=forecast_dem(hi_data ,0)    
for i in range(1,1996):
    #1996
    predict_df=pd.concat([predict_df,forecast_dem(hi_data ,i)],axis=0)
    
output_file_path= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件1预测值.xlsx'
predict_df.to_excel(output_file_path, index=False)
