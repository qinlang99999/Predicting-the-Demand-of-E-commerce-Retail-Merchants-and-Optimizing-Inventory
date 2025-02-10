#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

df1 = pd.read_excel('/附件数据及提交结果表格/附件表/附件1-商家历史出货量表.xlsx')
df5 = pd.read_excel('/附件数据及提交结果表格/附件表/附件5-新品历史出货量表.xlsx')


df1['date'] = pd.to_datetime(df1['date'])
df1 = df1.sort_values('date')
df5['date'] = pd.to_datetime(df5['date'])
df5 = df5.sort_values('date')
result_table_2 = pd.DataFrame(columns=['seller_no', 'warehouse_no', 'product_no', 'date', 'predicted_demand'])

similar_list = []
feature_cols = ['seller_no','product_no','warehouse_no']



df1.head()

# 对df1按照seller_no、warehouse_no、product_no分组，得到每个组内的数据。
df1_grouped = df1.groupby(['seller_no','product_no','warehouse_no'])
# 对df5按照seller_no、warehouse_no、product_no进行groupby
df5_grouped = df5.groupby(['seller_no','product_no','warehouse_no'])

i=0
i,j=0,0
for (seller_no, warehouse_no, product_no), df5_group in df5_grouped:
    

    df5_cos = df5[['seller_no','product_no','warehouse_no']][i:i+1]
    matched_sequence = None # 初始化匹配的序列为None
    for (s1, w1, p1), df1_group in df1_grouped:

        
        df1_cos = df1[['seller_no','product_no','warehouse_no']][j:j+1]
        df_concat = pd.concat([df1_cos, df5_cos], axis=0)
        
        #one-hot编码
        df_encoded = pd.get_dummies(df_concat)
        df1_encoded = df_encoded[:len(df1_cos)]
        df5_encoded = df_encoded[len(df5_cos):]
        
        array1 = df1_encoded.to_numpy()
        array2 = df5_encoded.to_numpy()
        
        #计算余弦相似度
        cosine_sim = cosine_similarity(array1, array2)
        # 计算平均相似度
        average_similarity = np.mean(cosine_sim)
        
        seq1 = list(df5_group['qty'])
        seq2 = list(df1_group['qty'])
        # 向前补充0使得seq1和seq2的长度相同
        if len(seq1) < len(seq2):
            seq2 = seq2[-len(seq1):]
        elif len(seq1) > len(seq2):
            seq1  = seq1[-len(seq2):]
        distance = euclidean(seq1,seq2)# 计算欧式距离
        
        min_distance = float('inf') # 初始化最小距离为无穷大
        max_similarity = 0
        
        if max_similarity < average_similarity and distance < min_distance:
            max_similarity = average_similarity
            min_distance = distance
            matched_sequence = (s1, w1, p1,seller_no, warehouse_no, product_no)
        
        j+=1
        
        
    i +=1
    
    if matched_sequence is not None:
        similar_list.append(matched_sequence)

 
            
print(similar_list)


result_df = pd.DataFrame(result, columns=['similar_list', 'warehouse_no_df1', 'product_no_df1', 'seller_no_df5', 'warehouse_no_df5', 'product_no_df5'])
result_df.to_excel('//序列结果.xlsx',index=False)


# In[ ]:


from statsmodels.tsa.arima.model import ARIMA
predictions = []

result_df = pd.DataFrame(similar_list, columns=['seller_no_df1', 'warehouse_no_df1', 'product_no_df1', 'seller_no_df5', 'warehouse_no_df5', 'product_no_df5'])
result_df = result_df.drop_duplicates(subset=['seller_no_df5', 'warehouse_no_df5', 'product_no_df5'])
result_table2 = []

for index, row in result_df.iterrows():#ARIMA模型
    seller_no, warehouse_no,product_no, seller_no_df5, product_no_df5, warehouse_no_df5 = row

    ts_data = df1[(df1['seller_no'] == seller_no)& (df1['product_no'] == product_no) & (df1['warehouse_no'] == warehouse_no)][['date', 'qty']]
    if len(ts_data)>0:
        ts_data = ts_data.sort_values('date')
        ts_data = ts_data.set_index('date')

        model = ARIMA(ts_data, order=(1, 1, 1)).fit()        # 训练模型进行训练
        predict = model.predict(start=len(ts_data)+1, end=len(ts_data)+15, dynamic=True)

        result = pd.DataFrame(columns=['seller_no','product_no','warehouse_no','date','forecast_qty'])
        result['seller_no'] = [seller_no_df5]*15
        result['product_no'] = [product_no_df5]*15
        result['warehouse_no'] = [warehouse_no_df5]*15
        result['date']= pd.date_range(start='2023-05-16', periods=15, freq='D')
        result['forecast_qty'] = list(predict)

        result_table2.append(result)


# In[ ]:


data = df1
#线性回归模型

train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 创建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测未来的时间点
future_time_points = np.arange(n_samples, n_samples + 20).reshape(-1, 1)
future_predictions = model.predict(future_time_points)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='观测值', color='b')
plt.plot(X, model.predict(X), label='线性回归', color='r')
plt.plot(future_time_points, future_predictions, label='未来预测', color='g', linestyle='--')
plt.legend()
plt.xlabel('时间')
plt.ylabel('数值')
plt.title('线性回归时间序列预测')
plt.show()


# In[ ]:


pd.concat(result_table2, axis=0, ignore_index=True).to_excel('结果表/结果表2-预测结果表.xlsx', index=False)


# In[ ]:




