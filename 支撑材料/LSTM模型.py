#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:40:40 2023

@author: pankehong
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# 读取数据

sales_data = pd.read_excel("/Users/pankehong/Desktop/mmb/问题一代码/附件表/附件1-商家历史出货量表.xlsx")
product_info = pd.read_csv("/Users/pankehong/Desktop/mmb/问题一代码/附件表/附件2-商品信息表.xlsx")
seller_info = pd.read_csv("/Users/pankehong/Desktop/mmb/问题一代码/附件表/附件3-商家信息表.xlsx")
warehouse_info = pd.read_csv("/Users/pankehong/Desktop/mmb/问题一代码/附件表/附件4-仓库信息表.xlsx")

 

# 合并数据

merged_data = pd.merge(sales_data, product_info, on="product_no")

merged_data = pd.merge(merged_data, seller_info, on="seller_no")

merged_data = pd.merge(merged_data, warehouse_info, on="warehouse_no")

 

# 特征工程

# 从日期字段提取年份、月份、日等时间特征

merged_data["date"] = pd.to_datetime(merged_data["date"])

merged_data["year"] = merged_data["date"].dt.year

merged_data["month"] = merged_data["date"].dt.month

merged_data["day"] = merged_data["date"].dt.day

 

# 选择需要的特征列作为模型输入

feature_cols = ["seller_no", "product_no", "warehouse_no", "year", "month", "day", "qty"]

data = merged_data[feature_cols]

 

# 时间序列化

data = data.sort_values(by="date")

 

# 创建训练集和测试集

train_size = int(len(data) * 0.8)

train_data = data[:train_size]

test_data = data[train_size:]

 

# 创建时间窗口数据集

def create_dataset(dataset, look_back=1):

    X, Y = [], []

    for i in range(len(dataset) - look_back):

        a = dataset.iloc[i:(i + look_back)]

        X.append(a[feature_cols[:-1]].values)

        Y.append(a["qty"].values)

    return np.array(X), np.array(Y)

 

look_back = 30  # 时间窗口大小

X_train, Y_train = create_dataset(train_data, look_back)

X_test, Y_test = create_dataset(test_data, look_back)

 

# 数据重塑

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

 

# 构建LSTM模型

model = keras.Sequential()

model.add(keras.layers.LSTM(50, input_shape=(1, len(feature_cols) - 1)))

model.add(keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

 

# 训练模型

model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

 

# 预测

train_predict = model.predict(X_train)

test_predict = model.predict(X_test)

 

# 反向缩放

scaler = MinMaxScaler()

scaler.fit(data[["qty"]])

train_predict = scaler.inverse_transform(train_predict)

test_predict = scaler.inverse_transform(test_predict)

 

# 评估模型性能

train_score = sqrt(mean_squared_error(Y_train, train_predict))

print('Train RMSE: %.2f' % train_score)

 

test_score = sqrt(mean_squared_error(Y_test, test_predict))

print('Test RMSE: %.2f' % test_score)