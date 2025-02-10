import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

date=['商品','商家','仓库']

# 读取商品信息表
df_product = pd.read_excel(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件2-商品信息表.xlsx')
# 读取商家信息表
df_seller = pd.read_excel(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件3-商家信息表.xlsx')
# 读取仓库信息表
df_warehouse = pd.read_excel(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件4-仓库信息表.xlsx')


def k_means_predict(use_data,col_lis,cat_cols,data_num):  
    # 对分类特征进行编码
    # 创建LabelEncoder对象
    label_encoder = LabelEncoder()
    
    # 需要进行数值编码的列名
    cat_cols = cat_cols
    
    # 循环对每个类别特征进行数值编码
    for col in cat_cols:
        use_data[col] = label_encoder.fit_transform(use_data[col])

    # 对数值特征进行归一化
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    df_feature_2 = use_data.drop(columns=col_lis)
    # 对df_feature进行归一化
    df_feature_normalized = pd.DataFrame(scaler.fit_transform(df_feature_2), columns=df_feature_2.columns)
    output_file_path_1=r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\第一题分类\%s归一化结果.xlsx'%date[data_num]
    df_feature_normalized.to_excel(output_file_path_1, index=False)
    
    # 创建一个空列表存储不同K值下的SSE得分
    sse_scores = []
    
    # 定义待测试的K值范围
    k_values = range(2, 10)
    
    # 定义最佳的K值和随机种子变量
    best_k = 0
    best_random_state = 0
    best_sse = float('inf')  # 初始值设为正无穷大
    
    # 循环测试不同的K值和随机种子
    for k in k_values:
        for random_state in range(10):
            # 初始化KMeans模型
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            # 训练模型
            kmeans.fit(df_feature_normalized)
    
            # 计算SSE得分并存储到列表中
            sse = kmeans.inertia_
            sse_scores.append((k, random_state, sse))
    
            # 比较得分，更新最佳的K值和随机种子
            if sse < best_sse:
                best_k = k
                best_random_state = random_state
                best_sse = sse
    
    # 打印最佳的K值和随机种子
    print("最佳的K值:", best_k)
    print("最佳的随机种子:", best_random_state)
    
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # 可视化最佳K值和随机种子下的聚类结果
    kmeans_best = KMeans(n_clusters=best_k, random_state=best_random_state)
    kmeans_best.fit(df_feature_normalized)
    labels = kmeans_best.labels_
    #簇标签分类
    
    
    if data_num!=2:
        # 使用PCA将数据降至3维
        pca = PCA(n_components=3)
        df_feature_3d = pca.fit_transform(df_feature_normalized)
    
        # 可视化最佳K值和随机种子下的聚类结果
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_feature_3d[:, 0], df_feature_3d[:, 1], df_feature_3d[:, 2], c=labels, cmap='viridis')
        ax.set_title('Clustering Result (K={}, Random State={})'.format(best_k, best_random_state))
        output_file_path_2= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\第一题分类\%s分类簇结果可视化.png'%date[data_num]
        plt.savefig(output_file_path_2,dpi=300)
        plt.show()
    else:
        plt.style.use('fivethirtyeight')
        
        x= df_feature_normalized[cat_cols[0]]
        y= df_feature_normalized[cat_cols[1]]
        print(x)
        print(y)
        
        plt.scatter(x, y)
        
        plt.title('散点图')
        
        plt.xlabel(cat_cols[0])
        plt.ylabel(cat_cols[1])
        plt.xticks(np.arange(0,1,0.5))
        plt.yticks(np.arange(0,1,0.05))        
        
        plt.tight_layout()
        output_file_path_2= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\第一题分类\%s分类簇结果可视化.png'%date[data_num]
        plt.savefig(output_file_path_2,dpi=300)
        plt.show()

    
    return labels
    
#商家分类
seller_lab=k_means_predict(df_seller,col_lis=['seller_no'],cat_cols=['seller_category','inventory_category','seller_level'],data_num=1)
df_seller['cluster']=seller_lab
df_seller.sort_values(by='cluster', inplace=True, ascending=True)
output_file_path= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\第一题分类\%s分类簇结果.xlsx'%date[1]
df_seller.to_excel(output_file_path, index=False)

#商品分类
pro_lab=k_means_predict(df_product,col_lis=['product_no'],cat_cols=['category1','category2','category3'],data_num=0)
df_product['cluster']=pro_lab
df_product.sort_values(by='cluster', inplace=True, ascending=True)
output_file_path= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\第一题分类\%s分类簇结果.xlsx'%date[0]
df_product.to_excel(output_file_path, index=False)

#仓库分类
war_lab=k_means_predict(df_warehouse,col_lis=['warehouse_no'],cat_cols=['warehouse _category','warehouse _region'],data_num=2)
df_warehouse['cluster']=war_lab
df_warehouse.sort_values(by='cluster', inplace=True, ascending=True)
output_file_path= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\第一题分类\%s分类簇结果.xlsx'%date[2]
df_warehouse.to_excel(output_file_path, index=False)


