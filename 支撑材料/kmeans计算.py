import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# 对数值特征进行归一化
from sklearn.preprocessing import MinMaxScaler

input_file_path =r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\问题一合并样本示例.xlsx'
df_feature= pd.read_excel(input_file_path)

# 创建MinMaxScaler对象
scaler = MinMaxScaler()
df_feature_2 = df_feature.drop(columns=['seller_no','product_no','warehouse_no'])
# 对df_feature进行归一化
df_feature_normalized = pd.DataFrame(scaler.fit_transform(df_feature_2), columns=df_feature_2.columns)
output_file_path= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\归一化结果.xlsx'
df_feature_normalized.to_excel(output_file_path, index=False)


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
best_k = 3
best_random_state = 7
# 可视化最佳K值和随机种子下的聚类结果
kmeans_best = KMeans(n_clusters=best_k, random_state=best_random_state)
kmeans_best.fit(df_feature_normalized)
labels = kmeans_best.labels_
#存储
df_feature_normalized['cluster']=labels
df_feature_normalized.sort_values(by='cluster', inplace=True, ascending=True)
output_file_path= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\第一题分类\总分类簇结果.xlsx'
df_feature_normalized.to_excel(output_file_path, index=False)



# 使用PCA将数据降至3维
pca = PCA(n_components=3)
df_feature_3d = pca.fit_transform(df_feature_normalized)

# 可视化最佳K值和随机种子下的聚类结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_feature_3d[:, 0], df_feature_3d[:, 1], df_feature_3d[:, 2], c=labels, cmap='viridis')
ax.set_title('Clustering Result (K={}, Random State={})'.format(best_k, best_random_state))
plt.savefig(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\img\2.png',dpi=300)
plt.show()