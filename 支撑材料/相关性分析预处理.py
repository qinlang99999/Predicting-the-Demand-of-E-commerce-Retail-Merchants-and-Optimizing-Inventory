import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# 读取商家历史出货量表
df_sales = pd.read_excel(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件1-商家历史出货量表.xlsx')
# 读取商品信息表
df_product = pd.read_excel(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件2-商品信息表.xlsx')
# 读取商家信息表
df_seller = pd.read_excel(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件3-商家信息表.xlsx')
# 读取仓库信息表
df_warehouse = pd.read_excel(r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件4-仓库信息表.xlsx')

# 将日期转换为时间序列
df_sales['date'] = pd.to_datetime(df_sales['date'])
# 合并商家历史出货量表和商品信息表
df_merge = df_sales.merge(df_product, on='product_no')
# 合并商家历史出货量表、商品信息表和商家信息表
df_merge = df_merge.merge(df_seller, on='seller_no')
# 合并商家历史出货量表、商品信息表、商家信息表和仓库信息表
df_merge = df_merge.merge(df_warehouse, on='warehouse_no')
# 选择需要的特征列


df_feature = df_merge[['seller_no', 'product_no', 'warehouse_no', 'category1', 'category2', 'category3', 'seller_category','inventory_category', 'seller_level', 'warehouse _category', 'warehouse _region', 'qty']]


# 对分类特征进行编码
# 创建LabelEncoder对象
label_encoder = LabelEncoder()

# 需要进行数值编码的列名
cat_cols = ['category1', 'category2', 'category3', 'seller_category','inventory_category', 'seller_level', 'warehouse _category', 'warehouse _region']

# 循环对每个类别特征进行数值编码
for col in cat_cols:
    df_feature[col] = label_encoder.fit_transform(df_feature[col])

output_file_path= r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\问题一合并样本示例.xlsx'
df_feature.to_excel(output_file_path, index=False)