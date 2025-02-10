import pandas as pd
"""
def selc(str_data):
    selc_data=str_data.split("/")
    print(selc_data)
    if len(selc_data[1])==1:
        selc_data[1]="".join(['0',selc_data[1]])
    if len(selc_data[2])==1:
        selc_data[2]="".join(['0',selc_data[2]])
    
    return int("".join(selc_data))
"""
input_file_path = r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件1-商家历史出货量表.xlsx'
output_file_path_1 = r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件1预处理.xlsx'
output_file_path_2 = r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件1预处理总览.xlsx'
df = pd.read_excel(input_file_path)
"""
sn=[]
for i in df["seller_no"]:
    if i not in sn:
        print(i)
        sn.append(i)
print(len(sn))
"""
#df['test'] = df.apply(lambda x: selc(str(x["date"])), axis=1)

df.sort_values(by=['seller_no','warehouse_no','product_no',"date"], inplace=True, ascending=True)

#df.drop('test', axis=1, inplace=True) 



df.to_excel(output_file_path_1, index=False)

sales_agg = df.groupby(['seller_no','product_no', 'warehouse_no', 'date']).agg({
    'qty': ['mean', 'sum']
}).reset_index()
sales_agg.columns = ['seller_no','product_no','warehouse_no', 'date', 'avg_qty', 'total_qty']
# # 构建时间窗口特征
sales_agg['rolling_mean_7d'] = sales_agg.groupby(['seller_no','product_no','warehouse_no'])['total_qty'].rolling(7).mean().reset_index(2, drop=True).reset_index()['total_qty']
sales_agg['rolling_mean_30d'] = sales_agg.groupby(['seller_no','product_no', 'warehouse_no'])['total_qty'].rolling(30).mean().reset_index(2, drop=True).reset_index()['total_qty']
sales_agg['rolling_std_30d'] = sales_agg.groupby(['seller_no','product_no', 'warehouse_no'])['total_qty'].rolling(7).std().reset_index(2, drop=True).reset_index()['total_qty']


sales_agg.to_excel(output_file_path_2, index=False)