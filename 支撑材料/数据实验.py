import pandas as pd

input_file_path =r'C:\Users\Administrator\Desktop\2023年MathorCup大数据挑战赛-赛道B初赛\附件数据及提交结果表格\附件表\附件1预处理.xlsx'

df = pd.read_excel(input_file_path)

sn=[]
for i in range(2,331337):
    #print(df.loc[i])
    if not ((df.loc[i-1])['seller_no']==(df.loc[i])['seller_no'] and (df.loc[i-1])['product_no']==(df.loc[i])['product_no'] and (df.loc[i-1])['warehouse_no']==(df.loc[i])['warehouse_no']):
        print(i)
        
