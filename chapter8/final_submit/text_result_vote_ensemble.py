#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
"""
对所有单模文字结果进行投票融合
""" 
# 所有单模的文字结果文件
file_list = [
     'test_result_detail_model_0.6774_0.7129-6235.csv',    # 原生bert768 LSTM 256  F1=0.48 315kb
]  
file_len = len(file_list)
print('融合文件数目:', file_len)
label_list = []
id_list = []

# 统计每条数据所有实体的出现次数
for i, file in enumerate(file_list):
    res_df = pd.read_csv(file)
    if i == 0:
        label_list = res_df['unknownEntities'].fillna('').tolist()
        id_list = res_df['id'].tolist()
    else:
        new_l_l = res_df['unknownEntities'].fillna('').tolist()
        for i, l_l in enumerate(new_l_l):
            l_l_list = l_l.split(';')
            label_list_item = label_list[i].split(';')
            label_list_item += l_l_list
#             print(label_list[i])
            label_list[i] = ";".join(label_list_item)
#             print(label_list[i])
#             print()
#         break  
def all_list(arr):
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result

# 将每条数据的实体出现次数从大到小排序
num_list = []
for i, label in enumerate(label_list):
    res = sorted(all_list(label.split(';')).items(), key=lambda d:d[1], reverse=True)
    for x in res:
        if x[0]!='':
            num_list.append(x[1])

# 融合的阈值
median = file_len // 3 + 3
print('融合的阈值:', median)

# 融合
entities_num = 0
for i, label in enumerate(label_list):
    print(id_list[i])
    res = sorted(all_list(label.split(';')).items(), key=lambda d:d[1], reverse=True)
    temp = []
    entities_list = []
    for x in res:
        if x[0]!='' and x[1] >= median:
            temp.append(x)
            entities_list.append(x[0])
    entities_num += len(entities_list)
    label_list[i] = ";".join(entities_list)
    print(label_list[i])
    print(temp)
    print()
print(entities_num)   

post_emsemble_df = pd.DataFrame({'id': id_list, 'unknownEntities': label_list})
post_emsemble_df.to_csv('post_emsemble_df_div3_14_all_34_recall_064.csv', encoding='UTF-8', index=False)