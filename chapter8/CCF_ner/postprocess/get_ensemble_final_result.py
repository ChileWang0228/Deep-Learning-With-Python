import pandas as pd
import json
import sys
sys.path.append("/home/wangzhili/chilewang/CCF_ner")   # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config

config = Config()
data_dir = config.new_data_process_quarter_final
ensemble_result_file = config.ensemble_result_file  # 单模结果路径

test_pd = pd.read_csv(data_dir + 'new_test_df.csv', encoding='utf-8')
test_id_list = test_pd['id'].tolist()
title_list = test_pd['title'].tolist()

test_result_pd = pd.read_csv(ensemble_result_file + 'test_result.csv', encoding='utf-8')
test_cut_text_list = test_result_pd['ldct_list_text'].tolist()
y_pred_label_list = test_result_pd['y_pred_label_list'].tolist()


def set_operation(row):
    content_list = row.split(';')
    content_list_after_set = list(set(content_list))
    return ";".join(content_list_after_set)


def label_operation(row):
    """
    删除nan和长度小于2的标签
    """
    label_list = row.split(';')
    result_label = []
    for label in label_list:
        if label != 'nan' and len(label) >= 2:
            result_label.append(label)

    return ";".join(result_label)

# 对被切分的测试集进行拼接
pre_index = 0
repair_text_list = []
repair_label_list = []
with open(data_dir + 'cut_index_list.json', 'r') as f:
    load_dict = json.load(f)
    cut_index_list = load_dict['cut_index_list']

# print(y_pred_label_list)
y_pred_label_list = [str(item) for item in y_pred_label_list]
for i, seg_num in enumerate(cut_index_list):
    # seg_num: 原始句子被分为了几段

    if i == 0:
        text = "".join(test_cut_text_list[: seg_num])
        label = ";".join(y_pred_label_list[: seg_num])
        repair_text_list.append(text)
        repair_label_list.append(label)

    else:
        text = "".join(test_cut_text_list[pre_index: pre_index + seg_num])
        label = ";".join([str(label) for label in y_pred_label_list[pre_index: pre_index + seg_num]])
        repair_text_list.append(text)
        repair_label_list.append(label)
    pre_index += seg_num

dict_data = {'id': test_id_list, 'unknownEntities': repair_label_list, 'text': repair_text_list, 'title': title_list}
final_result = pd.DataFrame(dict_data)
final_result['unknownEntities'] = final_result['unknownEntities'].apply(set_operation)
final_result['unknownEntities'] = final_result['unknownEntities'].apply(label_operation)

final_result.to_csv(ensemble_result_file + 'final_result_with_text.csv', index=False, encoding='utf-8')
final_result[['id', 'unknownEntities']].to_csv(ensemble_result_file + 'final_result.csv', index=False, encoding='utf-8')

label_list = final_result['unknownEntities'].tolist()
