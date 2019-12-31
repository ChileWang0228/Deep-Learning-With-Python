import os
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/home/wangzhili/chilewang/clean_ccf_ner")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config
from train_fine_tune import decode, get_text_and_label
config = Config()


def vote_ensemble(path, dataset, output_path, remove_list):
    single_model_list = [x for x in os.listdir(path) if dataset + '_result_detail' in x]
    print('ensemble from file: ')
    for file_name in single_model_list:
        print(file_name)

    pred_list = OrderedDict()
    ldct_list = []
    text_index = -1  # 保证加入的ldct不是ernie模型的
    for index, file in enumerate(single_model_list):
        if file not in remove_list:  # 预测所有模型
            text_index = index
            print(index)
            print('Text File: ', file)
            break
    print('Ensembling.....')
    for index, file in enumerate(single_model_list):

        if file in remove_list:
            # print('remove file: ', file)
            continue
        print('Ensemble file:', file)
        with open(path + file) as f:
            for i, line in tqdm(enumerate(f.readlines())):
                item = json.loads(line)

                if i not in pred_list:
                    pred_list[i] = []
                pred_list[i].append(item['pred'])
                if index == text_index:
                    ldct_list.append(item['ldct_list'])


    print(len(pred_list))
    print(len(ldct_list))
    y_pred_list = []
    print('Getting Result.....')
    for key in tqdm(pred_list.keys()):
        pred_key = np.concatenate(pred_list[key]) # 3维
        j = 0
        temp_list = []
        for i in range(config.batch_size):
            temp = []
            while True:
                try:
                    temp.append(pred_key[j])
                    j += config.batch_size
                except:
                    j = 0
                    j += i + 1
                    break

            temp_T = np.array(temp).T  # 转置
            pred = []
            for line in temp_T:
                pred.append(np.argmax(np.bincount(line)))  # 找出列表中出现次数最多的值
            temp_list.append(pred)
        y_pred_list.append(temp_list)

    ldct_list_tokens = np.concatenate(ldct_list)
    # print(ldct_list)
    ldct_list_text = []
    for tokens in tqdm(ldct_list_tokens):
        text = "".join(tokens)
        ldct_list_text.append(text)
    # 测试集
    print(len(ldct_list_tokens))
    y_pred_list, y_pred_label_list = get_text_and_label(ldct_list_tokens, y_pred_list)

    print(len(y_pred_label_list))
    dict_data = {
        'y_pred_label_list': y_pred_label_list,
        'ldct_list_text': ldct_list_text,
    }
    df = pd.DataFrame(dict_data)
    df = df.fillna("0")
    df.to_csv(output_path + 'test_result.csv', encoding='utf-8')

def score_average_ensemble(path, dataset, output_path, remove_list):

    single_model_list = [x for x in os.listdir(path) if dataset + '_result_detail' in x]
    print('ensemble from file: ', len(single_model_list))
    for file_name in single_model_list:
        print(file_name)
    logits_list = OrderedDict()
    trans_list = OrderedDict()
    lengths_list = OrderedDict()
    ldct_list = []

    text_index = -1
    for index, file in enumerate(single_model_list):
        if file not in remove_list:  # 预测所有模型
            text_index = index
            print('Text File: ', file)
            print(text_index)
            break

    for index, file in enumerate(single_model_list):
        if file in remove_list:
            print('remove file: ', file)
            continue
        with open(path + file) as f:
            for i, line in tqdm(enumerate(f.readlines())):
                item = json.loads(line)

                if i not in logits_list:
                    logits_list[i] = []
                    trans_list[i] = []
                    lengths_list[i] = []

                logits_list[i].append(item['logit'])
                trans_list[i].append(item['trans'])
                lengths_list[i].append(item['lengths'])
                if index == text_index:
                    ldct_list.append(item['ldct_list'])

    y_pred_list = []
    for key in tqdm(logits_list.keys()):

        logits_key = logits_list[key]
        logits_key = np.mean(logits_key, axis=0)

        trans_key = np.array(trans_list[key])
        trans_key = np.mean(trans_key, axis=0)

        lengths_key = np.array(lengths_list[key])
        lengths_key = np.mean(lengths_key, axis=0).astype(int)

        pred = decode(logits_key, lengths_key, trans_key)
        y_pred_list.append(pred)

    ldct_list_tokens = np.concatenate(ldct_list)
    ldct_list_text = []

    for tokens in tqdm(ldct_list_tokens):
        text = "".join(tokens)
        ldct_list_text.append(text)

    # 测试集
    print(len(ldct_list_tokens))
    y_pred_list, y_pred_label_list = get_text_and_label(ldct_list_tokens, y_pred_list)

    print(len(y_pred_label_list))
    dict_data = {
        'y_pred_label_list': y_pred_label_list,
        'ldct_list_text': ldct_list_text,
    }
    df = pd.DataFrame(dict_data)
    df = df.fillna("0")
    df.to_csv(output_path + 'test_result.csv', encoding='utf-8')


if __name__ == '__main__':

    remove_list = [
        'test_result_detail_model_0.6774_0.7129-6235.txt',  # 原生bert768 LSTM 256  F1=0.48 315kb
        'test_result_detail_model_0.6841_0.6867-4988.txt',  # 动态融合 LSTM=128  F1=0.496 313kb
                   ]
    # 测试集
    # score_average_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)
    vote_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)

    # 验证集
    # vote_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)
    # score_average_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)

