import sys
sys.path.append("/home/wangzhili/chilewang/CCF_ner")   # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config
import pandas as pd
import re
"""
获取融合结果的后处理结果
"""

config = Config()
ensemble_result_file = config.ensemble_result_file
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


def op(row):
    if str(row) == 'nan':
        row = ''
    return row


def islegitimate(x):
    if re.findall("\\"+"|\\".join(add_char), x):
        return False

    if x in old_entities:
        return False

    return True


def mark_op(label_list):
    """
    注意事项：后处理之后删除单字实体
    """

    """
    场景1：extra_chars = set("!,:@_！，：。[]")  # 直接干掉
    """
    extra_chars = set("!,:@_！，：。[]")
    flag = True
    for i, label in enumerate(label_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                for ec in extra_chars:
                    if ec in li:
                        flag = False
                        break
                if flag:
                    temp.append(li)
                flag = True
            label_list[i] = ";".join(temp)


    """
    场景2：extra_chars = set("·")  # 在头or尾直接舍弃
    """
    flag = True
    extra_chars = set("·")
    for i, label in enumerate(label_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                for ec in extra_chars:
                    if ec in li[0] or ec in li[-1]:
                        flag = False
                        break
                if flag:
                    temp.append(li)
                flag = True
            label_list[i] = ";".join(temp)


    """
    场景3：extra_chars = set(".")  # 去头去尾，‘.’在中间，要保证前后都是英文字符，出现中文字符则直接舍弃
    """
    flag = True
    extra_chars = set(".")
    for i, label in enumerate(label_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                for ec in extra_chars:
                    if ec in li:
                        if ec not in li[0] and ec not in li[-1]:
                            matchObj = re.match(r'[\u4e00-\u9fa5a]', li)  # 匹配中文,若匹配到则直接舍弃
                            if  matchObj:
                                flag = False
                                break

                        else:
                            li = li.replace(ec, '')
                    else:
                        flag = True
                if flag:
                    temp.append(li)
                    flag = True
            label_list[i] = ";".join(temp)

    """
    场景4：extra_chars = set("#$￥%+<=>?^`{|}~#%？《{}“”‘’【】")  # 直接替换成''
    """

    extra_chars = set("#$￥%+<=>?^`{|}~#%？《{}‘’【】*")
    for i, label in enumerate(label_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                for ec in extra_chars:
                    if ec in li:
                        li = li.replace(ec, '')
                temp.append(li)
            label_list[i] = ";".join(temp)

    """
    场景5：extra_chars = set("-&\/&")  # 去头去尾 ‘-’在实体中间是合法实体
    """
    flag = True
    extra_chars = set("-&\/&")
    for i, label in enumerate(label_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                for ec in extra_chars:
                    if ec in li:
                        if ec not in li[0] and ec not in li[-1]:
                            flag = True
                        else:
                            li = li.replace(ec, '')

                    else:
                        flag = True
                if flag:
                    temp.append(li)
                flag = True
            label_list[i] = ";".join(temp)


    """
    场景6：extra_chars = set("()（）")   # 若不是对应匹配的括号，括号半边在头与尾，替换成‘’，括号在实体中间则舍弃
    """
    flag = False
    extra_chars = set("()（）“”")
    for i, label in enumerate(label_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                if '(' not in li and  ')' not in li and '（' not in li and  '）'not in li and '“' not in li and  '”'not in li:
                    temp.append(li)
                else:
                    if '(' in  li and  ')' in li and li.index('(') < li.index(')'):
                        temp.append(li)

                    elif '（' in li and '）' in li and li.index('（') < li.index('）'):
                        temp.append(li)

                    elif '“' in li and '”' in li and li.index('“') < li.index('”'):
                        temp.append(li)
                    else:
                        for ec in extra_chars:
                            if  len(li) > 0 and (li[0] == ec or li[-1] == ec):  # 在头or尾， 否则舍弃
                                li = li.replace(ec, '')
                                flag = True

                        if flag:
                            temp.append(li)
                            flag = False

            label_list[i] = ";".join(temp)


    """
    场景7：extra_chars = set('、；、')  # split
    """
    flag = False
    extra_chars = ['、', '；', '跟', '与']
    for i, label in enumerate(label_list):
        if len(label) > 0:
            lt = label.split(';')
            temp = []
            for li in lt:  # 对每个标签
                temp_ec = ''
                for ec in extra_chars:
                    if ec not in li:
                        flag = True
                    else:
                        temp_ec = ec
                        flag = False
                if flag:
                    temp.append(li)
                else:
                    t_l = li.split(temp_ec)
                    for ti in t_l:
                        if len(ti) > 1:
                            temp.append(ti)
            label_list[i] = ";".join(temp)
    return label_list

if __name__ == '__main__':
    old_entities = []
    train_df = pd.read_csv(config.new_data_process_quarter_final + 'new_train_df.csv', encoding="utf-8-sig")  # 训练集的预测结果与真实结果的并集
    for x in list(train_df["unknownEntities"].fillna("")):
        old_entities.extend(x.split(";"))

    common_label = ['支付宝', '比特币', '天猫', '淘宝', '京东', '优品汇', 'QQ钱包', '火币', '微众银行', '蚂蚁金服', 'APP', '超级会员', '北京', '广州', '深圳', '上海', 'CEO']
    print(common_label)
    old_entities = set(old_entities)  # 训练集中出现的实体都是已知实体
    old_entities = list(set(old_entities).union(set(common_label)))
    print(old_entities)
    print(len(old_entities))

    add_char = {']', '：', '~', '！', '%', '[', '《', '】', ';', '”', ':', '》', '？', '>', '/', '#', '。', '；', '&', '=', '，',
                '“', '【'}

    final_result = pd.read_csv(ensemble_result_file + 'final_result.csv', encoding='utf-8')
    final_result['unknownEntities'] = final_result['unknownEntities'].apply(op)
    id_list = final_result['id'].tolist()
    label_list = final_result['unknownEntities'].tolist()
    """
    场景8：删除单字情况和删除训练集存在的实体
    """
    label_list = mark_op(label_list)  # 对实体的符号进行处理
    label_list = [x.replace(",","") for x in label_list]
    for i, label in enumerate(label_list):
        if len(label) > 0:
            temp = []
            lt = label.split(';')
            for li in lt:
                if islegitimate(li) and len(li) > 1:
                    temp.append(li)
            # print(i)
            print(id_list[i])
            print(label_list[i])
            label_list[i] = ";".join(temp)
            print(label_list[i])
            print()

    dict_data = {'id': id_list, 'unknownEntities': label_list}
    final_result = pd.DataFrame(dict_data)
    final_result['unknownEntities'] = final_result['unknownEntities'].apply(set_operation)
    final_result['unknownEntities'] = final_result['unknownEntities'].apply(label_operation)  # 删除'nan'与单字情况
    final_result.to_csv(ensemble_result_file + 'post_ensemble_result.csv', index=False, encoding='utf-8')