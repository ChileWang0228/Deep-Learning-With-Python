import os
import time
import json
import tqdm
from config import Config
from model import Model
from utils import DataIterator
from optimization import create_optimizer
import numpy as np
from bert import tokenization
import pandas as pd
from tensorflow.contrib.crf import viterbi_decode
import tensorflow as tf


gpu_id = 7
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

result_data_dir = Config().new_data_process_quarter_final
print('GPU ID: ', str(gpu_id))
print('Model Type: ', Config().model_type)
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('bilstm embedding ', Config().lstm_dim)
print('use original bert ', Config().use_origin_bert)


def train(train_iter, test_iter, config):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = Model(config)  # 读取模型结构图

            # 超参数设置
            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)

            normal_optimizer = tf.train.AdamOptimizer(learning_rate)  # 下接结构的学习率

            all_variables = graph.get_collection('trainable_variables')
            word2vec_var_list = [x for x in all_variables if 'bert' in x.name]  # BERT的参数
            normal_var_list = [x for x in all_variables if 'bert' not in x.name]  # 下接结构的参数
            print('bert train variable num: {}'.format(len(word2vec_var_list)))
            print('normal train variable num: {}'.format(len(normal_var_list)))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int(train_iter.num_records / config.batch_size * config.train_epoch)
            embed_step = tf.Variable(0, name='step', trainable=False)
            if word2vec_var_list:  # 对BERT微调
                print('word2vec trainable!!')
                word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                    model.loss, config.embed_learning_rate, num_train_steps=num_batch,
                    num_warmup_steps=int(num_batch * 0.05) , use_tpu=False ,  variable_list=word2vec_var_list
                )

                train_op = tf.group(normal_op, word2vec_op)  # 组装BERT与下接结构参数
            else:
                train_op = normal_op

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(config.model_dir, "runs_" + str(gpu_id), timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
                json.dump(config.__dict__, file)
            print("Writing to {}\n".format(out_dir))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            if config.continue_training:
                print('recover from: {}'.format(config.checkpoint_path))
                saver.restore(session, config.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            cum_step = 0
            """
            笔者在config.py设置了200个epoch，当然不能全部跑完，一般我们跑了3~4个epoch的时候，便可以停止了。
            这么设置的目的是多保存几个模型，再通过check_F1.py来查看每次训练得到的最高F1模型，取最优模型进行预测。
            """
            for i in range(config.train_epoch):  # 训练
                for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(
                        train_iter):

                    feed_dict = {
                        model.input_x_word: input_ids_list,
                        model.input_mask: input_mask_list,
                        model.input_relation: label_ids_list,
                        model.input_x_len: seq_length,

                        model.keep_prob: config.keep_prob,
                        model.is_training: True,
                    }

                    _, step, _, loss, lr = session.run(
                            fetches=[train_op,
                                     global_step,
                                     embed_step,
                                     model.loss,
                                     learning_rate
                                     ],
                            feed_dict=feed_dict)


                    if cum_step % 10 == 0:
                        format_str = 'step {}, loss {:.4f} lr {:.5f}'
                        print(
                            format_str.format(
                                step, loss, lr)
                        )
                    cum_step += 1

                P, R = set_test(model, test_iter, session)
                F = 2 * P * R / (P + R)
                print('dev set : step_{},precision_{},recall_{}'.format(cum_step, P, R))
                if F > 0:  # 保存F1大于0的模型
                    saver.save(session, os.path.join(out_dir, 'model_{:.4f}_{:.4f}'.format(P, R)),
                               global_step=step)


def get_text_and_label(input_tokens_list, y_list):
    """
    还原每一条数据的文本的标签
    :return:
    """
    temp = []
    for batch_y_list in y_list:
        temp += batch_y_list
    y_list = temp

    y_label_list = []  # 标签
    for i, input_tokens in enumerate(input_tokens_list):
        ys = y_list[i]  # 每条数据对应的数字标签列表
        temp = []
        label_list = []
        for index, num in enumerate(ys):

            if  num == 4 and len(temp) == 0:
                temp.append(input_tokens[index])
            elif num == 5 and len(temp) > 0:
                temp.append(input_tokens[index])
            elif len(temp) > 0:
                label = "".join(temp)
                if len(set(label)) > 1:  # 干掉单字重复情况
                    label_list.append("".join(temp))

                temp = []

        y_label_list.append(";".join(label_list))

    return y_list, y_label_list


def decode(logits, lengths, matrix):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * Config().relation_num + [0]])
    # print('length:', lengths)
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)

        paths.append(path[1:])

    return paths

def set_operation(row):
    content_list = row.split(';')
    content_list_after_set = list(set(content_list))
    return ";".join(content_list_after_set)


def get_P_R_F(dev_pd):
    dev_pd = dev_pd.fillna("0")
    dev_pd['y_pred_label'] = dev_pd['y_pred_label'].apply(set_operation)
    dev_pd['y_true_label'] = dev_pd['y_true_label'].apply(set_operation)
    y_true_label_list = list(dev_pd['y_true_label'])
    y_pred_label_list = list(dev_pd['y_pred_label'])
    print(y_pred_label_list)
    TP = 0
    FP = 0
    FN = 0
    for i, y_true_label in enumerate(y_true_label_list):
        y_pred_label = y_pred_label_list[i].split(';')
        y_true_label = y_true_label.split(';')
        current_TP = 0
        for y_pred in y_pred_label:
            if y_pred in y_true_label:
                current_TP += 1
            else:
                FP += 1
        TP += current_TP
        FN += (len(y_true_label) - current_TP)

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    try:
        F = 2 * P * R / (P + R)
    except:
        F = 0
    return P, R, F


def set_test(model, test_iter, session):

    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    y_true_list = []
    ldct_list_tokens = []
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(
            test_iter):

        feed_dict = {
            model.input_x_word: input_ids_list,
            model.input_x_len: seq_length,
            model.input_relation: label_ids_list,
            model.input_mask: input_mask_list,

            model.keep_prob: 1,
            model.is_training: False,
        }

        lengths, logits, trans = session.run(
            fetches=[model.lengths, model.logits, model.trans],
            feed_dict=feed_dict
        )

        predict = decode(logits, lengths, trans)
        y_pred_list.append(predict)
        y_true_list.append(label_ids_list)
        ldct_list_tokens.append(tokens_list)


    ldct_list_tokens = np.concatenate(ldct_list_tokens)
    ldct_list_text = []
    for tokens in ldct_list_tokens:
        text = "".join(tokens)
        ldct_list_text.append(text)

    # 获取验证集文本及其标签
    y_pred_list, y_pred_label_list = get_text_and_label(ldct_list_tokens, y_pred_list)
    y_true_list, y_true_label_list = get_text_and_label(ldct_list_tokens, y_true_list)

    print(len(y_pred_label_list))
    print(len(y_true_label_list))

    dict_data = {
        'y_true_label': y_true_label_list,
        'y_pred_label': y_pred_label_list,
        'y_pred_text': ldct_list_text
    }
    df = pd.DataFrame(dict_data)
    precision, recall, f1 = get_P_R_F(df)

    print('precision: {}, recall {}, f1 {}'.format(precision, recall, f1))

    return precision, recall


if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file  # 通用词典
    do_lower_case = False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'train.txt', use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    print('GET!!')
    dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert, tokenizer=tokenizer,
                            seq_length=config.sequence_length, is_test=True)

    train(train_iter, dev_iter, config)
