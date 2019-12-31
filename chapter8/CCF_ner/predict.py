from train_fine_tune import decode
from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
from utils import DataIterator

result_data_dir = Config().new_data_process_quarter_final
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_session(checkpoint_path):
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            try:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
            except OSError:
                saver = tf.train.import_meta_graph("{}.ckpt.meta".format(checkpoint_path))
            saver.restore(session, checkpoint_path)

            _input_x = graph.get_operation_by_name("input_x_word").outputs[0]
            _input_x_len = graph.get_operation_by_name("input_x_len").outputs[0]
            _input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            _input_relation = graph.get_operation_by_name("input_relation").outputs[0]
            _keep_ratio = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            _is_training = graph.get_operation_by_name('is_training').outputs[0]


            used = tf.sign(tf.abs(_input_x))
            length = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(length, tf.int32)
            logits = graph.get_operation_by_name('project/pred_logits').outputs[0]

            trans = graph.get_operation_by_name('transitions').outputs[0]

            def run_predict(feed_dict):
                return session.run([logits, lengths, trans], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _input_x_len, _input_mask, _input_relation, _keep_ratio, _is_training)

def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    ldct_list = []
    logits_list = []
    lengths_list = []
    trans_list = []
    predict_fun, feed_keys = get_session(model_file)
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(test_iter):
        # 对每一个batch的数据进行预测
        logits, lengths, trans = predict_fun(
            dict(
                zip(feed_keys, (input_ids_list, seq_length, input_mask_list, label_ids_list, 1, False))
                 )
        )

        logits_list.append(logits)
        lengths_list.append(lengths)
        trans_list.append(trans)
        pred = decode(logits, lengths, trans)
        y_pred_list.append(pred)
        ldct_list.append(tokens_list)

    """
    所需预测概率保存
    """
    if 'test' in dev_iter.data_file:
        result_detail_f = 'test_result_detail_{}.txt'.format(config.checkpoint_path.split('/')[-1])
    else:
        result_detail_f = 'dev_result_detail_{}.txt'.format(config.checkpoint_path.split('/')[-1])

    with open(config.ensemble_source_file + result_detail_f, 'w', encoding='utf-8') as detail:
        for idx in range(len(logits_list)):
            item = {}
            item['trans'] = trans_list[idx]
            item['lengths'] = lengths_list[idx]
            item['logit'] = logits_list[idx]
            item['pred'] = y_pred_list[idx]
            item['ldct_list'] = ldct_list[idx]
            detail.write(json.dumps(item, ensure_ascii=False, cls=NpEncoder) + '\n')


if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'test.txt', use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # print('Predicting dev.txt..........')
    # dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert,
    #                         seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)

    set_test(dev_iter, config.checkpoint_path)
