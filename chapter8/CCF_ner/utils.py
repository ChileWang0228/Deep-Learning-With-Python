import numpy as np
from bert import tokenization
from tqdm import tqdm
from config import Config

def load_data(data_file):
    """
    读取BIO的数据
    :param file:
    :return:
    """
    with open(data_file) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            # if len(contends) == 0 and words[-1] == '。':
            if len(contends) == 0:
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
    return lines


def create_example(lines):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s" % i
        text = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[0])
        examples.append(InputExample(guid=guid, text=text, label=label))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


def get_labels():
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]", '']


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class DataIterator:
    """
    数据迭代器
    """
    def __init__(self, batch_size, data_file, tokenizer, use_bert=False, seq_length=100, is_test=False,):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test

        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        self.label_map = {}
        for (i, label) in enumerate(get_labels(), 1):
            self.label_map[label] = i
        self.unknow_tokens = self.get_unk_token()

        print(self.unknow_tokens)
        print(self.num_records)

    def get_unk_token(self):
        unknow_token = set()
        for example_idx in self.all_idx:
            textlist = self.data[example_idx].text.split(' ')

            for i, word in enumerate(textlist):
                token = self.tokenizer.tokenize(word)

                if '[UNK]' in token:
                    unknow_token.add(word)
        return unknow_token

    def convert_single_example(self, example_idx):
        textlist = self.data[example_idx].text.split(' ')
        labellist = self.data[example_idx].label.split(' ')
        tokens = textlist  # 区分大小写
        labels = labellist

        if len(tokens) >= self.seq_length - 1:
            tokens = tokens[0:(self.seq_length - 2)]
            labels = labels[0:(self.seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[CLS]"])

        upper_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                        ]
        for i, token in enumerate(tokens):
            if token in self.unknow_tokens and token not in upper_letter:
                token = '[UNK]'
                ntokens.append(token)  # 全部转换成小写, 方便BERT词典
            else:
                ntokens.append(token.lower())  # 全部转换成小写, 方便BERT词典
            segment_ids.append(0)
            label_ids.append(self.label_map[labels[i]])

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ntokens.append("[SEP]")

        segment_ids.append(0)
        label_ids.append(self.label_map["[SEP]"])

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.seq_length :
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("**NULL**")
            tokens.append("**NULL**")

        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        assert len(label_ids) == self.seq_length
        assert len(tokens) == self.seq_length
        return input_ids, input_mask, segment_ids, label_ids, tokens

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if self.is_test == False:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        label_ids_list = []
        tokens_list = []

        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, label_ids, tokens = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_ids_list.append(label_ids)
            tokens_list.append(tokens)

            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break

        return input_ids_list, input_mask_list, segment_ids_list, label_ids_list, self.seq_length, tokens_list


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case =False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print(tokenizer.vocab)
    print(len(tokenizer.vocab))

    # data_iter = DataIterator(config.batch_size, data_file= config.dir_with_mission + 'train.txt', use_bert=True,
    #                         seq_length=config.sequence_length, tokenizer=tokenizer)
    #
    # dev_iter = DataIterator(config.batch_size, data_file=config.dir_with_mission + 'dev.txt', use_bert=True,
    #                          seq_length=config.sequence_length, tokenizer=tokenizer, is_test=True)
    dev_iter = DataIterator(config.batch_size, data_file=config.new_data_process_quarter_final + 'test.txt', use_bert=config.use_bert,
                            tokenizer=tokenizer,
                            seq_length=config.sequence_length, is_test=True)
    i = 0
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm(dev_iter):
        i += 1



