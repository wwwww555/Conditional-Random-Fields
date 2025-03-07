#数据预处理

import sys
import json
from transformers import BertTokenizer
import os

BERT_VOCAB = "bert-base-uncased"
MAX_SEQ_LENGTH = 128


def write_in_hsln_format(input_data, hsln_format_txt_dirpath, tokenizer):
    final_string = ''
    filename_sent_boundries = {}

    # 遍历每个文件
    for file in input_data:
        # 为每个文件生成唯一的id (如果原始数据没有id，可以用其它逻辑生成)
        file_name = f"file_{input_data.index(file)}"  # 生成一个唯一的文件ID

        # 更新final_string和filename_sent_boundries
        final_string += '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}

        # 遍历annotations中的每个result
        for annotation in file['annotations'][0]['result']:
            start = annotation['value']['start']
            end = annotation['value']['end']
            filename_sent_boundries[file_name]['sentence_span'].append([start, end])

            # 获取文本和标签
            sentence_txt = annotation['value']['text']
            sentence_label = annotation['value']['labels'][0]

            # 清理文本内容
            sentence_txt = sentence_txt.replace("\r", "")

            # 确保句子文本非空
            if sentence_txt.strip() != "":
                # 使用tokenizer进行分词，并限制最大长度
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
                sent_tokens = [str(i) for i in sent_tokens]  # 将token转化为字符串
                sent_tokens_txt = " ".join(sent_tokens)

                # 将分词结果和标签写入final_string
                final_string += sentence_label + "\t" + sent_tokens_txt + "\n"

        # 添加空行作为分隔
        final_string += "\n"

    # 将最终字符串写入指定路径
    os.makedirs(os.path.dirname(hsln_format_txt_dirpath), exist_ok=True)
    with open(hsln_format_txt_dirpath, "w+", encoding="utf-8") as file:
        file.write(final_string)


def tokenize():
    [_, train_input_json, dev_input_json, test_input_json] = sys.argv

    # 使用BertTokenizer进行分词
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    # 加载JSON文件
    train_json_format = json.load(open(train_input_json, 'r', encoding='utf-8'))
    dev_json_format = json.load(open(dev_input_json, 'r', encoding='utf-8'))
    test_json_format = json.load(open(test_input_json, 'r', encoding='utf-8'))

    # 调用write_in_hsln_format函数将数据写入文件
    write_in_hsln_format(train_json_format, 'datasets/train_scibert.txt',
                         tokenizer)
    write_in_hsln_format(dev_json_format, 'datasets/dev_scibert.txt',
                         tokenizer)
    write_in_hsln_format(test_json_format, 'datasets/test_scibert.txt',
                         tokenizer)


tokenize()
