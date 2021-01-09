# -*- coding: utf8 -*-
import csv
import pickle as pkl
import json
import sys
father_path = sys.path.append('../')


# 从txt文件中读出数据，utf-8编码后放到一个大list里
def read_sentiment_data(filename, dim):
    """
    input: filename
    output:list of data within filename
    """

    File = open(filename, 'r')
    Data = []
    if dim == 1:
        for line in File:
            # Data.append(line.strip().decode('utf8'))
            Data.append(line.strip())  # python3 不再编码，直接append
        
    elif dim == 2:
        for i, line in enumerate(File):
            line = line.strip()
            _line = line.split('\t')
            title = _line[0]
            contentList = _line[1].split('。')
            content1 = contentList[0] + '。'  # 第一句
            content2 = contentList[1] + '。'  # 第二句
            sentiment1 = _line[2][:2]
            sentiment2 = _line[2][-2:]
            List = []
            List.append(title)
            List.append(content1)
            List.append(content2)
            List.append(sentiment1)
            List.append(sentiment2)
            # Data.append(line.strip().decode('utf8').split())
            Data.append(List)
    else:
        raise("unvalid dimension")
    File.close()
    return Data


def read_data(filename, type):
    # 从txt文件中读出数据，utf-8编码后放到一个大list里
    """
    input: filename
    output:list of data within filename
    """

    File = open(filename, 'r')
    Data = []
    if type == 1:
        # import pdb
        # pdb.set_trace()
        f = File.read().split('\n')
        for line in f:
            # Data.append(line.strip().decode('utf8'))
            Data.append(line.strip())  # python3 不再编码，直接append

    elif type == 2:
        for line in File:
            line = line.strip()
            _line = line.split('\t')
            title = _line[0]
            contentList = _line[1].split('。')
            content1 = contentList[0] + '。'  # 第一句
            content2 = contentList[1] + '。'  # 第二句
            List = []
            List.append(title)
            List.append(content1)
            List.append(content2)

            # Data.append(line.strip().decode('utf8').split())
            Data.append(List)

    elif type == 3:
        for line in File:
            line = line.strip()
            _line = line.split('\t')
            title = _line[0]
            contentList = _line[1].split('。')
            content1 = contentList[0] + '。'  # 第一句
            content2 = contentList[1] + '。'  # 第二句
            sentiment1 = _line[2][:2]
            sentiment2 = _line[2][-2:]
            List = []
            List.append(title)
            List.append(content1)
            List.append(content2)
            List.append(sentiment1)
            List.append(sentiment2)
            # Data.append(line.strip().decode('utf8').split())
            Data.append(List)
    else:
        raise ("unvalid dimension")
    File.close()

    return Data


def read_all_poems(filename):
    File = open(filename, 'r')
    Data = []
    for line in File:
        line = line.strip()
        _line = line.split('\t')
        contentList = _line[1].split('。')
        # import pdb
        # pdb.set_trace()
        content1 = contentList[0] + '。'  # 第一句
        content2 = contentList[1] + '。'  # 第二句
        sentence1 = content1[:8]
        sentence2 = content1[8:]
        sentence3 = content2[:8]
        sentence4 = content2[8:]
        Data.append(sentence1)
        Data.append(sentence2)
        Data.append(sentence3)
        Data.append(sentence4)

    File.close()
    return Data


def prepare_sentiment_poem(data):
    """
        convert poems to training pairs
        """
    Data = dict()
    valid_data = data  # data 全部拿来做valid
    valid_s = []

    # prepare data for validating
    v = ''
    for poem in valid_data:
        if poem[0] != v:
            v = poem[0]
            valid_s.append([list(poem[0]),
                            list(poem[0]),
                            list(poem[1])[:int(len(list(poem[1])) / 2)],
                            list(poem[3][0])])
            valid_s.append([list(poem[0]),
                            list(poem[1])[:int(len(list(poem[1])) / 2)],
                            list(poem[1])[int(len(list(poem[1])) / 2):],
                            list(poem[3][1])])
            valid_s.append([list(poem[0]),
                            list(poem[1])[int(len(list(poem[1])) / 2):],
                            list(poem[2])[:int(len(list(poem[1])) / 2)],
                            list(poem[4][0])])
            valid_s.append([list(poem[0]),
                            list(poem[2])[:int(len(list(poem[2])) / 2)],
                            list(poem[2])[int(len(list(poem[2])) / 2):],
                            list(poem[4][1])])
        else:  # 如果这首新诗的题目和上一首一样，则不再添加从题目到第一句的训练样本
            print("Found poems with the same title")
            valid_s.append([list(poem[0]),
                            list(poem[1])[:int(len(list(poem[1])) / 2)],
                            list(poem[1])[int(len(list(poem[1])) / 2):],
                            list(poem[3][1])])
            valid_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1])) / 2):],
                            list(poem[2])[:int(len(list(poem[1])) / 2)],
                            list(poem[4][0])])
            valid_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2])) / 2)],
                            list(poem[2])[int(len(list(poem[2])) / 2):],
                            list(poem[4][1])])
    # import pdb
    # pdb.set_trace()
    Data['valid'] = valid_s
    return Data


def prepare_poem(data, train_lens, val_lens, test_lens, type=1):
    """
    convert poems to training pairs
    """
    Data = dict()
    train_data = data[:train_lens]  # 取前train_lens个数据
    valid_data = data[train_lens:train_lens+val_lens]  # 取接下来val_lens个数据
    test_data = data[-test_lens:]  # 取后test_lens个数据
    train_s = []
    valid_s = []
    test_s = []

    if type == 1:
        # prepare data for training
        t = ''
        count = 0
        for poem in train_data:
            if poem[0] != t:  # 不为空。。。。
                t = poem[0]  # title
                # title，title和第一句（也就是数据集里面的前半句）
                train_s.append([list(poem[0]), list(poem[0]), list(poem[1])[:int(len(list(poem[1]))/2)]])
                # title，第一句和第二句
                train_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1]))/2)], list(poem[1])[int(len(list(poem[1]))/2):]])
                # title，第二句和第三句
                train_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1]))/2):], list(poem[2])[:int(len(list(poem[1]))/2)]])
                # title，第三句和第四句
                train_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2]))/2)], list(poem[2])[int(len(list(poem[2]))/2):]])
            else:
                # print("Found poem with same title: {}".format(t))
                count += 1
                train_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1]))/2)], list(poem[1])[int(len(list(poem[1]))/2):]])
                train_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1]))/2):], list(poem[2])[:int(len(list(poem[1]))/2)]])
                train_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2]))/2)], list(poem[2])[int(len(list(poem[2]))/2):]])
        # import pdb
        # pdb.set_trace()

        # prepare data for validating
        v = ''
        for poem in valid_data:
            if poem[0] != v:
                v = poem[0]
                valid_s.append([list(poem[0]), list(poem[0]), list(poem[1])[:int(len(list(poem[1]))/2)]])
                valid_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1]))/2)], list(poem[1])[int(len(list(poem[1]))/2):]])
                valid_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1]))/2):], list(poem[2])[:int(len(list(poem[1]))/2)]])
                valid_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2]))/2)], list(poem[2])[int(len(list(poem[2]))/2):]])
            else:
                valid_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1]))/2)], list(poem[1])[int(len(list(poem[1]))/2):]])
                valid_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1]))/2):], list(poem[2])[:int(len(list(poem[1]))/2)]])
                valid_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2]))/2)], list(poem[2])[int(len(list(poem[2]))/2):]])

        # prepare data for testing
        for poem in test_data:
            if len(poem) == 3:
                # 两遍题目，接上原诗，只是显示原诗而已，不是做teacher forcing
                test_s.append([list(poem[0]), list(poem[0]), list(poem[1])[:int(len(list(poem[1]))/2)], list(poem[1])[int(len(list(poem[1]))/2):], list(poem[2])[:int(len(list(poem[2]))/2)], list(poem[2])[int(len(list(poem[2]))/2):]])
                # line1, line2, line3, line4 is for showing groundth during, not for hinting the decoding process,

    elif type == 2:
        # prepare data for training
        t = ''
        count = 0
        for poem in train_data:
            if poem[0] != t:  # 不为空。。。。
                t = poem[0]  # title
                # title，title和第一句（也就是数据集里面的前半句）
                train_s.append([list(poem[0]), list(poem[0]),
                                list(poem[1])[:int(len(list(poem[1])) / 2)], list(poem[3][0])])
                # title，第一句和第二句
                train_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1])) / 2)],
                                list(poem[1])[int(len(list(poem[1])) / 2):], list(poem[3][1])])
                # title，第二句和第三句
                train_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1])) / 2):],
                                list(poem[2])[:int(len(list(poem[1])) / 2)], list(poem[4][0])])
                # title，第三句和第四句
                train_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2])) / 2)],
                                list(poem[2])[int(len(list(poem[2])) / 2):], list(poem[4][1])])
            else:
                # print("Found poem with same title: {}".format(t))
                count += 1
                train_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1])) / 2)],
                                list(poem[1])[int(len(list(poem[1])) / 2):], list(poem[3][1])])
                train_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1])) / 2):],
                                list(poem[2])[:int(len(list(poem[1])) / 2)], list(poem[4][0])])
                train_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2])) / 2)],
                                list(poem[2])[int(len(list(poem[2])) / 2):], list(poem[4][1])])
        # import pdb
        # pdb.set_trace()

        # prepare data for validating
        v = ''
        for poem in valid_data:
            if poem[0] != v:
                v = poem[0]
                valid_s.append([list(poem[0]), list(poem[0]),
                                list(poem[1])[:int(len(list(poem[1])) / 2)], list(poem[3][0])])
                valid_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1])) / 2)],
                                list(poem[1])[int(len(list(poem[1])) / 2):], list(poem[3][1])])
                valid_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1])) / 2):],
                                list(poem[2])[:int(len(list(poem[1])) / 2)], list(poem[4][0])])
                valid_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2])) / 2)],
                                list(poem[2])[int(len(list(poem[2])) / 2):], list(poem[4][1])])
            else:
                valid_s.append([list(poem[0]), list(poem[1])[:int(len(list(poem[1])) / 2)],
                                list(poem[1])[int(len(list(poem[1])) / 2):], list(poem[3][1])])
                valid_s.append([list(poem[0]), list(poem[1])[int(len(list(poem[1])) / 2):],
                                list(poem[2])[:int(len(list(poem[1])) / 2)], list(poem[4][0])])
                valid_s.append([list(poem[0]), list(poem[2])[:int(len(list(poem[2])) / 2)],
                                list(poem[2])[int(len(list(poem[2])) / 2):], list(poem[4][1])])

        # prepare data for testing
        for poem in test_data:
            if len(poem) == 3:
                # 两遍题目，接上原诗，只是显示原诗而已，不是做teacher forcing
                test_s.append([list(poem[0]), list(poem[0]), list(poem[1])[:int(len(list(poem[1])) / 2)],
                               list(poem[1])[int(len(list(poem[1])) / 2):], list(poem[2])[:int(len(list(poem[2])) / 2)],
                               list(poem[2])[int(len(list(poem[2])) / 2):]])

    Data['train'] = train_s
    Data['valid'] = valid_s
    Data['test'] = test_s
    # import pdb
    # pdb.set_trace()
    return Data    


def prepare_test_data(data):
    test_s = []
    for poem in data:
        test_s.append([list(poem)])
    return test_s


if __name__ == '__main__':
    data_path = "../../final_data/train_data.txt"
    output_path = "../../final_data/train_data_with_sentiment.txt"
    output_file = open(output_path, 'w')
    data = read_all_poems(data_path)
    sub_len = len(data) // 100
    data = [data[i: i + sub_len] for i in range(0, len(data), sub_len)]
    # import pdb
    # pdb.set_trace()
    c_neg, c_wneg, c_neu, c_wpos, c_pos = 0, 0, 0, 0, 0
    for sub_data in data:
        test_all_poems(sub_data, output_file)
        # _, neg, wneg, neutral, wpos, pos = test_all_poems(sub_data, output_file)
    #     c_neg += neg
    #     c_wneg += wneg
    #     c_neu += neutral
    #     c_wpos += wpos
    #     c_pos += pos
    # c_all = c_neg + c_wneg + c_neu + c_wpos + c_pos
    # print("neg:%d, wneg:%d, neu:%d, wpos:%d, pos:%d, \n"
    #       "neg:%.2f, wneg:%.2f, neu:%.2f, wpos:%.2f, pos:%.2f"
    #       % (c_neg, c_wneg, c_neu, c_wpos, c_pos,
    #          c_neg/c_all, c_wneg/c_all, c_neu/c_all, c_wpos/c_all, c_pos/c_all))

    output_file.close()

    # neg:2928, wneg:49238, neu:179950, wpos:54173, pos:1711,
    # neg:0.01, wneg:0.17, neu:0.62, wpos:0.19, pos:0.01
    # 负向比中性比正向的比例为 18: 62: 20