# -*- coding: utf-8 -*-
import pickle as pkl
from collections import Counter
import numpy as np
import data_apis.data_process as data_process


class LoadPoem(object):
    # 即便是做align实验，vocab也不能变，必须和原先完全一样，因此build vocab所使用的的数据集还是之前的train数据集
    def __init__(self, corpus_path, test_path,  max_vocab_cnt, sentiment_path=None,
                 test_align=False, word2vec=None, word2vec_dim=None):
        """
        the folder that contains the demo data
        """
        self._path = corpus_path
        self._path_test = test_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.sentiment_path = sentiment_path
        self.unk_id = None

        if test_align:
            assert sentiment_path is not None

        Data = data_process.read_data(self._path, type=2)  # 从txt文件中读出数据，utf-8编码后放到一个大list里；
                                                      # 2是处理没有sentiment引导的数据，3是处理带有sentiment引导的数据
        data = data_process.prepare_poem(Data, 68000, 2000, 2000, type=1)  # train_len, val_len, test_len

        self.vocab_corpus = self.process(data["train"])
        self.build_vocab(max_vocab_cnt)

        self.train_corpus = self.process(data["train"], type=1)
        self.valid_corpus = self.process(data["valid"], type=1)

        test_tiles = data_process.read_data(self._path_test, 1)
        data["test"] = data_process.prepare_test_data(test_tiles)
        self.test_corpus = self.process_test(data["test"])

    # 根据输入的训练对情感的不同，分布五个类别，返回一个dict
    def process_sentiment(self, sentiment_data):
        """
        prepare sentiment validating data with starting and ending label
        transfer sentiment to integer
        """
        new_utts_dict = {'1':[], '2':[], '3':[], '4':[], '5':[]}
        for l in sentiment_data:
            title = ["<s>"] + l[0] + ["</s>"]
            context = ["<s>"] + l[1] + ["</s>"]
            target = ["<s>"] + l[2] + ["</s>"]
            sentiment = l[3][0]
            new_utts_dict[sentiment].append([title, context, target, sentiment])
        return new_utts_dict

    def process(self, data, type=1):
        """
        prepare training  and  validating data with starting and ending label 
        """
        new_utts = []
        if type == 1:
            for line in data:
                title = ["<s>"] + line[0] + ["</s>"]
                context = ["<s>"] + line[1] + ["</s>"]
                target = ["<s>"] + line[2] + ["</s>"]
                new_utts.append([title, context, target])

        elif type == 2:
            for line in data:
                title = ["<s>"] + line[0] + ["</s>"]
                context = ["<s>"] + line[1] + ["</s>"]
                target = ["<s>"] + line[2] + ["</s>"]
                sentiment = line[3]
                new_utts.append([title, context, target, sentiment])

        else:
            print("Invalid type in process function")
            return

        return new_utts        

    def process_test(self, data):
        """
        prepare testing data with starting and ending label
        """
        new_utts = []
        for line in data:
            tem = list()
            tem.append(line[0])
            tem.append(["<s>"] + [line[0][0]] + ["</s>"])
            tem.append(["<s>"] + [line[0][1]] + ["</s>"])
            tem.append(["<s>"] + [line[0][2]] + ["</s>"])
            tem.append(["<s>"] + [line[0][3]] + ["</s>"])
            new_utts.append(tem)
        return new_utts  # 以输入的测试标题为topic，以及四句藏头

    def build_vocab(self, max_vocab_cnt):
        """
        prepare the character vocabulary
        """

        all_words = []
        for tokens in self.vocab_corpus:
            all_words.extend(tokens[0])
            all_words.extend(tokens[1])
            all_words.extend(tokens[2])
        top_vocab = Counter(all_words).most_common()
        # raw_vocab_size = len(vocab_count)
        # discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        top_vocab = top_vocab[0: max_vocab_cnt]

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in top_vocab]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab["<unk>"]

    def get_test_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                tem = []
                for sent in line:
                    tem.append([self.rev_vocab.get(t, self.unk_id) for t in sent])
                results.append(tem)
            return results
        id_test = _to_id_corpus(self.test_corpus)
        return {'test': id_test}

    # 返回dict，其中两元素 train和valid
    def get_poem_corpus(self, type=1):
        def _to_id_corpus(data, type=1):
            results = []
            if type == 1:
                for line in data:
                    results.append([[self.rev_vocab.get(t, self.unk_id) for t in line[0]],   # 题目
                                    [self.rev_vocab.get(t, self.unk_id) for t in line[1]],   # last sentence
                                    [self.rev_vocab.get(t, self.unk_id) for t in line[2]]])  # current sentence (target)
            else:
                for line in data:
                    # pdb.set_trace()
                    results.append([[self.rev_vocab.get(t, self.unk_id) for t in line[0]],   # 题目
                                    [self.rev_vocab.get(t, self.unk_id) for t in line[1]],   # last sentence
                                    [self.rev_vocab.get(t, self.unk_id) for t in line[2]],
                                    line[3]])  # current sentence (target)
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus, type=type)
        id_valid = _to_id_corpus(self.valid_corpus, type=type)
        return {'train': id_train, 'valid': id_valid}

    def get_sentiment_poem_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                # import pdb
                # pdb.set_trace()
                results.append([[self.rev_vocab.get(t, self.unk_id) for t in line[0]],   # 题目
                                [self.rev_vocab.get(t, self.unk_id) for t in line[1]],   # last sentence
                                [self.rev_vocab.get(t, self.unk_id) for t in line[2]],
                                line[3]])  # current sentence (target)
            return results
        # convert the corpus into ID
        sent_1_corpus = _to_id_corpus(self.sentiment_corpus['1'])
        sent_2_corpus = _to_id_corpus(self.sentiment_corpus['2'])
        sent_3_corpus = _to_id_corpus(self.sentiment_corpus['3'])
        sent_4_corpus = _to_id_corpus(self.sentiment_corpus['4'])
        sent_5_corpus = _to_id_corpus(self.sentiment_corpus['5'])

        return {'sent_1': sent_1_corpus, 'sent_2': sent_2_corpus,
                'sent_3': sent_3_corpus, 'sent_4': sent_4_corpus,
                'sent_5': sent_5_corpus}