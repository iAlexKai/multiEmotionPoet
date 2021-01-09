# -*- coding: utf-8 -*-
import numpy as np
import random

# Data feed
class LongDataLoader(object):
    """
    A special efficient data loader for TBPTT
    """
    batch_size = 0
    ptr = 0
    num_batch = None
    data_size = None
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch)

    # def _prepare_batch(self, cur_grid, prev_grid):
    def _prepare_batch(self, cur_grid, type=1):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=False, sample_rate=1):
        """
        prepare shuffled batches
        """
        self.ptr = 0
        self.batch_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.all_batch = []
        for i in range(temp_num_batch):
            self.all_batch.append(self.data[i * self.batch_size:(i + 1) * self.batch_size])
        # 只采样sample_rate分之一的batch，总数量为len // sample_rate个batch
        if sample_rate==1 and shuffle is False:
            self.batch = self.all_batch
        else:
            self.batch = random.sample(self.all_batch, int(len(self.all_batch) // sample_rate))
        # import pdb
        # pdb.set_trace()
        # left_over = self.data_size - temp_num_batch * batch_size

        # shuffle batch indexes
        # if shuffle:
        #     self._shuffle_batch_indexes()

        self.num_batch = len(self.batch)

        # if sample_rate != 1:
        #     print("%s begins with 1/%d epoch with %d batches" % (self.name, sample_rate, self.num_batch))
        # else:
        #     # print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))
        #     print("%s begins with a epoch with %d batches" % (self.name, self.num_batch))

    def next_batch(self, type=1):
        """
        output the id of the next batch
        """
        if self.ptr < self.num_batch:
            current_grid = self.batch[self.ptr]  # 里面有batch_size条训练样本[title, last_sentence, current_sentence]
            # if self.ptr > 0:
            #     prev_grid = self.batch[self.ptr-1]
            # else:
            #     prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, type=type)
        else:
            return None

    def next_batch_test(self):
        """
        prepare testing data for feeding in the CVAE-D model
        """
        if self.ptr < self.num_batch:
            # print("当前： ", self.ptr, " ", self.num_batch)
            current_grid = self.batch[self.ptr]
            if self.ptr > 0:
                prev_grid = self.batch[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_test_batch(cur_grid=current_grid)
        else:
            return None


class SWDADataLoader(LongDataLoader):
    def __init__(self, name, data, config):
        self.name = name
        self.data = data
        self.data_size = len(data)
        self.max_utt_size = config.maxlen
        self.max_de_len = config.maxlen
        self.title_size = config.title_size

    def pad_to(self, tokens, size, do_pad=True):
        """
        pad the input data to a fixed length
        """
        # 截取前size-1个字，加上</s>
        if len(tokens) >= size:
            return tokens[0: size-1] + [tokens[-1]]
        # 将0pad到后面
        elif do_pad:
            return tokens + [0] * (size - len(tokens))
        else:
            return tokens

    def sample_one_batch(self, sentiment=False):
        sample_ptr = random.randint(0, self.num_batch-1)
        current_grid = self.batch[sample_ptr]  # 里面有batch_size条训练样本[title, last_sentence, current_sentence]
        # if self.ptr > 0:
        #     prev_grid = self.batch[self.ptr-1]
        # else:
        #     prev_grid = None
        if sentiment:
            return self._prepare_sentiment_batch(cur_grid=current_grid)
        else:
            return self._prepare_batch(cur_grid=current_grid)

    def next_sentiment_batch(self):
        """
                output the id of the next batch
                """
        if self.ptr < self.num_batch:
            current_grid = self.batch[self.ptr]  # 里面有batch_size条训练样本[title, last_sentence, current_sentence]
            # if self.ptr > 0:
            #     prev_grid = self.batch[self.ptr-1]
            # else:
            #     prev_grid = None
            self.ptr += 1
            return self._prepare_sentiment_batch(cur_grid=current_grid)
        else:
            return None

    def _prepare_batch(self, cur_grid, type=1):
        """
        prepare each batch of training data and validating data
        rows中每一行是一个训练样本，包含 [标题，上一句，当前句]，4个为一组（即一首诗），一首诗对应的4个训练样本的标题是一样的
        """
        if type == 1:
            rows = cur_grid  # 当前batch
            titles, context_utts, out_utts, out_lens = [], [], [], []  # 初始化4个list
            for row in rows:  # 遍历当前batch中的每一个训练样本
                title = row[0]   # 取标题
                in_row = row[1]   # 取输入句
                out_utt = row[2]   # 取输出句
                # 分布append到当前batch的标题list、输入句（也可以理解为context）list、输出句list，并将len append到输出句长度list
                titles.append(self.pad_to(title, size=self.title_size))
                context_utts.append(self.pad_to(in_row, size=self.max_utt_size))
                out_utt = self.pad_to(out_utt, size=self.max_utt_size)
                out_lens.append(len(out_utt))
                out_utts.append(out_utt)

            # 将4个list转为4个np的vector
            vec_title = np.zeros((self.batch_size, self.title_size), dtype=np.int64)
            vec_context = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int64)
            vec_outs = np.zeros((self.batch_size, self.max_de_len), dtype=np.int64)
            vec_out_lens = np.array(out_lens)

            for b_id in range(self.batch_size):
                vec_title[b_id, :] = np.array(titles[b_id])
                vec_outs[b_id, 0: vec_out_lens[b_id]] = out_utts[b_id][0: vec_out_lens[b_id]]
                vec_context[b_id, :] = np.array(context_utts[b_id])
            return vec_title, vec_context, vec_outs, vec_out_lens  # 将4个vector返回给模型训练

        elif type == 2:
            rows = cur_grid  # 当前batch
            titles, context_utts, out_utts, out_lens, out_sentiment_lead = [], [], [], [], []  # 初始化4个list
            for row in rows:  # 遍历当前batch中的每一个训练样本
                title = row[0]  # 取标题
                in_row = row[1]  # 取输入句
                out_utt = row[2]  # 取输出句

                out_sentiment = int(row[3][0])-1
                titles.append(self.pad_to(title, size=self.title_size))
                context_utts.append(self.pad_to(in_row, size=self.max_utt_size))
                out_utt = self.pad_to(out_utt, size=self.max_utt_size)
                out_lens.append(len(out_utt))
                out_utts.append(out_utt)

                if out_sentiment == 0 or out_sentiment == 1:
                    out_sentiment_lead.append(0)
                if out_sentiment == 3 or out_sentiment == 4:
                    out_sentiment_lead.append(2)
                else:
                    out_sentiment_lead.append(1)

            # 将4个list转为4个np的vector
            vec_title = np.zeros((self.batch_size, self.title_size), dtype=np.int64)
            vec_context = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int64)
            vec_outs = np.zeros((self.batch_size, self.max_de_len), dtype=np.int64)
            vec_out_lens = np.array(out_lens)
            vec_out_sentiment_lead = np.zeros(self.batch_size, dtype=np.int64)

            for b_id in range(self.batch_size):
                vec_title[b_id, :] = np.array(titles[b_id])
                vec_outs[b_id, 0: vec_out_lens[b_id]] = out_utts[b_id][0: vec_out_lens[b_id]]
                vec_context[b_id, :] = np.array(context_utts[b_id])
                # vec_out_sentiment_lead[b_id][out_sentiment_lead[b_id]] = 1
                vec_out_sentiment_lead[b_id] = out_sentiment_lead[b_id]

            return vec_title, vec_context, vec_outs, vec_out_lens, vec_out_sentiment_lead

        else:
            print("Invalid type in _prepare_batch function!")
            return

    def _prepare_sentiment_batch(self, cur_grid):
        rows = cur_grid  # 当前batch
        titles, context_utts, out_utts, out_lens, out_sentiment_mask = [], [], [], [], []  # 初始化4个list
        for row in rows:  # 遍历当前batch中的每一个训练样本
            title = row[0]  # 取标题
            in_row = row[1]  # 取输入句
            out_utt = row[2]  # 取输出句
            out_sentiment = int(row[3])-1
            # 分布append到当前batch的标题list、输入句（也可以理解为context）list、输出句list，并将len append到输出句长度list
            titles.append(self.pad_to(title, size=self.title_size))
            context_utts.append(self.pad_to(in_row, size=self.max_utt_size))
            out_utt = self.pad_to(out_utt, size=self.max_utt_size)
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            # out_sentiment_mask.append(out_sentiment)

            if out_sentiment == 0 or out_sentiment == 1:
                out_sentiment_mask.append(0)
            elif out_sentiment == 3 or out_sentiment == 4:
                out_sentiment_mask.append(2)
            else:
                out_sentiment_mask.append(1)

        # 将4个list转为4个np的vector
        vec_title = np.zeros((self.batch_size, self.title_size), dtype=np.int64)
        vec_context = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int64)
        vec_outs = np.zeros((self.batch_size, self.max_de_len), dtype=np.int64)
        vec_out_lens = np.array(out_lens)
        # vec_out_sentiment_mask = np.zeros((self.batch_size, 5), dtype=np.int64)
        vec_out_sentiment_mask = np.zeros((self.batch_size, 3), dtype=np.int64)

        # import pdb
        # pdb.set_trace()
        for b_id in range(self.batch_size):
            vec_title[b_id, :] = np.array(titles[b_id])
            vec_outs[b_id, 0: vec_out_lens[b_id]] = out_utts[b_id][0: vec_out_lens[b_id]]
            vec_context[b_id, :] = np.array(context_utts[b_id])
            vec_out_sentiment_mask[b_id][out_sentiment_mask[b_id]] = 1
        return vec_title, vec_context, vec_outs, vec_out_lens, vec_out_sentiment_mask  # 将4个vector返回给模型训练

    def _prepare_test_batch(self, cur_grid):
        """
        prepare every batch of testing data
        """
        rows = cur_grid
        titles, headers = [], []

        for row in rows:
            titles.append(self.pad_to(row[0], size=self.title_size))

        for i in range(4):
            headers.append([[cur_grid[0][i+1][1]]])

        # 将4个list转为4个np的vector
        vec_title = np.zeros((self.batch_size, self.title_size), dtype=np.int64)

        for b_id in range(self.batch_size):
            vec_title[b_id, :] = np.array(titles[b_id])

        return vec_title, headers  # 返回标题和藏头

