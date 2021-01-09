import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import os
import numpy as np
import random
import sys
from helper import to_tensor

parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
# from helper import to_tensor
from modules import Encoder, MixVariation, Decoder
from .poemwae import PoemWAE


class PoemWAE_GMP(PoemWAE):
    def __init__(self, config, api, PAD_token=0):
        super(PoemWAE_GMP, self).__init__(config, api, PAD_token)
        self.n_components = config.n_prior_components  # 5 --> 3
        self.gumbel_temp = config.gumbel_temp

        # 由于Poem这里context是title和last sentence双向GRU编码后的直接cat，4*hidden
        # 原：Variation(config.n_hidden * 4, config.z_size)  # p(e|c)
        self.prior_net = MixVariation(input_size=config.n_hidden*4, z_size=config.z_size,
                                      n_components=self.n_components, dropout_rate=config.dropout,
                                      init_weight=config.init_weight)  # p(e|c)

    # 输入的数据均为同一种情感
    def align(self, valid_loader):
        self.seq_encoder.eval()
        self.decoder.eval()
        choice_statistic = [0.0 for _ in range(self.n_components)]
        while True:
            # batch是一个以情感为key的dict
            batch = valid_loader.next_sentiment_batch()
            if batch is None:
                break

            title, context, target, target_lens, sentiments = batch
            title, context, target, target_lens = \
                to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens)

            title_last_hidden, _ = self.seq_encoder(title)
            context_last_hidden, _ = self.seq_encoder(context)
            c = torch.cat((title_last_hidden, context_last_hidden), 1)  # (batch, 2 * hidden_size * 2)
            current_statistic = self.sample_code_prior_sentiment(c, True)
            choice_statistic = [choice_statistic[i] + current_statistic[i] for i in range(self.n_components)]

        print("%s distribution: %s" % (valid_loader.name, str(choice_statistic)[1: -1]))



    


