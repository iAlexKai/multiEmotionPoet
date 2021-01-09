"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import to_tensor


# 编码topic和上一句
class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidirectional, n_layers, noise_radius=0.2):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        assert type(self.bidirectional) == bool
        
        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): 
            if w.dim() > 1:
                weight_init.orthogonal_(w)
                
    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    # encoder用来编码标题或上一句，以标题为例
    # inputs: 标题句 (batch, len)
    def forward(self, inputs, input_lens=None, noise=False):

        # if self.embedding is not None:
        inputs = self.embedding(inputs)  # 过embedding
        
        batch_size, seq_len, emb_size = inputs.size()  # (batch, len, emb_size) len是12，即标题的最大长度

        # inputs = F.dropout(inputs, 0.5, self.training)  # embedding先不做dropout
        
        if input_lens is not None:
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        # inputs: (batch, len, emb_dim)
        # init_hidden: (2, batch, n_hidden)
        init_hidden = to_tensor(torch.zeros(self.n_layers * (1 + self.bidirectional), batch_size, self.hidden_size))
        # hids: (batch, len, 2 * n_hidden)
        # h_n: (2, batch, n_hidden)
        # self.rnn.flatten_parameters()
        hids, h_n = self.rnn(inputs, init_hidden)

        if input_lens is not None:
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)

        # h_n (1, 2, batch, n_hidden) 按层排列
        h_n = h_n.view(self.n_layers, (1 + self.bidirectional), batch_size, self.hidden_size)
        # 取最后一层 (2, batch, n_hidden)
        h_n = h_n[-1]  # 取last hidden的最后一层作为encoder的last hidden并返回
        # (batch_size, 1 * 2 * hidden_size) 后面全给弄到一起
        enc = h_n.transpose(1,0).contiguous().view(batch_size, -1)

        if noise and self.noise_radius > 0:
            gauss_noise = to_tensor(torch.normal(means=torch.zeros(enc.size()),std=self.noise_radius))
            enc = enc + gauss_noise
            
        return enc, hids


# class ContextEncoder(nn.Module):
#     def __init__(self, utt_encoder, input_size, hidden_size, n_layers=1, noise_radius=0.2):
#         super(ContextEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.noise_radius=noise_radius
#
#         self.n_layers = n_layers
#
#         self.utt_encoder=utt_encoder
#         self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.init_weights()
#
#     def init_weights(self):
#         for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
#             if w.dim()>1:
#                 weight_init.orthogonal_(w)
#
#     def store_grad_norm(self, grad):
#         norm = torch.norm(grad, 2, 1)
#         self.grad_norm = norm.detach().data.mean()
#         return grad
#
#     def forward(self, context, context_lens, utt_lens, floors, noise=False):
#         batch_size, max_context_len, max_utt_len = context.size()
#         utts=context.view(-1, max_utt_len)
#         utt_lens=utt_lens.view(-1)
#         utt_encs,_ = self.utt_encoder(utts, utt_lens)
#         utt_encs = utt_encs.view(batch_size, max_context_len, -1)
#
#         floor_one_hot = to_tensor(torch.zeros(floors.numel(), 2))
#         floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
#         floor_one_hot = floor_one_hot.view(-1, max_context_len, 2)
#         utt_floor_encs = torch.cat([utt_encs, floor_one_hot], 2)
#
#         utt_floor_encs = F.dropout(utt_floor_encs, 0.25, self.training)
#         context_lens_sorted, indices = context_lens.sort(descending=True)
#         utt_floor_encs = utt_floor_encs.index_select(0, indices)
#         utt_floor_encs = pack_padded_sequence(utt_floor_encs, context_lens_sorted.data.tolist(), batch_first=True)
#
#         init_hidden=to_tensor(torch.zeros(1, batch_size, self.hidden_size))
#         hids, h_n = self.rnn(utt_floor_encs, init_hidden)
#
#         _, inv_indices = indices.sort()
#         h_n = h_n.index_select(1, inv_indices)
#
#         enc = h_n.transpose(1,0).contiguous().view(batch_size, -1)
#
#         if noise and self.noise_radius > 0:
#             gauss_noise = to_tensor(torch.normal(means=torch.zeros(enc.size()),std=self.noise_radius))
#             enc = enc + gauss_noise
#         return enc


class Variation(nn.Module):
    def __init__(self, input_size, z_size, dropout_rate, init_weight):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size=z_size
        self.init_w = init_weight
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1200),
            nn.BatchNorm1d(1200, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(1200, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout_rate),
        )
        self.context_to_mu = nn.Linear(z_size, z_size)  # activation???
        self.context_to_logsigma = nn.Linear(z_size, z_size)
        
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-self.init_w, self.init_w)
            # nn.init.kaiming_normal_(m.weight.data)
            # nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size, _ = context.size()  # prior: (batch, 4 * hidden)
        context = self.fc(context)
        mu = self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context) 
        std = torch.exp(0.5 * logsigma)
        
        epsilon = to_tensor(torch.randn([batch_size, self.z_size]))
        z = epsilon * std + mu  
        return z, mu, logsigma 
    

class MixVariation(nn.Module):
    def __init__(self, input_size, z_size, n_components, dropout_rate, init_weight):
        super(MixVariation, self).__init__()
        self.input_size = input_size
        self.z_size=z_size  
        self.n_components = n_components
        self.init_w = init_weight
        self.gumbel_temp = 0.1
        
        self.pi_net = nn.Sequential(
            nn.Linear(z_size, n_components),
            nn.Dropout(dropout_rate),
            # nn.Linear(z_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            # nn.LeakyReLU(0.1),
            # nn.Linear(z_size, n_components)
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1200),
            nn.BatchNorm1d(1200, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(1200, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
        )
        self.context_to_mus = nn.Linear(z_size, n_components*z_size)
        #     nn.Linear(z_size, n_components*z_size),
        #     nn.BatchNorm1d(n_components*z_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        #     nn.Linear(n_components*z_size, n_components*z_size),
        #     nn.BatchNorm1d(n_components*z_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        # )
        # self.context_to_mu_1 = nn.Linear(z_size, z_size)
        # self.context_to_mu_2 = nn.Linear(z_size, z_size)
        # self.context_to_mu_3 = nn.Linear(z_size, z_size)
        # self.context_to_mu_4 = nn.Linear(z_size, z_size)
        # self.context_to_mu_5 = nn.Linear(z_size, z_size)
        self.context_to_logsigmas = nn.Linear(z_size, n_components*z_size)
        # self.context_to_logsigmas = nn.Sequential(
        #     nn.Linear(z_size, n_components * z_size),
        #     nn.BatchNorm1d(n_components * z_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        #     nn.Linear(n_components * z_size, n_components * z_size),
        #     nn.BatchNorm1d(n_components * z_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        # )

        # self.context_to_logsigma_1 = nn.Linear(z_size, z_size)
        # self.context_to_logsigma_2 = nn.Linear(z_size, z_size)
        # self.context_to_logsigma_3 = nn.Linear(z_size, z_size)
        # self.context_to_logsigma_4 = nn.Linear(z_size, z_size)
        # self.context_to_logsigma_5 = nn.Linear(z_size, z_size)

        self.pi_net.apply(self.init_pi_net)
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mus)
        self.init_weights(self.context_to_logsigmas)

        # self.init_weights(self.context_to_mu_1)
        # self.init_weights(self.context_to_mu_2)
        # self.init_weights(self.context_to_mu_3)
        # self.init_weights(self.context_to_mu_4)
        # self.init_weights(self.context_to_mu_5)
        # self.init_weights(self.context_to_logsigma_1)
        # self.init_weights(self.context_to_logsigma_2)
        # self.init_weights(self.context_to_logsigma_3)
        # self.init_weights(self.context_to_logsigma_4)
        # self.init_weights(self.context_to_logsigma_5)

        # self.candidates_mu = [self.context_to_mu_1, self.context_to_mu_2, self.context_to_mu_3]
        #                  self.context_to_mu_4, self.context_to_mu_5]
        # self.candidates_sigma = [self.context_to_logsigma_1, self.context_to_logsigma_2, self.context_to_logsigma_3]
        #                     self.context_to_logsigma_4, self.context_to_logsigma_5]
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-self.init_w, self.init_w)
            m.bias.data.fill_(0)

    def init_pi_net(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    # context (batch, input_size)
    # sentiment_mask (batch, 1)
    # ###################################
    # sentiment_mask不为空-->说明是标注集
    # sentimen_mask为空，mask_type不为空-->说明是测试使用某一个高斯的生成结果
    # sentimen_mask为空，mask_type为空, sentiment_lead不为空-->说明是未标注集训练，使用sentiment_lead做一级引导
    def forward(self, context, sentiment_mask=None, mask_type=None):
        batch_size, _ = context.size()
        context = self.fc(context)
        mus = self.context_to_mus(context)
        logsigmas = self.context_to_logsigmas(context)
        stds = torch.exp(0.5 * logsigmas)  # (batch, 5 * z_size)
        # epsilons (batch, 5 * z_size)
        epsilons = to_tensor(torch.randn([batch_size, self.n_components * self.z_size]))
        zi = (epsilons * stds + mus).view(batch_size, self.n_components, self.z_size)  # (batch, 5, z_size)
        pi = None
        pi_final = None

        if sentiment_mask is None:
            # import pdb
            # pdb.set_trace()
            if mask_type is not None:
                # pi = torch.zeros(batch_size, 5)
                pi = torch.zeros(batch_size, 3)

                # if mask_type == "negative":
                #     pi[:, 0:2] = torch.zeros(batch_size, 2)
                # # mask正向
                # elif mask_type == "positive":
                #     pi[:, 3:5] = torch.zeros(batch_size, 2)
                # # 强行去掉中性情感
                # # mask中性
                # elif mask_type == "neutral":
                #     pi[:, 2] = torch.zeros(batch_size)
                if mask_type == "0":
                    pi[:, 0] = 1
                elif mask_type == "1":
                    pi[:, 1] = 1
                elif mask_type == "2":
                    pi[:, 2] = 1
                # elif mask_type == "3":
                #     pi[:, 3] = 1
                # elif mask_type == "4":
                #     pi[:, 4] = 1
                else:
                    print("Mask type invalid")
                pi_final = pi.cuda()

            else:
                pi = self.pi_net(context)  # (batch, 5)

                pi_hard = F.gumbel_softmax(pi, tau=self.gumbel_temp, hard=True, eps=1e-10)
                pi_soft = F.gumbel_softmax(pi, tau=self.gumbel_temp, hard=False, eps=1e-10)
                pi_final = pi_hard - pi_soft.detach() + pi_soft

            pi_final = pi_final.unsqueeze(1)  # (batch, 1, 5)
            z = torch.bmm(pi_final, zi).squeeze(1)  # (batch, 1, z_size) --> (batch, z_size)
            mu = torch.bmm(pi_final, mus.view(batch_size, self.n_components, self.z_size))  # (batch, z_size)
            logsigma = torch.bmm(pi_final, logsigmas.view(batch_size, self.n_components, self.z_size))  # (batch, z_size)

        else:
            # mu = self.candidates_mu[force_choice](context)  # (batch, z_size)
            # logsigma = self.candidates_sigma[force_choice](context)  # (batch, z_size)
            # std = torch.exp(0.5 * logsigma)  # (batch, z_size)
            # epsilon = to_tensor(torch.randn([batch_size, self.z_size]))
            # z = epsilon * std + mu  # (batch, z_size)
            # import pdb
            # pdb.set_trace()
            sentiment_mask = sentiment_mask.unsqueeze(1)
            z = torch.bmm(sentiment_mask.float(), zi).squeeze(1)
            mu = torch.bmm(sentiment_mask.float(), mus.view(batch_size, self.n_components, self.z_size))
            logsigma = torch.bmm(sentiment_mask.float(), logsigmas.view(batch_size, self.n_components, self.z_size))  # (batch, z_size)

        return z, mu, logsigma, pi, pi_final

    
class Decoder(nn.Module):
    # Decoder(self.embedder, config.emb_size, config.n_hidden*4 + config.z_size, self.vocab_size, n_layers=1)
    def __init__(self, embedder, input_size, hidden_size, vocab_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size= input_size 
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size 

        self.embedding = embedder
        # 给decoder的init_hidden加一层非线性变换relu
        # encoder可以是双向的GRU，但decoder一定是单向的
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        for w in self.rnn.parameters():
            if w.dim()>1:
                weight_init.orthogonal_(w)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.fill_(0)

    # init_hidden: (batch, z_size + 4*hidden) --unsqueeze->  (1, batch, z_size+4*hidden)
    # self.decoder(torch.cat((z, c), 1), None, target[:, :-1], target_lens-1)
    def forward(self, init_hidden, context=None, inputs=None, lens=None):
        batch_size, maxlen = inputs.size()
        inputs = self.embedding(inputs)

        if context is not None:
            repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1)
            inputs = torch.cat([inputs, repeated_context], 2)

        # inputs = F.dropout(inputs, 0.5, self.training)

        hids, h_n = self.rnn(inputs, init_hidden.unsqueeze(0))
        decoded = self.out(hids.contiguous().view(-1, self.hidden_size))  # reshape before linear over vocab
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded

    # 生成结果，可以直接当做test的输出，也可以做evaluate的metric值（如BLEU）计算
    # init_hidden是prior_z cat c
    # self.decoder.sampling(torch.cat((prior_z, c), 1), None, self.maxlen, SOS_tok, EOS_tok, "greedy")
    def sampling(self, init_hidden, maxlen, go_id, eos_id, mode='greedy'):
        batch_size = init_hidden.size(0)  # batch_size等于调用时候的repeat
        sample_lens = np.zeros(batch_size, dtype=np.int)  # (batch中每一个测试样本的生成句的长度)

        decoder_input = to_tensor(torch.LongTensor([[go_id] * batch_size]).view(batch_size,1))
        decoder_input = self.embedding(decoder_input)
        decoder_hidden = init_hidden.unsqueeze(0)
        pred_outs = np.zeros((batch_size, maxlen), dtype=np.int64)
        # import pdb
        # pdb.set_trace()
        for di in range(maxlen - 1):

            # 为什么相同的decoder_input（重复了10遍）输入同一个decoder得到的结果不一样呢？？
            # self.rnn.flatten_parameters()
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            decoder_output = self.out(decoder_output.contiguous().view(-1, self.hidden_size))  # (batch, vocab_size)
            if mode == 'greedy':
                topi = decoder_output.max(1, keepdim=True)[1]
            elif mode == 'sample':
                topi = torch.multinomial(F.softmax(decoder_output[:,-1], dim=1), 1)

            ni = topi.squeeze().cpu().numpy()
            pred_outs[:, di] = ni
            decoder_input = self.embedding(topi)

        # 结束for生成了一句话
        for i in range(batch_size):
            for word in pred_outs[i]:
                if word == eos_id:
                    break
                sample_lens[i] = sample_lens[i] + 1
        return pred_outs, sample_lens

    # 生成结果，可以直接当做test的输出，也可以做evaluate的metric值（如BLEU）计算
    # init_hidden是prior_z cat c
    # batch_size是1，一次只测试一首诗

    # init_hidden (prior_z和c的cat)
    # max_len： config.maxlen 即10
    # SOS_tok: 即<s>对应的token
    def testing(self, init_hidden, maxlen, go_id, header, mode="greedy"):
        batch_size = init_hidden.size(0)
        assert batch_size == 1
        decoder_input = to_tensor(torch.LongTensor([[go_id]]).view(1, 1))  # (batch, 1)
        header_input = to_tensor(torch.LongTensor(header).view(1, 1))
        # input: (batch=1, len=1, emb_size)
        decoder_input = self.embedding(decoder_input)  # (batch, 1, emb_dim)
        # hidden: (batch=1, 2, hidden_size * 2)
        decoder_hidden = init_hidden.unsqueeze(0)  # (1, batch, 4*hidden+z_size)
        pred_outs = np.zeros((batch_size, maxlen), dtype=np.int64)
        for di in range(maxlen - 1):  # decode要的是从<s>后一位开始，因此总长度是max_len-1
            # 输入decoder
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)  # (1, 1, hidden)
            decoder_output = self.out(decoder_output.contiguous().view(-1, self.hidden_size))  # (1, vocab_size)
            # import pdb
            # pdb.set_trace()
            if mode == "greedy":
                topi = decoder_output.max(1, keepdim=True)[1]
            else:
                topi = decoder_output.max(1, keepdim=True)[1]
                topi = torch.multinomial(F.softmax(decoder_output[:, -1], dim=1), 1)
            # 拿到pred_outs以返回

            ni = topi.squeeze().cpu().numpy()

            # 为下一次decode准备输入字
            if di != 0:
                decoder_input = self.embedding(topi)
                pred_outs[:, di] = ni
            # 将藏头字直接当做第一个位置
            else:
                decoder_input = self.embedding(header_input)
                pred_outs[:, di] = header[0][0]
        # 结束for完成一句诗的token预测
        return pred_outs
