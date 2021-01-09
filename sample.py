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

import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import json

import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)  # add parent folder to path so as to import common modules

from data_apis.corpus import LoadPoem
from helper import indexes2sent, to_tensor
import models, experiments, configs
# from models.poemwae import PoemWAE
# from models.poemwae_gmp import PoemWAE_GMP
from experiments.metrics import Metrics

PAD_token = 0


# 等于说evaluate，需要已转成文字的(repeat, max_len, 1)，对应的长度(repeat, 1)，以及转成文字的target(max_len)

# 输入topic和前一句，生成后一句，然后计算BLEU等相关metrics的值
def evaluate(model, metrics, valid_loader, vocab, rev_vocab, f_eval, repeat, early_stop=False):
    print("Start sampling...")
    recall_bleus, prec_bleus, bows_extrema, bows_avg, bows_greedy, intra_dist1s, intra_dist2s, avg_lens, inter_dist1s, inter_dist2s\
        = [], [], [], [], [], [], [], [], [], []
    local_t = 0
    while True:
        local_t += 1
        batch = valid_loader.next_batch(type=1)

        if batch is None or (not early_stop and local_t > 1000) or (early_stop and local_t > 100):  # end of epoch
            break
        if local_t % 100 == 0:
            print("Evaluate batch %d" % local_t)
        title, context, target, target_lens = batch  # batch_size是1
        title, context = to_tensor(title), to_tensor(context)

        f_eval.write("Batch %d \n" % (local_t))  # print the context

        # f_eval.write("Context %d-%d: %s\n" % (t_id, floors[0, t_id], context_str))
        # print the true outputs    
        # ref_str, _ = indexes2sent(response[0], vocab, rev_vocab["</s>"], rev_vocab["<s>"])

        ref_tokens = [vocab[e] for e in target[0] if e not in [rev_vocab['</s>'], rev_vocab['<s>'], 0]]
        # f_eval.write("Target   >> %s\n" % (" ".join(ref_tokens)))
        f_eval.write("%s\n" % (" ".join(ref_tokens)))
        # ref_tokens = ref_str.split(' ')  # 原诗分词成list
        # 从model.sample中输出生成的结果，调用2sent后打印出来
        # 调用sample函数，负责拿到生成句和生成句的长度 (batch, len, 1)
        # (repeat, max_len, 1)  (repeat, 1)  数字

        sample_words_list, sample_lens = model.sample(title, context, repeat, rev_vocab["<s>"], rev_vocab["</s>"])
        # nparray: [repeat x seq_len]
        pred_tokens = [[vocab[e] for e in sample_words
                        if e not in [rev_vocab['</s>'], rev_vocab['<s>'], 0]]
                       for sample_words in sample_words_list]  # 诗句分词成list
        for r_id, pred_sents in enumerate(pred_tokens):
            f_eval.write("%s\n" % (' '.join(pred_sents)))
        f_eval.write("\n")

        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)

        # 计算bow和distinct时候需要用tensor
        bow_extrema, bow_avg, bow_greedy = \
            metrics.sim_bow(to_tensor(sample_words_list).detach(), sample_lens,
                            to_tensor(target[:, 1:-1]).detach(), target_lens)
        bows_extrema.append(bow_extrema)
        bows_avg.append(bow_avg)
        bows_greedy.append(bow_greedy)

        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = metrics.div_distinct(sample_words_list, sample_lens)
        intra_dist1s.append(intra_dist1)
        intra_dist2s.append(intra_dist2)
        avg_lens.append(np.mean(sample_lens))
        inter_dist1s.append(inter_dist1)
        inter_dist2s.append(inter_dist2)
                
        f_eval.write("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2 * (prec_bleu * recall_bleu) / (prec_bleu + recall_bleu + 10e-12)
    bow_extrema = float(np.mean(bows_extrema))
    bow_avg = float(np.mean(bows_avg))
    bow_greedy = float(np.mean(bows_greedy))
    intra_dist1 = float(np.mean(intra_dist1s))
    intra_dist2 = float(np.mean(intra_dist2s))
    avg_len = float(np.mean(avg_lens))
    inter_dist1 = float(np.mean(inter_dist1s))
    inter_dist2 = float(np.mean(inter_dist2s))
    report = "Avg recall BLEU: %f, avg precision: BLEU %f,\nF1: %f,\nbow_extrema: %f, bow_avg: %f, " \
             "bow_greedy: %f,\n" \
             "intra_dist1: %f, intra_dist2: %f,\navg_len: %f,\ninter_dist1: %f, inter_dist2: %f " \
             "(only 1 ref, not final results)\n" \
             % (recall_bleu, prec_bleu, f1, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len,
                inter_dist1, inter_dist2)
    # print(report)
    f_eval.write(report + "\n")
    # print("Done testing")
    
    return recall_bleu, prec_bleu, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2


# 评测其他模型如CVAE-D的生成结果
# 等于说evaluate，需要已转成文字的(repeat, max_len, 1)，对应的长度(repeat, 1)，以及转成文字的target(max_len)
def evaluate_other_model(result_list, length, target, f_eval, vocab, rev_vocab):
    recall_bleus, prec_bleus, bows_extrema, bows_avg, bows_greedy, intra_dist1s, intra_dist2s, avg_lens, inter_dist1s, inter_dist2s \
        = [], [], [], [], [], [], [], [], [], []
    local_t = 0
    while True:
        ref_tokens = [vocab[e] for e in target[0] if e not in [rev_vocab['</s>'], rev_vocab['<s>'], 0]]
        f_eval.write("Target >> %s\n" % (" ".join(ref_tokens)))

        # 先将别的模型生成的文字用本模型中的vocab，转成本模型下的index

        sample_words_list, sample_lens = model.sample(title, context, repeat, rev_vocab["<s>"], rev_vocab["</s>"])
        # nparray: [repeat x seq_len]
        pred_tokens = [[vocab[e] for e in sample_words
                        if e not in [rev_vocab['</s>'], rev_vocab['<s>'], 0]]
                       for sample_words in sample_words_list]  # 诗句分词成list
        for r_id, pred_sents in enumerate(pred_tokens):
            f_eval.write("Sample %d >> %s\n" % (r_id, ' '.join(pred_sents)))

        # 输入的都是数字list
        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)

        # 计算bow和distinct时候需要用tensor
        # 输入的是数字list转成的tensor，包括sample和target的长度
        bow_extrema, bow_avg, bow_greedy = \
            metrics.sim_bow(to_tensor(sample_words_list).detach(), to_tensor(sample_lens).detach(),
                            to_tensor(target[:, 1:-1]).detach(), to_tensor(target_lens - 2).detach())
        bows_extrema.append(bow_extrema)
        bows_avg.append(bow_avg)
        bows_greedy.append(bow_greedy)
        # import pdb
        # pdb.set_trace()
        # 输入的还是数字list
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = metrics.div_distinct(sample_words_list, sample_lens)
        intra_dist1s.append(intra_dist1)
        intra_dist2s.append(intra_dist2)
        avg_lens.append(np.mean(sample_lens))
        inter_dist1s.append(inter_dist1)
        inter_dist2s.append(inter_dist2)

        f_eval.write("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2 * (prec_bleu * recall_bleu) / (prec_bleu + recall_bleu + 10e-12)
    bow_extrema = float(np.mean(bows_extrema))
    bow_avg = float(np.mean(bows_avg))
    bow_greedy = float(np.mean(bows_greedy))
    intra_dist1 = float(np.mean(intra_dist1s))
    intra_dist2 = float(np.mean(intra_dist2s))
    avg_len = float(np.mean(avg_lens))
    inter_dist1 = float(np.mean(inter_dist1s))
    inter_dist2 = float(np.mean(inter_dist2s))
    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f, bow_extrema %f, bow_avg %f, bow_greedy %f,\
            intra_dist1 %f, intra_dist2 %f, avg_len %f, inter_dist1 %f, inter_dist2 %f (only 1 ref, not final results)" \
             % (recall_bleu, prec_bleu, f1, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len,
                inter_dist1, inter_dist2)
    print(report)
    f_eval.write(report + "\n")
    print("Done testing")



def main(args):
    # Load the models
    conf = getattr(configs, 'config_'+args.model)()
    model=torch.load(f='./output/{}/{}/{}/models/model_epo{}.pckl'.format(args.model, args.expname, args.dataset, args.reload_from))
    model.eval()
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")
    
    
    # Load data
    # api = LoadPoem(args.train_data_dir, args.test_data_dir, max_vocab_cnt=args.max_vocab_size,
    #                word2vec=args.word2vec_path, word2vec_dim=config.emb_size)
    # metrics = Metrics(api.vocab)
    
    f_eval = open("./output/{}/{}/{}/results.txt".format(args.model, args.expname, args.dataset), "w")
    repeat = args.n_samples
    
    evaluate_other_model(model, metrics, test_loader, vocab, ivocab, f_eval, repeat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DialogGAN for Eval')
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='SWDA', help='name of dataset, SWDA or DailyDial')
    parser.add_argument('--model', type=str, default='DialogWAE', help='model name')
    parser.add_argument('--expname', type=str, default='basic', help='experiment name, disinguishing different parameter settings')
    parser.add_argument('--reload_from', type=int, default=40, help='directory to load models from, SWDA 8, 40, DailyDial 6, 40')
    
    parser.add_argument('--n_samples', type=int, default=10, help='Number of responses to sampling')
    parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    print(vars(args))
    main(args)
