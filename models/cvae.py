import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from helper import to_tensor, gaussian_kld
from modules import Encoder, Variation, Decoder


class CVAE(nn.Module):
    def __init__(self, config, api, PAD_token=0):
        super(CVAE, self).__init__()
        self.vocab = api.vocab
        self.vocab_size = len(self.vocab)
        self.rev_vocab = api.rev_vocab
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.maxlen = config.maxlen
        self.clip = config.clip
        self.temp = config.temp
        self.full_kl_step = config.full_kl_step
        self.z_size = config.z_size
        self.init_w = config.init_weight
        self.softmax = nn.Softmax(dim=1)

        self.embedder = nn.Embedding(self.vocab_size, config.emb_size, padding_idx=PAD_token)
        # 对title, 每一句诗做编码
        self.seq_encoder = Encoder(embedder=self.embedder, input_size=config.emb_size, hidden_size=config.n_hidden,
                                   bidirectional=True, n_layers=config.n_layers, noise_radius=config.noise_radius)

        # 先验网络的输入是 标题encode结果 + 上一句诗过encoder的结果 + 上一句情感过encoder的结果
        self.prior_net = Variation(config.n_hidden*4, config.z_size, dropout_rate=config.dropout, init_weight=self.init_w)
        # 后验网络，再加上x的2*hidden
        # self.post_net = Variation(config.n_hidden * 6, config.z_size*2)
        self.post_net = Variation(config.n_hidden*6, config.z_size, dropout_rate=config.dropout, init_weight=self.init_w)
        # 词包loss的MLP
        self.bow_project = nn.Sequential(
            nn.Linear(config.n_hidden*4 + config.z_size, 400),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(400, self.vocab_size)
        )
        self.init_decoder_hidden = nn.Sequential(
            nn.Linear(config.n_hidden*4 + config.z_size, config.n_hidden),
            nn.BatchNorm1d(config.n_hidden, eps=1e-05, momentum=0.1),
            nn.LeakyReLU()
        )
        # self.post_generator = nn.Sequential(
        #     nn.Linear(config.z_size, config.z_size),
        #     nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
        #     nn.LeakyReLU(),
        #     nn.Linear(config.z_size, config.z_size),
        #     nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
        #     nn.LeakyReLU(),
        #     nn.Linear(config.z_size, config.z_size)
        # )
        # self.post_generator.apply(self.init_weights)

        # self.prior_generator = nn.Sequential(
        #     nn.Linear(config.z_size, config.z_size),
        #     nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(config.z_size, config.z_size),
        #     nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(config.z_size, config.z_size)
        # )
        # self.prior_generator.apply(self.init_weights)

        self.init_decoder_hidden.apply(self.init_weights)
        self.bow_project.apply(self.init_weights)
        self.post_net.apply(self.init_weights)

        self.decoder = Decoder(embedder=self.embedder, input_size=config.emb_size,
                               hidden_size=config.n_hidden,
                               vocab_size=self.vocab_size, n_layers=1)

        # self.optimizer_lead = optim.Adam(list(self.seq_encoder.parameters())\
        #                                + list(self.prior_net.parameters()), lr=config.lr_lead)
        self.optimizer_AE = optim.Adam(list(self.seq_encoder.parameters())\
                                       + list(self.prior_net.parameters())\
                                       # + list(self.prior_generator.parameters())
                                       + list(self.post_net.parameters())\
                                       # + list(self.post_generator.parameters())
                                       + list(self.bow_project.parameters())\
                                       + list(self.init_decoder_hidden.parameters())\
                                       + list(self.decoder.parameters()), lr=config.lr_ae)

        # self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size=10, gamma=0.6)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.criterion_sent_lead = nn.CrossEntropyLoss()

    def set_full_kl_step(self, kl_full_step):
        self.full_kl_step = kl_full_step

    def force_change_lr(self, new_init_lr_ae):
        self.optimizer_AE = optim.Adam(list(self.seq_encoder.parameters()) \
                                       + list(self.prior_net.parameters()) \
                                       # + list(self.prior_generator.parameters())
                                       + list(self.post_net.parameters()) \
                                       # + list(self.post_generator.parameters())
                                       + list(self.bow_project.parameters()) \
                                       + list(self.init_decoder_hidden.parameters()) \
                                       + list(self.decoder.parameters()), lr=new_init_lr_ae)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-self.init_w, self.init_w)
            m.bias.data.fill_(0)

    def sample_code_post(self, x, c):
        # import pdb
        # pdb.set_trace()
        # mulogsigma = self.post_net(torch.cat((x, c), dim=1))
        # mu, logsigma = torch.chunk(mulogsigma, chunks=2, dim=1)
        # batch_size = c.size(0)
        # std = torch.exp(0.5 * logsigma)
        # epsilon = to_tensor(torch.randn([batch_size, self.z_size]))
        # z = epsilon * std + mu
        z, mu, logsigma = self.post_net(torch.cat((x, c), 1))  # 输入：(batch, 3*2*n_hidden)
        # z = self.post_generator(z)
        return z, mu, logsigma

    def sample_code_prior(self, c, sentiment_mask=None, mask_type=None):
        return self.prior_net(c, sentiment_mask=sentiment_mask, mask_type=mask_type)

    # # input: (batch, 3)
    # # target: (batch, 3)
    # def criterion_sent_lead(self, input, target):
    #     softmax_res = self.softmax(input)
    #     negative_log_softmax_res = -torch.log(softmax_res + 1e-10)  # (batch, 3)
    #     cross_entropy_loss = torch.sum(negative_log_softmax_res * target, dim=1)
    #     avg_cross_entropy = torch.mean(cross_entropy_loss)
    #     return avg_cross_entropy

    # sentiment_lead: (batch, 3)
    def train_AE(self, global_t, title, context, target, target_lens, sentiment_mask=None, sentiment_lead=None):
        self.seq_encoder.train()
        self.decoder.train()
        # batch_size = title.size(0)
        # 每一句的情感用第二个分类器来预测，输入当前的m_hidden，输出分类结果

        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)

        # import pdb
        # pdb.set_trace()
        x, _ = self.seq_encoder(target[:, 1:], target_lens-1)
        condition_prior = torch.cat((title_last_hidden, context_last_hidden), dim=1)

        z_prior, prior_mu, prior_logvar, pi, pi_final = self.sample_code_prior(condition_prior, sentiment_mask=sentiment_mask)
        z_post, post_mu, post_logvar = self.sample_code_post(x, condition_prior)
        # import pdb
        # pdb.set_trace()
        if sentiment_lead is not None:
            self.sent_lead_loss = self.criterion_sent_lead(input=pi, target=sentiment_lead)
        else:
            self.sent_lead_loss = 0

        # if sentiment_lead is not None:
        #     self.optimizer_lead.zero_grad()
        #     self.sent_lead_loss.backward()
        #     self.optimizer_lead.step()
        #     return [('lead_loss', self.sent_lead_loss.item())], global_t

        final_info = torch.cat((z_post, condition_prior), dim=1)
        # reconstruct_loss
        # import pdb
        # pdb.set_trace()
        output = self.decoder(init_hidden=self.init_decoder_hidden(final_info), context=None, inputs=target[:, :-1])
        flattened_output = output.view(-1, self.vocab_size)
        # flattened_output = self.softmax(flattened_output) + 1e-10
        # flattened_output = torch.log(flattened_output)
        dec_target = target[:, 1:].contiguous().view(-1)

        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz * seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        self.rc_loss = self.criterion_ce(masked_output / self.temp, masked_target)

        # kl散度
        kld = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
        self.avg_kld = torch.mean(kld)
        self.kl_weights = min(global_t / self.full_kl_step, 1.0)  # 退火
        self.kl_loss = self.kl_weights * self.avg_kld

        # avg_bow_loss
        self.bow_logits = self.bow_project(final_info)
        # 说白了就是把target所有词的预测loss求个和
        labels = target[:, 1:]
        label_mask = torch.sign(labels).detach().float()
        # 取符号变成正数，从而通过最小化来optimize
        # soft_result = self.softmax(self.bow_logits) + 1e-10
        # bow_loss = -torch.log(soft_result).gather(1, labels) * label_mask
        bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
        sum_bow_loss = torch.sum(bow_loss, 1)
        self.avg_bow_loss = torch.mean(sum_bow_loss)
        self.aug_elbo_loss = self.avg_bow_loss + self.kl_loss + self.rc_loss
        self.total_loss = self.aug_elbo_loss + self.sent_lead_loss

        # 变相增加标注集的学习率
        if sentiment_mask is not None:
            self.total_loss = self.total_loss * 13.33

        self.optimizer_AE.zero_grad()
        self.total_loss.backward()
        self.optimizer_AE.step()

        avg_total_loss = self.total_loss.item()
        avg_lead_loss = 0 if sentiment_lead is None else self.sent_lead_loss.item()
        avg_aug_elbo_loss = self.aug_elbo_loss.item()
        avg_kl_loss = self.kl_loss.item()
        avg_rc_loss = self.rc_loss.data.item()
        avg_bow_loss = self.avg_bow_loss.item()
        global_t += 1

        return [('avg_total_loss', avg_total_loss),
                ('avg_lead_loss', avg_lead_loss),
                ('avg_aug_elbo_loss', avg_aug_elbo_loss),
                ('avg_kl_loss', avg_kl_loss),
                ('avg_rc_loss', avg_rc_loss),
                ('avg_bow_loss', avg_bow_loss),
                ('kl_weight', self.kl_weights)], global_t

    def valid_AE(self, global_t, title, context, target, target_lens, sentiment_mask=None, sentiment_lead=None):
        self.seq_encoder.eval()
        self.decoder.eval()

        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)

        # import pdb
        # pdb.set_trace()
        x, _ = self.seq_encoder(target[:, 1:], target_lens - 1)
        condition_prior = torch.cat((title_last_hidden, context_last_hidden), dim=1)

        z_prior, prior_mu, prior_logvar, pi, pi_final = self.sample_code_prior(condition_prior, sentiment_mask=sentiment_mask)
        z_post, post_mu, post_logvar = self.sample_code_post(x, condition_prior)
        if sentiment_lead is not None:
            self.sent_lead_loss = self.criterion_sent_lead(input=pi, target=sentiment_lead)
        else:
            self.sent_lead_loss = 0

        # if sentiment_lead is not None:
        #     return [('valid_lead_loss', self.sent_lead_loss.item())], global_t

        final_info = torch.cat((z_post, condition_prior), dim=1)

        output = self.decoder(init_hidden=self.init_decoder_hidden(final_info), context=None, inputs=target[:, :-1])
        flattened_output = output.view(-1, self.vocab_size)
        # flattened_output = self.softmax(flattened_output) + 1e-10
        # flattened_output = torch.log(flattened_output)
        dec_target = target[:, 1:].contiguous().view(-1)

        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz * seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        self.rc_loss = self.criterion_ce(masked_output / self.temp, masked_target)

        # kl散度
        kld = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
        self.avg_kld = torch.mean(kld)
        self.kl_weights = min(global_t / self.full_kl_step, 1.0)  # 退火
        self.kl_loss = self.kl_weights * self.avg_kld
        # avg_bow_loss
        self.bow_logits = self.bow_project(final_info)
        # 说白了就是把target所有词的预测loss求个和
        labels = target[:, 1:]
        label_mask = torch.sign(labels).detach().float()

        bow_loss = -F.log_softmax(self.bow_logits, dim=1).gather(1, labels) * label_mask
        sum_bow_loss = torch.sum(bow_loss, 1)
        self.avg_bow_loss = torch.mean(sum_bow_loss)
        self.aug_elbo_loss = self.avg_bow_loss + self.kl_loss + self.rc_loss

        avg_aug_elbo_loss = self.aug_elbo_loss.item()
        avg_kl_loss = self.kl_loss.item()
        avg_rc_loss = self.rc_loss.data.item()
        avg_bow_loss = self.avg_bow_loss.item()
        avg_lead_loss = 0 if sentiment_lead is None else self.sent_lead_loss.item()

        return [('valid_lead_loss', avg_lead_loss),
                ('valid_aug_elbo_loss', avg_aug_elbo_loss),
                ('valid_kl_loss', avg_kl_loss),
                ('valid_rc_loss', avg_rc_loss),
                ('valid_bow_loss', avg_bow_loss)], global_t

    # batch_size = 1 只输入了一个标题
    # test的时候，只有先验，没有后验，更没有所谓的kl散度
    def test(self, title_tensor, title_words, mask_type=None):
        self.seq_encoder.eval()
        self.decoder.eval()
        assert title_tensor.size(0) == 1
        tem = [[2, 3] + [0] * (self.maxlen - 2)]
        pred_poems = []
        # 过滤掉标题中的<s> </s> 0,只为了打印
        title_tokens = [self.vocab[e] for e in title_words[0].tolist() if e not in [0, self.eos_id, self.go_id]]
        pred_poems.append(title_tokens)

        for i in range(4):
            tem = to_tensor(np.array(tem))
            context = tem
            if i == 0:
                context_last_hidden, _ = self.seq_encoder(title_tensor)
            else:
                context_last_hidden, _ = self.seq_encoder(context)
            title_last_hidden, _ = self.seq_encoder(title_tensor)

            condition_prior = torch.cat((title_last_hidden, context_last_hidden), dim=1)
            z_prior, prior_mu, prior_logvar, _, _ = self.sample_code_prior(condition_prior, mask_type=mask_type)
            final_info = torch.cat((z_prior, condition_prior), 1)

            decode_words = self.decoder.testing(init_hidden=self.init_decoder_hidden(final_info), maxlen=self.maxlen, go_id=self.go_id, mode="greedy")
            decode_words = decode_words[0].tolist()

            if len(decode_words) >= self.maxlen:
                tem = [decode_words[0: self.maxlen]]
            else:
                tem = [[0] * (self.maxlen - len(decode_words)) + decode_words]
            pred_tokens = [self.vocab[e] for e in decode_words[:-1] if e != self.eos_id and e != 0 and e != self.go_id]
            pred_poems.append(pred_tokens)

        gen = ""
        for line in pred_poems:
            cur_line = " ".join(line)
            gen = gen + cur_line + '\n'

        return gen

    def sample(self, title, context, repeat, go_id, end_id):
        self.seq_encoder.eval()
        self.decoder.eval()

        assert title.size(0) == 1
        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)
        condition_prior = torch.cat((title_last_hidden, context_last_hidden), 1)
        condition_prior_repeat = condition_prior.expand(repeat, -1)

        z_prior_repeat, _, _, _, _ = self.sample_code_prior(condition_prior_repeat)

        final_info = torch.cat((z_prior_repeat, condition_prior_repeat), dim=1)
        sample_words, sample_lens = self.decoder.sampling(init_hidden=self.init_decoder_hidden(final_info), maxlen=self.maxlen,
                                                          go_id=self.go_id, eos_id=self.eos_id, mode="greedy")

        return sample_words, sample_lens

    # def adjust_lr(self):
    #     self.lr_scheduler_AE.step()