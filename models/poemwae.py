import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import to_tensor
from modules import Encoder, Variation, Decoder
    
one = torch.tensor(1, dtype=torch.float).cuda()
minus_one = one * -1    


class PoemWAE(nn.Module):
    def __init__(self, config, api, PAD_token=0, pretrain_weight=None):
        super(PoemWAE, self).__init__()
        self.vocab = api.vocab
        self.vocab_size = len(self.vocab)
        self.rev_vocab = api.rev_vocab
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.maxlen = config.maxlen
        self.clip = config.clip
        self.lambda_gp = config.lambda_gp
        self.lr_gan_g = config.lr_gan_g
        self.lr_gan_d = config.lr_gan_d
        self.n_d_loss = config.n_d_loss
        self.temp = config.temp
        self.init_w = config.init_weight

        self.embedder = nn.Embedding(self.vocab_size, config.emb_size, padding_idx=PAD_token)
        if pretrain_weight is not None:
            self.embedder.weight.data.copy_(torch.from_numpy(pretrain_weight))
        # 用同一个seq_encoder来编码标题和前后两句话
        self.seq_encoder = Encoder(self.embedder, config.emb_size, config.n_hidden,
                                     True, config.n_layers, config.noise_radius)
        # 由于Poem这里context是title和last sentence双向GRU编码后的直接cat，4*hidden
        # 注意如果使用Poemwar_gmp则使用子类中的prior_net，即混合高斯分布的一个先验分布
        self.prior_net = Variation(config.n_hidden * 4, config.z_size, dropout_rate=config.dropout,
                                   init_weight=self.init_w)  # p(e|c)

        # 注意这儿原来是给Dialog那个任务用的，3*hidden
        # Poem数据集上，将title和上一句，另外加上x都分别用双向GRU编码并cat，因此是6*hidden
        self.post_net = Variation(config.n_hidden * 6, config.z_size, dropout_rate=config.dropout,
                                  init_weight=self.init_w)

        self.post_generator = nn.Sequential(
            nn.Linear(config.z_size, config.z_size),
            nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config.z_size, config.z_size),
            nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config.z_size, config.z_size)
        )
        self.post_generator.apply(self.init_weights)

        self.prior_generator = nn.Sequential(
            nn.Linear(config.z_size, config.z_size),
            nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config.z_size, config.z_size),
            nn.BatchNorm1d(config.z_size, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config.z_size, config.z_size)
        )
        self.prior_generator.apply(self.init_weights)

        self.init_decoder_hidden = nn.Sequential(
            nn.Linear(config.n_hidden * 4 + config.z_size, config.n_hidden * 4),
            nn.BatchNorm1d(config.n_hidden * 4, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )

        # 由于Poem这里context是title和last sentence双向GRU编码后的直接cat，因此hidden_size变为z_size + 4*hidden
        # 修改：decoder的hidden_size还设为n_hidden, init_hidden使用一个MLP将cat变换为n_hidden
        self.decoder = Decoder(self.embedder, config.emb_size, config.n_hidden * 4,
                               self.vocab_size, n_layers=1)

        self.discriminator = nn.Sequential(
            # 因为Poem的cat两个双向编码，这里改为4*n_hidden + z_size
            nn.Linear(config.n_hidden * 4 + config.z_size, config.n_hidden * 2),
            nn.BatchNorm1d(config.n_hidden * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config.n_hidden * 2, config.n_hidden * 2),
            nn.BatchNorm1d(config.n_hidden * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config.n_hidden * 2, 1),
        )
        self.discriminator.apply(self.init_weights)

        # optimizer 定义，分别对应三个模块的训练，注意！三个模块的optimizer不相同
        # self.optimizer_AE = optim.SGD(list(self.seq_encoder.parameters())
        self.optimizer_AE = optim.SGD(list(self.seq_encoder.parameters())
                                      + list(self.post_net.parameters())
                                      + list(self.post_generator.parameters())
                                      + list(self.init_decoder_hidden.parameters())
                                      + list(self.decoder.parameters()), lr=config.lr_ae)
        self.optimizer_G = optim.RMSprop(list(self.post_net.parameters())
                                         + list(self.post_generator.parameters())
                                         + list(self.prior_net.parameters())
                                         + list(self.prior_generator.parameters()),
                                         lr=self.lr_gan_g)
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=self.lr_gan_d)

        self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size=10, gamma=0.8)

        self.criterion_ce = nn.CrossEntropyLoss()
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-self.init_w, self.init_w)
            # nn.init.kaiming_normal_(m.weight.data)
            # nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.fill_(0)

    # x: (batch, 2*n_hidden)
    # c: (batch, 2*2*n_hidden)
    def sample_code_post(self, x, c):
        z, _, _ = self.post_net(torch.cat((x, c), 1))  # 输入：(batch, 3*2*n_hidden)
        z = self.post_generator(z)
        return z

    def sample_code_prior_sentiment(self, c, align):
        choice_statistic = self.prior_net(c, align)  # e: (batch, z_size)
        return choice_statistic

    def sample_code_prior(self, c):
        z, _, _ = self.prior_net(c)  # e: (batch, z_size)
        z = self.prior_generator(z)  # z: (batch, z_size)
        return z

    # 输入 title, context, target, target_lens.
    # c由title和context encode之后的hidden相concat而成
    def train_AE(self, title, context, target, target_lens):
        self.seq_encoder.train()
        self.decoder.train()
        # import pdb
        # pdb.set_trace()
        # (batch, 2 * hidden_size)
        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)

        # (batch, 2 * hidden_size)
        x, _ = self.seq_encoder(target[:, 1:], target_lens-1)
        # context_embedding
        c = torch.cat((title_last_hidden, context_last_hidden), 1)  # (batch, 2 * hidden_size * 2)
        z = self.sample_code_post(x, c)  # (batch, z_size)

        # 标准的autoencoder的decode，decoder初态为x, c的cat，将target错位输入
        # output: (batch, len, vocab_size) len是9，即7+标点+</s>

        output = self.decoder(self.init_decoder_hidden(torch.cat((z, c), 1)), None, target[:, :-1], target_lens-1)
        flattened_output = output.view(-1, self.vocab_size)
        
        dec_target = target[:, 1:].contiguous().view(-1)
        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)  # [(batch_sz * seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        
        self.optimizer_AE.zero_grad()
        loss = self.criterion_ce(masked_output / self.temp, masked_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(self.seq_encoder.parameters())+list(self.decoder.parameters()), self.clip)
        self.optimizer_AE.step()

        return [('train_loss_AE', loss.item())]        

    # G是来缩短W距离的，可以类比VAE里面的缩小KL散度项
    def train_G(self, title, context, target, target_lens, sentiment_mask=None, mask_type=None):
        self.seq_encoder.eval()
        self.optimizer_G.zero_grad()
        
        for p in self.discriminator.parameters():
            p.requires_grad = False
        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)
        c = torch.cat((title_last_hidden, context_last_hidden), 1)  # (batch, 2 * hidden_size * 2)

        # -----------------posterior samples ---------------------------
        x, _ = self.seq_encoder(target[:, 1:], target_lens - 1)
        z_post = self.sample_code_post(x.detach(), c.detach())  # 去掉梯度，防止梯度向encoder的传播 (batch, z_size)


        errG_post = torch.mean(self.discriminator(torch.cat((z_post, c.detach()), 1))) * self.n_d_loss  # (batch, z_size + 4 * hidden)
        errG_post.backward(minus_one)

        # ----------------- prior samples ---------------------------
        prior_z = self.sample_code_prior(c.detach())
        errG_prior = torch.mean(self.discriminator(torch.cat((prior_z, c.detach()), 1))) * self.n_d_loss
        # import pdb
        # pdb.set_trace()
        errG_prior.backward(one)
        self.optimizer_G.step()

        for p in self.discriminator.parameters():
            p.requires_grad = True  
        
        costG = errG_prior - errG_post

        return [('train_loss_G', costG.item())]

    # D是用来拟合W距离，loss下降说明拟合度变好，增大gradient_penalty一定程度上可以提高拟合度
    # n_iters_n越大，D训练的次数越多，对应的拟合度也越好
    def train_D(self, title, context, target, target_lens):
        self.seq_encoder.eval()
        self.discriminator.train()
        self.optimizer_D.zero_grad()

        batch_size = context.size(0)

        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)
        c = torch.cat((title_last_hidden, context_last_hidden), 1)  # (batch, 2, hidden_size * 2)
        x, _ = self.seq_encoder(target[:, 1:], target_lens - 1)
        post_z = self.sample_code_post(x, c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z.detach(), c.detach()), 1))) * self.n_d_loss
        errD_post.backward(one)
 
        prior_z = self.sample_code_prior(c)
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z.detach(), c.detach()), 1))) * self.n_d_loss
        errD_prior.backward(minus_one)
        # import pdb
        # pdb.set_trace()
    
        alpha = to_tensor(torch.rand(batch_size, 1))
        alpha = alpha.expand(prior_z.size())
        interpolates = alpha * prior_z.data + ((1 - alpha) * post_z.data)
        interpolates = Variable(interpolates, requires_grad=True)

        d_input = torch.cat((interpolates, c.detach()), 1)
        disc_interpolates = torch.mean(self.discriminator(d_input))
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                               grad_outputs=to_tensor(torch.ones(disc_interpolates.size())),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.contiguous().view(gradients.size(0), -1)
                             .norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        gradient_penalty.backward()
    
        self.optimizer_D.step()
        costD = -(errD_prior - errD_post) + gradient_penalty
        return [('train_loss_D', costD.item())]   
    
    def valid(self, title, context, target, target_lens, sentiment_mask=None):
        self.seq_encoder.eval()
        self.discriminator.eval()
        self.decoder.eval()

        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)
        c = torch.cat((title_last_hidden, context_last_hidden), 1)  # (batch, 2 * hidden_size * 2)
        x, _ = self.seq_encoder(target[:, 1:], target_lens-1)

        post_z = self.sample_code_post(x, c)
        prior_z = self.sample_code_prior(c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z, c), 1)))
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z, c), 1)))
        costD = -(errD_prior - errD_post)
        costG = -costD 
        
        dec_target = target[:, 1:].contiguous().view(-1)  # (batch_size * len)
        mask = dec_target.gt(0)  # 即判断target的token中是否有0（pad项）
        masked_target = dec_target.masked_select(mask)  # 选出非pad项
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)

        output = self.decoder(self.init_decoder_hidden(torch.cat((post_z, c), 1)), None, target[:, :-1], (target_lens-1))
        flattened_output = output.view(-1, self.vocab_size) 
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        lossAE = self.criterion_ce(masked_output/self.temp, masked_target)
        return [('valid_loss_AE', lossAE.item()), ('valid_loss_G', costG.item()), ('valid_loss_D', costD.item())]

    # 正如论文中说的，测试生成的时候，从先验网络中拿到噪声，用G生成prior_z（即代码中的sample_code_prior(c)）
    # 然后decoder将prior_z和c的cat当做输入，decode出这句诗（这和论文里面不太一样，论文里面只把prior_z当做输入）
    # batch_size是1，一次测一句

    # title 即标题
    # context 上一句
    def test(self, title_tensor, title_words, headers):
        self.seq_encoder.eval()
        self.discriminator.eval()
        self.decoder.eval()
        # tem初始化为[2,3,0,0,0,0,0,0,0]

        tem = [[2, 3] + [0] * (self.maxlen - 2)]
        pred_poems = []

        title_tokens = [self.vocab[e] for e in title_words[0].tolist() if e not in [0, self.eos_id, self.go_id]]
        pred_poems.append(title_tokens)
        for sent_id in range(4):
            tem = to_tensor(np.array(tem))
            context = tem

            # vec_context = np.zeros((batch_size, self.maxlen), dtype=np.int64)
            # for b_id in range(batch_size):
            #     vec_context[b_id, :] = np.array(context[b_id])
            # context = to_tensor(vec_context)

            title_last_hidden, _ = self.seq_encoder(title_tensor)  # （batch=1, 2*hidden）
            if sent_id == 0:
                context_last_hidden, _ = self.seq_encoder(title_tensor)  # (batch=1, 2*hidden)
            else:
                context_last_hidden, _ = self.seq_encoder(context)  # (batch=1, 2*hidden)
            c = torch.cat((title_last_hidden, context_last_hidden), 1)  # (batch, 4*hidden_size)
            # 由于一次只有一首诗，batch_size = 1，因此不必repeat
            prior_z = self.sample_code_prior(c)

            # decode_words 是完整的一句诗
            decode_words = self.decoder.testing(init_hidden=self.init_decoder_hidden(torch.cat((prior_z, c), 1)),
                                             maxlen=self.maxlen, go_id=self.go_id,
                                             mode="greedy", header=headers[sent_id])

            decode_words = decode_words[0].tolist()
            # import pdb
            # pdb.set_trace()
            if len(decode_words) > self.maxlen:
                tem = [decode_words[0: self.maxlen]]
            else:
                tem = [[0] * (self.maxlen - len(decode_words)) + decode_words]

            pred_tokens = [self.vocab[e] for e in decode_words[:-1] if e != self.eos_id and e != 0]
            pred_poems.append(pred_tokens)

        gen = ''
        for line in pred_poems:
            true_str = " ".join(line)
            gen = gen + true_str + '\n'

        return gen

    def sample(self, title, context, repeat, go_id, end_id):
        self.seq_encoder.eval()
        self.decoder.eval()

        title_last_hidden, _ = self.seq_encoder(title)
        context_last_hidden, _ = self.seq_encoder(context)
        c = torch.cat((title_last_hidden, context_last_hidden), 1)  # (batch, 2 * hidden_size * 2)

        c_repeated = c.expand(repeat, -1)  # 注意，我们输入的batch_size是1，这里复制repeat遍，为了后面的BLEU计算

        prior_z = self.sample_code_prior(c_repeated)  # c_repeated: (batch_size=repeat, 4*hidden_size)

        # (batch, max_len, 1)  (batch_size, 1)
        sample_words, sample_lens = self.decoder.sampling(self.init_decoder_hidden(torch.cat((prior_z, c_repeated), 1)),
                                                          self.maxlen, go_id, end_id, "greedy")
        return sample_words, sample_lens

    # def adjust_lr(self):
    #     self.lr_scheduler_AE.step()
