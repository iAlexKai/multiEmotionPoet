import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from modules import Encoder
from helper import to_tensor


class AttnDecoder(nn.Module):
    def __init__(self, config, embedder, vocab_size):
        super(AttnDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_hidden = config.n_hidden
        self.embedder = embedder
        self.attn = nn.Linear(self.n_hidden + config.emb_size, config.maxlen)
        self.drop_out = nn.Dropout(config.dropout)

        self.attn_combine = nn.Linear(self.n_hidden + config.emb_size, self.n_hidden)
        self.rnn = nn.GRU(input_size=self.n_hidden, hidden_size=self.n_hidden, batch_first=True)
        self.out = nn.Linear(self.n_hidden, self.vocab_size)

        self.soft = nn.LogSoftmax(dim=1)

    # decoder_input: (batch) 逐位输入计算attention
    # init_hidden: (1, batch, n_hidden)
    # encoder_output: (batch, len, n_hidden)
    def forward(self, decoder_input, init_hidden, encoder_output):
        embedded = self.embedder(decoder_input).unsqueeze(1)  # # (batch, 1, emb_dim)
        # embedded = self.drop_out(embedded)
        mutual_info = torch.cat((init_hidden.squeeze().unsqueeze(1), embedded), dim=2)  # (batch, 1, emb_dim + n_hidden)
        attn_weight = self.attn(mutual_info)  # (batch, 1, 10)
        attn_weight = F.softmax(attn_weight, dim=2)  # (batch, 1, 10)
        attn_applied = torch.bmm(attn_weight, encoder_output)  # (batch, 1, n_hidden)

        rnn_input = torch.cat((attn_applied, embedded), dim=2)  # (batch, 1, n_hidden+emb_dim)
        rnn_input = self.attn_combine(rnn_input)  # (batch, 1, n_hidden)
        rnn_input = F.relu(rnn_input)  # (batch, 1, n_hidden)
        # init_hidden: (1, batch, n_hidden)
        # hids: (batch, 1, n_hidden)  h_n: (1, batch, n_hidden)
        hids, h_n = self.rnn(rnn_input, init_hidden.contiguous())
        decoded = self.out(hids.contiguous().squeeze(1))  # (batch, vocab_size)
        decoded = self.soft(decoded)  # 求取softmax后求对数，全部转为负数

        return decoded, h_n

    # init_hidden: (1, batch, n_hidden)  (1,1,400)
    # encoder_output: (batch, len, n_hidden)  (1, 10, 400)
    def testing(self, init_hidden, encoder_output, maxlen, go_id, mode="greedy"):
        batch_size = init_hidden.size(0)
        assert batch_size == 1
        decoder_input = to_tensor(torch.LongTensor([go_id]))  # (batch)
        decoder_hidden = init_hidden
        pred_outs = np.zeros((batch_size, maxlen), dtype=np.int64)

        for di in range(maxlen - 1):  # 从第一个字decode到</s> 共maxlen-1位

            embedded = self.embedder(decoder_input).unsqueeze(1)  # (batch, 1, emb_dim)
            # embedded = self.drop_out(embedded)
            mutual_info = torch.cat((decoder_hidden, embedded),
                                    dim=2)  # (batch, 1, emb_dim + n_hidden)
            attn_weight = self.attn(mutual_info)  # (batch, 1, 10)
            attn_weight = F.softmax(attn_weight, dim=2)  # (batch, 1, 10)
            attn_applied = torch.bmm(attn_weight, encoder_output)  # (batch, 1, n_hidden)

            rnn_input = torch.cat((attn_applied, embedded), dim=2)  # (batch, 1, n_hidden+emb_dim)
            rnn_input = self.attn_combine(rnn_input)  # (batch, 1, n_hidden)
            rnn_input = F.relu(rnn_input)  # (batch, 1, n_hidden)
            decoder_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.contiguous())
            decoder_output = self.soft(self.out(decoder_output.contiguous().squeeze(1)))
            if mode == "greedy":
                topi = decoder_output.max(1, keepdim=True)[1]  # [0]是概率，[1]是idx
            else:
                topi = torch.multinomial(F.softmax(decoder_output[:, -1], dim=1), 1)

            ni = topi.squeeze().cpu().numpy()
            pred_outs[:, di] = ni

            decoder_input = topi[0]

        return pred_outs

    # init_hidden: (1, batch, n_hidden)  (1,10,400)
    # encoder_output: (batch, len, n_hidden)  (10, 10, 400)
    def sampling(self, init_hidden, encoder_output, maxlen, go_id, eos_id, mode='greedy'):

        batch_size = init_hidden.size(1)
        decoder_input = to_tensor(torch.LongTensor(batch_size * [go_id]))  # (batch, 1)
        decoder_hidden = init_hidden  # (1, batch, hidden)
        pred_outs = np.zeros((batch_size, maxlen), dtype=np.int64)
        sample_lens = np.zeros(batch_size, dtype=np.int64)

        for di in range(maxlen - 1):  # 从第一个字decode到</s> 共maxlen-1位

            embedded = self.embedder(decoder_input).unsqueeze(1)  # (batch, 1, emb_dim)
            # embedded = self.drop_out(embedded)
            mutual_info = torch.cat((decoder_hidden.squeeze(0).unsqueeze(1), embedded),
                                    dim=2)  # (batch, 1, emb_dim + n_hidden)
            attn_weight = self.attn(mutual_info)  # (batch, 1, 10)
            attn_weight = F.softmax(attn_weight, dim=2)  # (batch, 1, 10)
            attn_applied = torch.bmm(attn_weight, encoder_output)  # (batch, 1, n_hidden)

            rnn_input = torch.cat((attn_applied, embedded), dim=2)  # (batch, 1, n_hidden+emb_dim)
            rnn_input = self.attn_combine(rnn_input)  # (batch, 1, n_hidden)
            rnn_input = F.relu(rnn_input)  # (batch, 1, n_hidden)
            decoder_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.contiguous())
            decoder_output = self.soft(self.out(decoder_output.contiguous().squeeze(1)))
            if mode == "greedy":
                topi = decoder_output.max(1, keepdim=True)[1]  # [0]是概率，[1]是idx
            else:
                topi = torch.multinomial(F.softmax(decoder_output[:, -1], dim=1), 1)

            ni = topi.squeeze().cpu().numpy()
            pred_outs[:, di] = ni

            decoder_input = topi.squeeze(1)

        # import pdb
        # pdb.set_trace()
        for i in range(batch_size):
            for word in pred_outs[i]:
                if word == eos_id:
                    break
                sample_lens[i] = sample_lens[i] + 1
        return pred_outs, sample_lens


class Seq2Seq(nn.Module):
    def __init__(self, config, api, pad_token=0):
        super(Seq2Seq, self).__init__()
        self.vocab = api.vocab
        self.vocab_size = len(self.vocab)
        self.rev_vocab = api.rev_vocab
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.maxlen = config.maxlen

        self.embedder = nn.Embedding(self.vocab_size, config.emb_size, padding_idx=pad_token)
        self.encoder = Encoder(self.embedder, config.emb_size, config.n_hidden,
                               True, config.n_layers, config.noise_radius)
        self.decoder = AttnDecoder(config=config, embedder=self.embedder, vocab_size=self.vocab_size)

        self.criterion = nn.NLLLoss(reduction='none')
        self.optimizer = optim.Adam(list(self.encoder.parameters())
                                   + list(self.decoder.parameters()),
                                   lr=config.lr_s2s)
        self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.6)

    # 将title和诗句都限制成10个字，不够了pad，超了截取
    # 每次输入进来一个batch的context和target
    def train_model(self, context, target, target_lens):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        # (batch, 2 * n_hidden), (batch, len, 2*n_hidden)
        encoder_last_hidden, encoder_output = self.encoder(context)
        batch_size = encoder_last_hidden.size(0)
        hidden_size = encoder_last_hidden.size(1) // 2
        # (1, batch, n_hidden)
        last_hidden = encoder_last_hidden.view(batch_size, 2, -1)[:, -1, :].squeeze().unsqueeze(0)
        # (batch, len, n_hidden)
        encoder_output = encoder_output.view(batch_size, -1, 2, hidden_size)[:, :, -1]

        decoder_input = target[:, :-1]  # (batch, 9)
        decoder_target = target[:, 1:]  # (batch, 9
        step_losses = []
        for i in range(self.maxlen - 1):
            decoded_result, last_hidden = \
                self.decoder(decoder_input=decoder_input[:, i], init_hidden=last_hidden, encoder_output=encoder_output)
            step_loss = self.criterion(decoded_result, decoder_target[:, i])
            step_losses.append(step_loss)

        stack_loss = torch.stack(step_losses, 1)  # (batch, maxlen)
        sum_loss = torch.sum(stack_loss, 1)  # 对每一行求和

        avg_loss_batch = sum_loss / (target_lens.float() - 1)  # 对每一行先求平均, decode时候每一行是9个字符
        loss = torch.mean(avg_loss_batch)  # 再对所有行一起求和
        loss.backward()
        self.optimizer.step()
        return [('train_loss', loss.item())]

    def valid(self, context, target, target_lens):
        self.encoder.eval()
        self.decoder.eval()

        encoder_last_hidden, encoder_output = self.encoder(context)
        batch_size = encoder_last_hidden.size(0)
        hidden_size = encoder_last_hidden.size(1) // 2
        # (1, batch, n_hidden)
        last_hidden = encoder_last_hidden.view(batch_size, 2, -1)[:, -1, :].squeeze().unsqueeze(0)
        # (batch, len, n_hidden)
        encoder_output = encoder_output.view(batch_size, -1, 2, hidden_size)[:, :, -1]

        decoder_input = target[:, :-1]  # (batch, 9)
        decoder_target = target[:, 1:]  # (batch, 9
        step_losses = []
        for i in range(self.maxlen - 1):
            decoded_result, last_hidden = \
                self.decoder(decoder_input=decoder_input[:, i], init_hidden=last_hidden, encoder_output=encoder_output)
            step_loss = self.criterion(decoded_result, decoder_target[:, i])
            step_losses.append(step_loss)

        stack_loss = torch.stack(step_losses, 1)  # (batch, maxlen)
        sum_loss = torch.sum(stack_loss, 1)  # 对每一行求和

        avg_loss_batch = sum_loss / (target_lens.float() - 1)  # 对每一行先求平均, decode时候每一行是9个字符
        loss = torch.mean(avg_loss_batch)  # 再对所有行一起求和

        return [('valid_loss', loss.item())]

    def test(self, title, title_list, batch_size):
        self.encoder.eval()
        self.decoder.eval()

        assert title.size(0) == 1
        tem = title[0][0: self.maxlen].unsqueeze(0)

        pred_poems = []
        title_tokens = [self.vocab[e] for e in title_list[0].tolist() if e not in [0, self.eos_id, self.go_id]]
        pred_poems.append(title_tokens)

        for sent_id in range(4):
            context = tem
            if type(context) is list:
                vec_context = np.zeros((batch_size, self.maxlen), dtype=np.int64)
                for b_id in range(batch_size):
                    vec_context[b_id, :] = np.array(context[b_id])
                context = to_tensor(vec_context)

            encoder_last_hidden, encoder_output = self.encoder(context)
            batch_size = encoder_last_hidden.size(0)
            hidden_size = encoder_last_hidden.size(1) // 2
            # (1, 1, n_hidden)
            last_hidden = encoder_last_hidden.view(batch_size, 2, -1)[:, -1, :].unsqueeze(0)
            # (batch, len, n_hidden)
            encoder_output = encoder_output.view(batch_size, -1, 2, hidden_size)[:, :, -1]

            # decode_words 是完整的一句诗
            decode_words = self.decoder.testing(init_hidden=last_hidden, encoder_output=encoder_output,
                                             maxlen=self.maxlen, go_id=self.go_id, mode="greedy")

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
        self.encoder.eval()
        self.decoder.eval()
        encoder_last_hidden, encoder_output = self.encoder(context)
        batch_size = encoder_last_hidden.size(0)
        hidden_size = encoder_last_hidden.size(1) // 2
        # (1, batch, n_hidden)
        last_hidden = encoder_last_hidden.view(batch_size, 2, -1)[:, -1].unsqueeze(0)
        # (batch, len, n_hidden)
        encoder_output = encoder_output.view(batch_size, -1, 2, hidden_size)[:, :, -1]

        last_hidden = last_hidden.expand(1, repeat, hidden_size)
        encoder_output = encoder_output.expand(repeat, -1, hidden_size)

        sample_words, sample_lens = self.decoder.sampling(last_hidden, encoder_output, self.maxlen,
                                                          go_id, end_id, "greedy")
        return sample_words, sample_lens

    def adjust_lr(self):
        self.lr_scheduler_AE.step()





















