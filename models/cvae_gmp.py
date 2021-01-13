import torch
import numpy as np
from .cvae import CVAE
from modules import MixVariation
from helper import to_tensor



class CVAE_GMP(CVAE):
    def __init__(self, config, api, PAD_token=0):
        super(CVAE_GMP, self).__init__(config, api, PAD_token=PAD_token)
        self.n_components = config.n_prior_components
        self.gumbel_temp = config.gumbel_temp
        self.temp_size = config.temp_size
        self.prior_net = MixVariation(input_size=self.prior_input_dim, temp_size=self.temp_size, z_size=config.z_size,
                                      n_components=self.n_components, dropout_rate=config.dropout,
                                      init_weight=self.init_w)

    def sample_code_prior(self, c, sentiment_mask=None, mask_type=None):
        return self.prior_net(c, sentiment_mask=sentiment_mask, mask_type=mask_type)

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
            # z_prior, prior_mu, prior_logvar, _, _ = self.sample_code_prior(condition_prior, mask_type=mask_type)
            z_prior, prior_mu, prior_logvar = self.sample_code_prior(condition_prior, mask_type=mask_type)
            final_info = torch.cat((z_prior, condition_prior), 1)

            decode_words = self.decoder.testing(init_hidden=self.init_decoder_hidden(final_info),
                                                maxlen=self.maxlen, go_id=self.go_id, mode="greedy")
            decode_words = decode_words[0].tolist()

            if len(decode_words) >= self.maxlen:
                tem = [decode_words[0: self.maxlen]]
            else:
                tem = [[0] * (self.maxlen - len(decode_words)) + decode_words]
            pred_tokens = [self.vocab[e] for e in decode_words[:-1] if
                           e != self.eos_id and e != 0 and e != self.go_id]
            pred_poems.append(pred_tokens)

        gen = "\n"
        for line in pred_poems:
            cur_line = " ".join(line)
            gen = gen + cur_line + '\n'

        return gen
