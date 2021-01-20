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
        self.prior_net = MixVariation(sent_class_size=self.sent_class_size, sent_emb_size=self.sent_emb_size, input_size=self.prior_input_dim, temp_size=self.temp_size, z_size=config.z_size,
                                      n_components=self.n_components, dropout_rate=config.dropout,
                                      init_weight=self.init_w)