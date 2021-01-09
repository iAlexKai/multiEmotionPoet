from .cvae import CVAE
from modules import MixVariation

class CVAE_GMP(CVAE):
    def __init__(self, config, api, PAD_token=0):
        super(CVAE_GMP, self).__init__(config, api, PAD_token=PAD_token)
        self.n_components = config.n_prior_components
        self.gumbel_temp = config.gumbel_temp

        self.prior_net = MixVariation(input_size=config.n_hidden*4, z_size=config.z_size,
                                      n_components=self.n_components, dropout_rate=config.dropout,
                                      init_weight=config.init_weight)