class MaicParameters():
    def __init__(self, n_agents, latent_dim=8, nn_hidden_size=64, rnn_hidden_dim=64, 
                 attention_dim=32, var_floor=0.002, mi_loss_weight=0.001, 
                 entropy_loss_weight=0.01, com=True):
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.nn_hidden_size = nn_hidden_size
        self.rnn_hidden_dim = rnn_hidden_dim
        self.attention_dim = attention_dim
        self.var_floor = var_floor
        self.mi_loss_weight = mi_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.com = com
