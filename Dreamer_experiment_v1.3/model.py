import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import torch.distributions as td
import random


'''
Model compornents :
    RSSM
        Recurrent Model : h_t = f_phai(h_t-1, z_t-1, a_t-1)
        Representation Model : z_t ~ q_phai(z_t|h_t, x_t)  posterior
        Transition Predictor : z^_t ~ p_phai(z^_t|h_t)  prior
    Image Predictor : x^_t ~ p_phai(x^_t|h_t, z_t)
    Reward Predictor : r^_t ~ p_phai(r^_t|h_t, z_t)
    Discount Predictor : gamma^_t ~ p_phai(gamma^_t|h_t, z_t)

    Actor Model : a^_t ~ p_psi(a^_t|h_t, z^_t)
    Value Model : v^_t = V(h_t, z^_t)

Model class :
    Encoder : image obserbation to vector
    RSSM
    Obserbation Model
    RewardModel
    Discount Model
    Actor Model(Continuous or Discrete)
    Value Model
'''

class Encoder(nn.Module):
    """
        Encoder to embed image observation to vector

        feat: (batch_size, 1, height, width)
        out:  (batch_size, 1024)
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # CNN 出力計算 : ceil((N-F+1)/S) ceil:整数値に切り上げる
        # model1:dim[128, 128]
        # self.cv1 = nn.Conv2d(1, 16, kernel_size=6, stride=2) # 1x128x128 -> 16x62x62
        # self.cv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # 16x62x62 -> 32x30x30
        # self.cv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # 32x30x30 -> 64x14x14
        # self.cv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2) # 64x14x14 -> 128x6x6
        # self.cv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2) # 128x6x6 -> 256x2x2 = 1024
        
        # model2:dim[64, 64]
        # Dreamer v2 : layer=4, depth=48, 96, 192, 384, act=elu, kernels: [4, 4, 4, 4], stride=2, (norm=layer)
        # Dreamer v1 : layer=4, depth=32, 64, 128, 256, act=relu, kernels: [4, 4, 4, 4], stride=2,
        self.cv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2) # 1x64x64 -> 16x31x31
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # 32x31x31 -> 64x14x14
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2) # 64x14x14 -> 128x6x6
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2) # 128x6x6 -> 256x2x2

    def forward(self, obs):
        # model1
        # hidden = F.elu(self.cv1(obs))
        # hidden = F.elu(self.cv2(hidden))
        # hidden = F.elu(self.cv3(hidden))
        # hidden = F.elu(self.cv4(hidden))
        # embedded_obs = F.elu(self.cv5(hidden)).reshape(hidden.size(0), -1)
        
        # model2
        hidden = F.elu(self.cv1(obs))
        hidden = F.elu(self.cv2(hidden))
        hidden = F.elu(self.cv3(hidden))
        embedded_obs = F.elu(self.cv4(hidden)).reshape(hidden.size(0), -1)
        return embedded_obs


class RecurrentStateSpaceModel(nn.Module):
    """
        This class includes multiple components
        Deterministic state model: h_t = f(h_t-1, s_t-1, a_t-1)
        Stochastic state model (prior): p(s_t | h_t)
        State posterior: q(s_t | h_t, o_t)
    """
    def __init__(self, latent_dim, n_atoms, action_dim, rnn_hidden_dim,
                 hidden_dim=512, min_stddev=0.1, act=F.elu):
        super(RecurrentStateSpaceModel, self).__init__()
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.state_dim = latent_dim * n_atoms
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc_state_action = nn.Linear(self.state_dim + action_dim, hidden_dim)

        self.fc_state_prior1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_prior2 = nn.Linear(hidden_dim, self.latent_dim*self.n_atoms)

        self.fc_state_posterior1 = nn.Linear(rnn_hidden_dim + 1024, hidden_dim)
        self.fc_state_posterior2 = nn.Linear(hidden_dim, self.latent_dim*self.n_atoms)

        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act

    def forward(self, state, action, rnn_hidden, embedded_next_obs, nonterms):
        """
            h_t = f(h_t-1, s_t-1, a_t-1)
            Return prior p(s_t | h_t) and posterior p(s_t | h_t, o_t)
            for model training
            return 
        """
        logits_prior, z_prior, rnn_hidden = self.prior(state, action, rnn_hidden, nonterms)
        logits_post, z_post = self.posterior(rnn_hidden, embedded_next_obs)

        return logits_prior, z_prior, logits_post, z_post, rnn_hidden

    def prior(self, state, action, rnn_hidden, nonterms=True):
        """
            h_t = f(h_t-1, s_t-1, a_t-1)
            Compute prior p(s_t | h_t)
            prior_s_t : Categorical

            state:      (batch_size, state_dim)
            action:     (batch_size, action_dim)
            rnn_hidden: (batch_size, rnn_hidden_dim)
            out:        (batch_size, latent_dim, n_atoms)
        """

        hidden = self.act(self.fc_state_action(torch.cat([state*nonterms, action*nonterms], dim=1)))
        rnn_hidden = self.rnn(hidden, rnn_hidden*nonterms)

        hidden = self.act(self.fc_state_prior1(rnn_hidden))
        logits = self.fc_state_prior2(hidden)

        logits = torch.reshape(logits, (logits.shape[0], self.latent_dim, self.n_atoms)) # dim:[batch_size, latent_dim, n_atoms]

        #: batch_shape=[batch_size, latent_dim] event_shape=[n_atoms]
        dist = td.OneHotCategorical(logits=logits)
        # print(dist.batch_shape)
        # print(dist.event_shape)

        z = dist.sample()

        #: Reparameterization trick for OneHotCategorcalDist
        z = z + dist.probs - dist.probs.detach()

        return logits, z, rnn_hidden  # z = dim:[batch_size, latent_dim, n_atoms]

    def posterior(self, rnn_hidden, embedded_obs):
        """
            Compute posterior q(s_t | h_t, o_t)
            posterior_s_t : Categorical

            state:      (batch_size, 1024)
            rnn_hidden: (batch_size, rnn_hidden_dim)
            out:        (batch_size, latent_dim, n_atoms)
        """
        hidden = self.act(self.fc_state_posterior1(torch.cat([rnn_hidden, embedded_obs], dim=1)))
        logits = self.fc_state_posterior2(hidden)

        logits = torch.reshape(logits, (logits.shape[0], self.latent_dim, self.n_atoms)) # dim:[batch_size, latent_dim, n_atoms]

        #: batch_shape=[batch_size, latent_dim] event_shape=[n_atoms]
        dist = td.OneHotCategorical(logits=logits)
        # print(dist.batch_shape)
        # print(dist.event_shape)

        z = dist.sample().to(torch.float)

        #: Reparameterization trick for OneHotCategorcalDist
        z = z + dist.probs - dist.probs.detach()

        return logits, z # dim:[batch_size, latent_dim, n_atoms]


class ObservationModel(nn.Module):
    """
        p(o_t | s_t, h_t)
        Observation model to reconstruct image observation
        from state and rnn hidden state
        
        feat: (batch_size, state_dim + rnn_hidden_dim)
        out:  (batch_size, 1, height, width)
    """
    def __init__(self, state_dim, rnn_hidden_dim):
        super(ObservationModel, self).__init__()
        # model1:dim[128, 128]
        # self.fc = nn.Linear(state_dim + rnn_hidden_dim, 1024)
        # self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2) # 1024x1x1 -> 128x5x5
        # self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2) # 128x5x5 -> 64x13x13
        # self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2) # 64x13x13 -> 32x29x29
        # self.dc4 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2) # 32x29x29 -> 16x62x62
        # self.dc5 = nn.ConvTranspose2d(16, 1, kernel_size=6, stride=2) # 16x62x62 -> 1x128x128

        # model2:dim[64, 64]
        # Dreamer v2 : layer=1+4, depth=192, 96, 48, act=elu, kernels: [5, 5, 6, 6], stride=2, (norm=layer)
        # Dreamer v1 : layer=1+4, depth=128, 64, 32, act=relu, kernels: [5, 5, 6, 6], stride=2,
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2) # 1024x1x1 -> 128x5x5
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2) # 128x5x5 -> 64x13x13
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2) # 64x13x13 -> 32x30x30
        self.dc4 = nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2) # 32x30x30 -> 1x64x64

    def forward(self, state, rnn_hidden):
        # model1
        # hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        # hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        # hidden = F.elu(self.dc1(hidden))
        # hidden = F.elu(self.dc2(hidden))
        # hidden = F.elu(self.dc3(hidden))
        # hidden = F.elu(self.dc4(hidden))
        # obs = self.dc5(hidden)

        # model2
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.elu(self.dc1(hidden))
        hidden = F.elu(self.dc2(hidden))
        hidden = F.elu(self.dc3(hidden))
        obs = self.dc4(hidden)

        return obs


class RewardModel(nn.Module):
    """
        p(r_t | s_t, h_t)
        Reward model to predict reward from state and rnn hidden state

        feat: (batch_size, state_dim + rnn_hidden_dim)
        out:  (batch_size, 1)
    """
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward


class DiscountModel(nn.Module):
    """
        p(gamma_t | s_t, h_t)
        Discount model to predict gamma from state and rnn hidden state

        feat: (batch_size, state_dim + rnn_hidden_dim)
        out:  (batch_size, 1)
    """
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(DiscountModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        discount = self.fc4(hidden)
        return discount


class ValueModel(nn.Module):
    """
        Value model to predict state-value of current policy (action_model)
        from state and rnn_hidden

        feat: (batch_size, state_dim + rnn_hidden_dim)
        out:  (batch_size, 1)
    """
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act=F.elu):
        super(ValueModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        state_value = self.fc4(hidden)
        return state_value


class ActionModel(nn.Module):
    """
        Action model to compute action from state and rnn_hidden

        feat: (batch_size, state_dim + rnn_hidden_dim)
        out:  (batch_size, action_dim)
    """
    def __init__(self, state_dim, rnn_hidden_dim, action_dim, hidden_dim=400, action_discrete=True,
                 epsilon_begin=0.8 ,epsilon_end=0.01 ,epsilon_decay=50000, act=F.elu):
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.act = act
        self.action_dim = action_dim

        self.action_discrete = action_discrete

        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_func = lambda step: max(self.epsilon_end, self.epsilon_begin - (self.epsilon_begin - self.epsilon_end) * (step / self.epsilon_decay))

    def forward(self, state, rnn_hidden):
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        logits = self.fc4(hidden)

        if self.action_discrete == True:
            #: batch_shape=[batch_size] event_shape=[action_dim]
            action_dist = td.OneHotCategorical(logits=logits)

            actions_sample = action_dist.sample()
            action = actions_sample + action_dist.probs - action_dist.probs.detach() # straight-through gradients(discrete)
            
        else: # continuous control
            raise NotImplementedError

        # if self.action_discrete == True:
        #     if training:
        #         #: batch_shape=[batch_size] event_shape=[action_dim]
        #         action_dist = td.OneHotCategorical(logits=logits)

        #         actions_sample = action_dist.sample()
        #         action = actions_sample + action_dist.probs - action_dist.probs.detach() # straight-through gradients(discrete)
        #     else:
            
        # else: # continuous control
        #     # refer Dreamer v1's paper
        #     mean = self.fc_mean(hidden)
        #     mean = 5.0 * torch.tanh(mean / 5.0)
        #     stddev = self.fc_stddev(hidden)
        #     stddev = F.softplus(stddev + self.init_stddev) + self.min_stddev
            
        #     if training:
        #         action_dist = Normal(mean, stddev)

        #         actions_sample = action_dist.rsample()
        #         action = torch.tanh(actions_sample)
        #     else:
        #         action = torch.tanh(mean)

        return action, action_dist # dim:[B, action_space]

    def get_greedy_action(self, action_dist: torch.distributions):
        if self.action_discrete == True:
            action_prob = action_dist.probs
            index = torch.argmax(action_prob)
            action_onehot = torch.zeros_like(action_prob)
            action_onehot[:, index] = 1

        else: # continuous control
            raise NotImplementedError
        
        return action_onehot # dim:[1, action_space]

    def add_exploration(self, action_onehot: torch.Tensor, itr: int):
        if self.action_discrete == True:
            # epsilon_greedy
            epsilon = self.epsilon_func(itr)
            if random.random() < epsilon:
                # action = random.randrange(self.action_dim)
                index = torch.randint(0, self.action_dim, action_onehot.shape[:-1], device=action_onehot.device)
                action_onehot = torch.zeros_like(action_onehot)
                action_onehot[:, index] = 1

        else: # continuous control
            raise NotImplementedError

        return action_onehot # dim:[1, action_space]