import torch
import numpy as np
from utils import preprocess_obs


class Agent:
    def __init__(self, encoder, rssm, action_model, obs_model, observation_space):
        self.encoder = encoder
        self.rssm = rssm
        self.action_model = action_model
        self.obs_model = obs_model

        self.device = next(self.action_model.parameters()).device # model.parameters()があるdeviceと同じ場所で計算する
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

        self.recon_obs = np.zeros(observation_space, dtype=np.uint8)

    def __call__(self, obs, itr, training=True):

        # preprocess observation and transpose for torch style (channel-first)
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(0) # dim:[batch_size, channel, height, width]

        with torch.no_grad():
            # embed observation, compute state posterior, sample from state posterior
            # and get action using sampled state and rnn_hidden as input
            embedded_obs = self.encoder(obs)
            _, z_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = torch.reshape(z_posterior, (z_posterior.shape[0], -1)) # dim:[batch_size, latent_dim*n_atoms]
            action, action_dist = self.action_model(state, self.rnn_hidden)
            print("action prob", action_dist.probs)
            
            if training:
                action = self.action_model.add_exploration(action, itr)
            else:
                recon_obs = self.obs_model(state, self.rnn_hidden) # dim:[1, obs_space], tensor
                recon_obs = (recon_obs.squeeze().cpu().numpy() + 0.5).clip(0.0, 1.0) * 255
                self.recon_obs = recon_obs.astype(np.uint8)

                action = self.action_model.get_greedy_action(action_dist) # dim:[1, action_dim]
    
            # update rnn_hidden for next step
            _, _, self.rnn_hidden = self.rssm.prior(state, action, self.rnn_hidden)

        return action.squeeze().cpu().numpy() # dim:[action_dim]

    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
