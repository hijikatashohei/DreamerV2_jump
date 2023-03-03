import argparse
from datetime import datetime
import json
import os
import sys
from pprint import pprint
import time
import math
import numpy as np
import random
import torch
import torch.distributions as td
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from torch.optim import Adam

import csv
import datetime

# from agent import Agent
from model import (Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel, DiscountModel,
                   ValueModel, ActionModel)
from utils import TransitionBuffer, preprocess_obs, preprocess_reward, compute_return, save_eval_numpy
# from env import Environment_410, DataExchange, Read2HSCv2
from module import get_parameters, FreezeParameters
from config import Config


def main(args):
    config = Config(
        exp_info=args.exp_info,
        dir=args.dir,
        bottom_height=args.bottom_height,
        save_buffer_epi_interval=args.save_buffer_epi_interval
    )

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define replay buffer
    replay_buffer = TransitionBuffer(config.buffer_capacity, config.observation_space, config.action_space,
                                     config.chunk_length, config.batch_size, config.observation_dtype, config.action_dtype)

    assert os.path.isdir(config.dir) == True, "directory:{} does not exist".format(config.dir)

    replay_buffer.load_buffer(config.dir, config.load_buffer_episode, config.load_buffer_idx)

    # define models and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(config.latent_dim, config.n_atoms,
                                    config.action_space, config.rnn_hidden_dim).to(device)
    obs_model = ObservationModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)
    reward_model = RewardModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)
    discount_model = DiscountModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)

    world_list = [encoder, rssm, obs_model, reward_model, discount_model]

    model_optimizer = Adam(get_parameters(world_list), lr=config.model_lr, eps=config.eps)

    print(encoder)
    print(rssm)
    print(obs_model)
    print(reward_model)
    print(discount_model)

    # フォルダを作成
    os.chdir(config.dir)

    os.makedirs('Pre_Train_Weights')
    os.chdir('Pre_Train_Weights')
    os.makedirs('iter_weight')

    date = datetime.datetime.today()
    a0 = date.year
    b0 = date.month
    c0 = date.day
    a=str(a0).zfill(4)
    b=str(b0).zfill(2)
    c=str(c0).zfill(2)

    count = 1

    file_count_0 = count
    file_count = str(file_count_0).zfill(2)
    # ログデータのヘッダー
    header = ['imagine_iteration','model_loss','kl_loss','obs_loss','reward_loss','discount_loss']
    f = open(str(a) + str(b) + str(c) + '305' + str(file_count) + '_' + config.exp_info + '.csv','a',newline="")
    writer = csv.writer(f)
    writer.writerow(header)

    os.chdir('../')
    os.chdir('../')

    print()
    print("main train loop start")
    imagine_iteration = 0

    # main training loop
    start = time.perf_counter()
    for update_step in range(config.pre_train_iter):
        imagine_iteration += 1

        # ---------------------------------------------------------------
        #      update model (encoder, rssm, obs_model, reward_model, discount_model)
        # ---------------------------------------------------------------
        observations, actions, rewards, dones = replay_buffer.sample() # dim:[L, B, *x.shape]

        # preprocess observations and transpose tensor for RNN training
        observations = preprocess_obs(observations)
        rewards = preprocess_reward(rewards) # transform tanh()
        
        observations = torch.as_tensor(observations, dtype=torch.float32, device=device) # dim:[L, B, C, H, W]  t ~ t+L
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device) # dim:[L, B, action_space]  t-1 ~ t+L-1
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1) # dim:[L, B, 1]  t-1 ~ t+L-1
        nonterms = torch.as_tensor(1-dones, dtype=torch.float32, device=device).unsqueeze(-1) # dim:[L, B, 1]  t-1 ~ t+L-1

        # embed observations with CNN
        embedded_observations = encoder(
            observations.reshape(-1, *config.observation_space)).view(config.chunk_length, config.batch_size, -1)

        # prepare Tensor to maintain states sequence and rnn hidden states sequence <-- dim:[L, B, state_dim or rnn_hidden_dim]
        states = torch.zeros(config.chunk_length, config.batch_size, config.latent_dim*config.n_atoms, device=device) #  t ~ t+L
        rnn_hiddens = torch.zeros(config.chunk_length, config.batch_size, config.rnn_hidden_dim, device=device)

        # initialize state and rnn hidden state with 0 vector
        state = torch.zeros(config.batch_size, config.latent_dim*config.n_atoms, device=device)
        rnn_hidden = torch.zeros(config.batch_size, config.rnn_hidden_dim, device=device)

        # compute state and rnn hidden sequences and kl loss
        kl_loss = 0
        for l in range(config.chunk_length):
            logits_prior, _, logits_post, z_post, rnn_hidden = \
                rssm(state, actions[l], rnn_hidden, embedded_observations[l], nonterms[l])
            state = torch.reshape(z_post, (z_post.shape[0], -1)) # dim:[B, latent_dim, n_atoms] -->[B, latent_dim*n_atoms]
            states[l] = state
            rnn_hiddens[l] = rnn_hidden

            # KL Balancing: See 2.2 BEHAVIOR LEARNING Algorithm 2
            # logits dim:[B, latent_dim, n_atoms] 
            # dist: batch_shape=[batch_size, latent_dim] event_shape=[n_atoms]
            
            kl_div1 = kl_divergence(
                td.OneHotCategorical(logits=logits_post.detach()),
                td.OneHotCategorical(logits=logits_prior)
                ).mean()
            kl_div2 = kl_divergence(
                td.OneHotCategorical(logits=logits_post),
                td.OneHotCategorical(logits=logits_prior.detach())
                ).mean()

            kl_div1 = kl_div1.clamp(min=config.free_nats)
            kl_div2 = kl_div2.clamp(min=config.free_nats)

            kl_loss += config.kl_alpha * kl_div1 + (1. - config.kl_alpha) * kl_div2

        kl_loss /= config.chunk_length

        # calculate reconstructed observations and predicted rewards and discounts losses
        flatten_states = states[:-1].view(-1, config.latent_dim*config.n_atoms) # dim:[L-1*B, state_dim]  t ~ t+L-1
        flatten_rnn_hiddens = rnn_hiddens[:-1].view(-1, config.rnn_hidden_dim) # dim:[L-1*B, rnn_hidden_dim]  t ~ t+L-1

        recon_observations = obs_model(flatten_states, flatten_rnn_hiddens).view(
            config.chunk_length-1, config.batch_size, *config.observation_space) # dim:[L-1, B, C, H, W]  t ~ t+L-1
        predicted_rewards = reward_model(flatten_states, flatten_rnn_hiddens).view(
            config.chunk_length-1, config.batch_size, 1) # dim:[L-1, B, 1]  t ~ t+L-1
        predicted_discounts = discount_model(flatten_states, flatten_rnn_hiddens).view(
            config.chunk_length-1, config.batch_size, 1) # dim:[L-1, B, 1]  t ~ t+L-1

        obs_dist = td.Independent(td.Normal(recon_observations, 1), reinterpreted_batch_ndims=len(config.observation_space)) # batch_shape=[L-1, B], event_shape=[C, H, W]
        obs_loss = -torch.mean(obs_dist.log_prob(observations[:-1]))

        reward_dist = td.Independent(td.Normal(predicted_rewards, 1), reinterpreted_batch_ndims=1)
        reward_loss = -torch.mean(reward_dist.log_prob(rewards[1:]))

        discount_dist = td.Independent(td.Bernoulli(logits=predicted_discounts), reinterpreted_batch_ndims=1)
        discount_loss = -torch.mean(discount_dist.log_prob(nonterms[1:]))

        # sum all losses
        model_loss = config.kl_scale * kl_loss + obs_loss + reward_loss + config.discount_scale * discount_loss
        
        model_optimizer.zero_grad()
        model_loss.backward()
        clip_grad_norm_(get_parameters(world_list), config.clip_grad_norm)
        model_optimizer.step()

        # print losses
        print('imagine_iter: %3d, model_loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: %.5f, discount_loss: %.5f'
                % (imagine_iteration, model_loss.item(), kl_loss.item(), obs_loss.item(), reward_loss.item(), discount_loss.item()))

        loss_data = []

        loss_data.append(imagine_iteration)
        loss_data.append(model_loss.item())
        loss_data.append(kl_loss.item())
        loss_data.append(obs_loss.item())
        loss_data.append(reward_loss.item())
        loss_data.append(discount_loss.item())

        writer.writerow(loss_data)
        loss_data = []

        print('elasped time for update: %.2fs' % (time.perf_counter() - start))

        if (update_step+1) % 5000 == 0:
            # save learned model parameters
            torch.save(encoder.state_dict(), config.dir + '/Pre_Train_Weights/iter_weight/iter=' +  str(update_step+1) + '_encoder.pth')
            torch.save(rssm.state_dict(), config.dir + '/Pre_Train_Weights/iter_weight/iter=' + str(update_step+1) + '_rssm.pth')
            torch.save(obs_model.state_dict(), config.dir + '/Pre_Train_Weights/iter_weight/iter=' + str(update_step+1) + '_obs_model.pth')
            torch.save(reward_model.state_dict(), config.dir + '/Pre_Train_Weights/iter_weight/iter=' + str(update_step+1) + '_reward_model.pth')
            torch.save(discount_model.state_dict(), config.dir + '/Pre_Train_Weights/iter_weight/iter=' + str(update_step+1) + '_discount_model.pth')

    # save learned model parameters
    torch.save(encoder.state_dict(), config.dir + '/Pre_Train_Weights/iter=' + str(config.pre_train_iter) + '_encoder.pth')
    torch.save(rssm.state_dict(), config.dir + '/Pre_Train_Weights/iter=' + str(config.pre_train_iter) + '_rssm.pth')
    torch.save(obs_model.state_dict(), config.dir + '/Pre_Train_Weights/iter=' + str(config.pre_train_iter) + '_obs_model.pth')
    torch.save(reward_model.state_dict(), config.dir + '/Pre_Train_Weights/iter=' + str(config.pre_train_iter) + '_reward_model.pth')
    torch.save(discount_model.state_dict(), config.dir + '/Pre_Train_Weights/iter=' + str(config.pre_train_iter) + '_discount_model.pth')

    print("learned world model")

    print('Finish')
    print()
    print("Closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='model1')
    parser.add_argument('--exp_info', type=str, default='pre_train')
    parser.add_argument('--bottom_height', type=int, default=82) # not use
    parser.add_argument('--save_buffer_epi_interval', type=int, default=20) # not use

    args = parser.parse_args()
    main(args)
