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

from agent import Agent
from model import (Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel, DiscountModel,
                   ValueModel, ActionModel)
from utils import TransitionBuffer, preprocess_obs, preprocess_reward, compute_return, save_eval_numpy
from env import Environment_410, DataExchange, Read2HSCv2
from module import get_parameters, FreezeParameters
from config import Config


def main(args):
    config = Config(
        exp_info=args.exp_info,
        dir=args.dir,
        bottom_height=args.bottom_height,
        save_buffer_epi_interval=args.save_buffer_epi_interval
    )

    load_weight_episode = args.load_weight_episode

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(config.latent_dim, config.n_atoms,
                                    config.action_space, config.rnn_hidden_dim).to(device)
    obs_model = ObservationModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)

    actor_model = ActionModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim, config.action_space,
                              config.hidden_dim, config.action_discrete,
                              config.epsilon_begin ,config.epsilon_end ,config.epsilon_decay).to(device)

    # load learned world model
    model_load_path_encoder = '{}/Train_Weights/epi={}_encoder.pth'.format(config.dir, load_weight_episode)
    model_load_path_rssm = '{}/Train_Weights/epi={}_rssm.pth'.format(config.dir, load_weight_episode)
    model_load_path_obs = '{}/Train_Weights/epi={}_obs_model.pth'.format(config.dir, load_weight_episode)
    model_load_path_actor = '{}/Train_Weights/epi={}_actor_model.pth'.format(config.dir, load_weight_episode)

    encoder.load_state_dict(torch.load(model_load_path_encoder))
    rssm.load_state_dict(torch.load(model_load_path_rssm))
    obs_model.load_state_dict(torch.load(model_load_path_obs))
    actor_model.load_state_dict(torch.load(model_load_path_actor))
    
    print("Loaded Model")
    print(encoder)
    print(rssm)
    print(obs_model)
    print(actor_model)

    # define env
    r_hsc = Read2HSCv2(height=config.observation_space[1], width=config.observation_space[2])
    data_ex = DataExchange(config.dir, config.exp_info)
    env = Environment_410(r_hsc, data_ex, config.action_discrete, config.bottom_height, config.episode_end_steps)
    
    # episode終了フラグ
    env.done = False

    env.data_ex.serial_reset() # serial loop_out and close
    
    print()
    print("eval loop start")
    env_iteration = 0

    # main loop
    for episode in range(load_weight_episode-1, config.all_episodes):
        print("episode: ", episode+1)

        # ----------------------------------------------
        #      evaluation without exploration noise
        # ----------------------------------------------

        print()
        print('Evaluation without exploration noise')
        
        start = time.perf_counter()
        policy = Agent(encoder, rssm, actor_model, obs_model, config.observation_space)

        hsc_image = []
        recon_image = []

        env.data_ex.reset_button_nanoKON()
        obs, end_sign = env.preparation_for_next_episode(episode+1)
        done = False
        total_reward = 0

        episode_start_time = time.perf_counter()
        step_timer = 0
        step_end_timer = 0
        
        # 終了判定
        if end_sign:
            print('-----nanokon Save Stop-----')
            break
        
        while not done:
            step_timer = time.perf_counter() - episode_start_time
            env_iteration += 1

            action = policy(obs, 0, training=False)
            hsc_image.append(obs)
            recon_image.append(policy.recon_obs)

            obs, reward, done, end_sign = env.step(action, step_timer, train_mode=False)
            total_reward += reward

            step_end_timer = time.perf_counter() - episode_start_time
            print("eval_time: {0:.5f}".format(step_end_timer - step_timer))

            # 実行周期の調整：loop_timeだけ経つまで，delayさせる
            while step_end_timer - step_timer < config.loop_time:
                step_end_timer = time.perf_counter() - episode_start_time

        env.data_ex.serial_reset() # serial loop_out and close

        save_eval_numpy(env.data_ex.directory, hsc_image, recon_image, episode+1, config.exp_info)
        hsc_image = []
        recon_image = []

        print('Total test reward at episode [%4d/%4d] is %f' %
                (episode+1, config.all_episodes, total_reward))
        print('elasped time for test: %.2fs' % (time.perf_counter() - start))

        print()
        print('1 episode finished')
        

    print('Finish')
    env.data_ex.end_process()
    env.r_hsc.end_hsc()
    print()
    print("Closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='model1')
    parser.add_argument('--exp_info', type=str, default='test')
    parser.add_argument('--bottom_height', type=int, default=82)
    parser.add_argument('--save_buffer_epi_interval', type=int, default=20) # not use

    parser.add_argument('--load_weight_episode', type=int, default=60)# only test.py

    args = parser.parse_args()
    main(args)
