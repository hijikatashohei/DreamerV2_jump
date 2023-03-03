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

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define replay buffer
    replay_buffer = TransitionBuffer(config.buffer_capacity, config.observation_space, config.action_space,
                                     config.chunk_length, config.batch_size, config.observation_dtype, config.action_dtype)

    assert config.load_buffer_episode == config.seed_episodes, "Load_buffer_episode dont match seed_episodes.Should check."

    replay_buffer.load_buffer(config.dir, config.load_buffer_episode, config.load_buffer_idx)

    # define models and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(config.latent_dim, config.n_atoms,
                                    config.action_space, config.rnn_hidden_dim).to(device)
    obs_model = ObservationModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)
    reward_model = RewardModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)
    discount_model = DiscountModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)

    value_model = ValueModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)
    target_model = ValueModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim).to(device)
    with torch.no_grad():
        target_model.load_state_dict(value_model.state_dict())

    actor_model = ActionModel(config.latent_dim*config.n_atoms, config.rnn_hidden_dim, config.action_space,
                              config.hidden_dim, config.action_discrete,
                              config.epsilon_begin ,config.epsilon_end ,config.epsilon_decay).to(device)

    if config.pre_train:
        # load learned world model
        model_load_path_encoder = '{}/Pre_Train_Weights/iter={}_encoder.pth'.format(config.dir, config.pre_train_iter)
        model_load_path_rssm = '{}/Pre_Train_Weights/iter={}_rssm.pth'.format(config.dir, config.pre_train_iter)
        model_load_path_obs = '{}/Pre_Train_Weights/iter={}_obs_model.pth'.format(config.dir, config.pre_train_iter)
        model_load_path_reward = '{}/Pre_Train_Weights/iter={}_reward_model.pth'.format(config.dir, config.pre_train_iter)
        model_load_path_discount = '{}/Pre_Train_Weights/iter={}_discount_model.pth'.format(config.dir, config.pre_train_iter)

        encoder.load_state_dict(torch.load(model_load_path_encoder))
        rssm.load_state_dict(torch.load(model_load_path_rssm))
        obs_model.load_state_dict(torch.load(model_load_path_obs))
        reward_model.load_state_dict(torch.load(model_load_path_reward))
        discount_model.load_state_dict(torch.load(model_load_path_discount))
        print("Loaded World Model (Did not load Action, Value Model)")
    else:
        print("Did not load all Models")
        raise NotImplementedError

    world_list = [encoder, rssm, obs_model, reward_model, discount_model]
    value_list = [value_model]
    actor_list = [actor_model]

    model_optimizer = Adam(get_parameters(world_list), lr=config.model_lr, eps=config.eps)
    value_optimizer = Adam(get_parameters(value_list), lr=config.value_lr, eps=config.eps)
    actor_optimizer = Adam(get_parameters(actor_list), lr=config.action_lr, eps=config.eps)

    print(encoder)
    print(rssm)
    print(obs_model)
    print(reward_model)
    print(discount_model)
    print(value_model)
    print(target_model)
    print(actor_model)

    # define env
    r_hsc = Read2HSCv2(height=config.observation_space[1], width=config.observation_space[2])
    data_ex = DataExchange(config.dir, config.exp_info)
    env = Environment_410(r_hsc, data_ex, config.action_discrete, config.bottom_height, config.episode_end_steps)
    
    # episode終了フラグ
    env.done = False
    env.data_ex.serial_reset() # serial loop_out and close

    # 事前学習を行うため，seed episodeを行わない．

    print()
    print('Please push start(41)')
    env.data_ex.reset_button_nanoKON()

    while True:
        env.data_ex.catch_nanoKON()
        if env.data_ex.nanokon == 41: # ボタンで開始
            break
    
    print()
    print("main train loop start")
    imagine_iteration = 0
    env_iteration = 0


    # main training loop
    for episode in range(config.load_buffer_episode, config.all_episodes):
        # update parameters of model, value model, action model
        print("episode: ", episode+1)
        start = time.perf_counter()

        for update_step in range(config.collect_interval):
            imagine_iteration += 1

            # --------------------------------------------------------------------------
            #      update model (encoder, rssm, obs_model, reward_model, discount_model)
            # --------------------------------------------------------------------------

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

                #: KL Balancing: See 2.2 BEHAVIOR LEARNING Algorithm 2
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

            # calculate reconstructed observations and predicted rewards and discounts loss
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

            # ----------------------------------------------
            #      update value_model and actor_model
            # ----------------------------------------------
            
            # prepare initial state and rnn_hidden t=0
            with torch.no_grad():
                flatten_states = states.view(-1, config.latent_dim*config.n_atoms).detach() # dim:[L*B, state_dim]  t ~ t+L
                flatten_rnn_hiddens = rnn_hiddens.view(-1, config.rnn_hidden_dim).detach() # dim:[L*B, rnn_hidden_dim]  t ~ t+L

            # prepare tensor to maintain imaginated trajectory's states and rnn_hiddens
            imaginated_states = torch.zeros(config.imagination_horizon, *flatten_states.shape, device=flatten_states.device) # dim:[H, L*B, state_dim]
            imaginated_rnn_hiddens = torch.zeros(config.imagination_horizon, *flatten_rnn_hiddens.shape, device=flatten_rnn_hiddens.device) # dim:[H, L*B, rnn_hidden_dim]
            
            # prepare tensor to maintain action_entropy and action_log_probs
            action_entropy = torch.zeros(config.imagination_horizon, config.chunk_length*config.batch_size, device=device) # dim:[H, L*B]
            action_log_probs = torch.zeros(config.imagination_horizon, config.chunk_length*config.batch_size, device=device) # dim:[H, L*B]
            
            with FreezeParameters(world_list):
                # compute imaginated trajectory using action from actor_model <-- t = 1~Hまでのs,hの計算,t = 0~H-1 までのa_log_prob, a_entropyの計算
                for h in range(config.imagination_horizon):
                    actions, actions_dist = actor_model(flatten_states.detach(), flatten_rnn_hiddens.detach()) # dim:[L*B, action_space]
                    _, flatten_states_prior, flatten_rnn_hiddens = rssm.prior(flatten_states, actions, flatten_rnn_hiddens)
                    flatten_states = torch.reshape(flatten_states_prior, (flatten_states_prior.shape[0], -1)) # dim:[L*B, state_dim]

                    imaginated_states[h] = flatten_states # t ~ t+H
                    imaginated_rnn_hiddens[h] = flatten_rnn_hiddens
                    action_entropy[h] = actions_dist.entropy() # array=0~H-1 : t-1 ~ t+H-1 --> t ~ t+H-1を使用
                    action_log_probs[h] = actions_dist.log_prob(torch.round(actions.detach())) # <-- NOTE:need action.detach()? actions_dist have grad?

            with FreezeParameters(world_list+value_list+[target_model]+[discount_model]):
                # compute rewards and values corresponding to imaginated states and rnn_hiddens
                flatten_imaginated_states = imaginated_states.view(-1, config.latent_dim*config.n_atoms) # dim:[H*L*B, state_dim]
                flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(-1, config.rnn_hidden_dim) # dim:[H*L*B, rnn_hidden_dim]

                imaginated_rewards = reward_model(flatten_imaginated_states, flatten_imaginated_rnn_hiddens).view(
                                                  config.imagination_horizon, config.chunk_length*config.batch_size, 1) # dim:[H, L*B, 1]
                imaginated_discounts = discount_model(flatten_imaginated_states, flatten_imaginated_rnn_hiddens).view(
                                                      config.imagination_horizon, config.chunk_length*config.batch_size, 1) # dim:[H, L*B, 1]
                imaginated_target_values = target_model(flatten_imaginated_states, flatten_imaginated_rnn_hiddens).view(
                                                config.imagination_horizon, config.chunk_length*config.batch_size, 1) # dim:[H, L*B, 1])  t ~ t+H

                reward_dist = td.Independent(td.Normal(imaginated_rewards, 1), reinterpreted_batch_ndims=1)
                reward_mean = reward_dist.mean # dim:[H, L*B, 1] t ~ t+H

                discount_dist = td.Independent(td.Bernoulli(logits=imaginated_discounts), reinterpreted_batch_ndims=1)
                discount_arr = config.gamma*torch.round(discount_dist.base_dist.probs) # dim:[H, L*B, 1] t ~ t+H  mean = prob(disc==1)

            v_lambda = torch.zeros(config.imagination_horizon-1, config.chunk_length*config.batch_size, 1, device=device) # dim:[H-1, L*B, 1] t ~ t+H-1
            # v_lambda = torch.zeros(config.imagination_horizon, config.chunk_length*config.batch_size, 1, device=device) # dim:[H, L*B, 1] t ~ t+H

            # Calculate V_lambda <t=H>
            # v_lambda[-1] = reward_mean[-1] + discount_arr[-1] * imaginated_target_values[-1] # V_lambda_t=t+H dim:[L*B, 1]
            # v_lambda[-1] = imaginated_target_values[-1] # V_lambda_t=t+H dim:[L*B, 1]

            # # Calculate V_lambda <t=1:H-1>
            # for t in range(config.imagination_horizon - 2, -1, -1): # array = H-2, H-3, ..., 0  ===>  t = H-1, H-2, ..., 1
            #     v_lambda[t] = reward_mean[t] + discount_arr[t] * ((1-config.lambda_)*imaginated_target_values[t+1] + config.lambda_*v_lambda[t+1])

            v_lambda = compute_return(reward_mean[:-1], imaginated_target_values[:-1], discount_arr[:-1], bootstrap=imaginated_target_values[-1], lambda_=config.lambda_)

            discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
            discount = torch.cumprod(discount_arr[:-1], 0) # dim:[H-1, L*B, 1] t ~ t+H-1
            
            # update actor model <t=1:H-1>
            if config.action_discrete:
                objective_actor = discount * (action_log_probs[1:].unsqueeze(-1) * (v_lambda - imaginated_target_values[:-1]).detach() \
                                    + config.actor_ent_weight * action_entropy[1:].unsqueeze(-1)) # dim:[H-1, L*B, 1]
            else:
                objective_actor = discount * (v_lambda + config.actor_ent_weight * action_entropy[1:].unsqueeze(-1))
            
            actor_loss = -torch.sum(torch.mean(objective_actor, dim=1)) # 最大化のためにマイナスをかける

            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(get_parameters(actor_list), config.clip_grad_norm)
            actor_optimizer.step()
            
            # update value model <t=1:H-1>
            with torch.no_grad():
                flatten_value_imaginated_states = imaginated_states[:-1].view(-1, config.latent_dim*config.n_atoms).detach() # dim:[(H-1)*L*B, state_dim]
                flatten_value_imaginated_rnn_hiddens = imaginated_rnn_hiddens[:-1].view(-1, config.rnn_hidden_dim).detach() # dim:[(H-1)*L*B, rnn_hidden_dim]
                value_discount = discount.detach()
                value_target = v_lambda.detach()

            imaginated_values = value_model(flatten_value_imaginated_states, flatten_value_imaginated_rnn_hiddens).view(
                                            config.imagination_horizon-1, config.chunk_length*config.batch_size, 1) # dim:[H-1, L*B, 1])  t ~ t+H-1

            objective_value = 0.5 * torch.square(imaginated_values - value_target) # dim:[H-1, L*B, 1]
            value_loss = (value_discount * objective_value).mean(dim=1).sum()

            value_optimizer.zero_grad()
            value_loss.backward()
            clip_grad_norm_(get_parameters(value_list), config.clip_grad_norm)
            value_optimizer.step()

            # ターゲットネットワークを定期的に同期(or soft-target)
            if not imagine_iteration % config.target_interval:
                mix = config.slow_target_fraction if config.slow_target else 1
                for param, target_param in zip(value_model.parameters(), target_model.parameters()):
                    target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

            # print losses
            print('imagine_iter: %3d update_step: %3d model_loss: %.5f, kl_loss: %.5f, '
                  'obs_loss: %.5f, reward_loss: %.5f, discount_loss: %.5f, '
                  'value_loss: %.5f actor_loss: %.5f'
                  % (imagine_iteration, update_step+1, model_loss.item(), kl_loss.item(),
                     obs_loss.item(), reward_loss.item(), discount_loss.item(), value_loss.item(), actor_loss.item()))

            # write Loss to csv file
            env.data_ex.csv_loss_write(episode+1, imagine_iteration, update_step+1, model_loss.item(), kl_loss.item(), 
                                        obs_loss.item(), reward_loss.item(), discount_loss.item(), value_loss.item(), actor_loss.item())

        print('elasped time for update: %.2fs' % (time.perf_counter() - start))

        # save learned model parameters
        torch.save(encoder.state_dict(), config.dir + '/Train_Weights/epi=' + str(episode+1) + '_encoder.pth')
        torch.save(rssm.state_dict(), config.dir + '/Train_Weights/epi=' + str(episode+1) + '_rssm.pth')
        torch.save(obs_model.state_dict(), config.dir + '/Train_Weights/epi=' + str(episode+1) + '_obs_model.pth')
        torch.save(reward_model.state_dict(), config.dir + '/Train_Weights/epi=' + str(episode+1) + '_reward_model.pth')
        torch.save(discount_model.state_dict(), config.dir + '/Train_Weights/epi=' + str(episode+1) + '_discount_model.pth')
        torch.save(value_model.state_dict(), config.dir + '/Train_Weights/epi=' + str(episode+1) + '_value_model.pth')
        torch.save(actor_model.state_dict(), config.dir + '/Train_Weights/epi=' + str(episode+1) + '_actor_model.pth')

        # -----------------------------
        #      collect experiences
        # -----------------------------

        print()
        print("updated World Model and Actor-Critic")
        env.data_ex.reset_button_nanoKON()

        print()
        print('collect experience')

        start = time.perf_counter()
        policy = Agent(encoder, rssm, actor_model, obs_model, config.observation_space)

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

            action = policy(obs, env_iteration)

            next_obs, reward, done, end_sign = env.step(action, step_timer)
            replay_buffer.add(obs, action, reward, done)
            obs = next_obs
            total_reward += reward

            step_end_timer = time.perf_counter() - episode_start_time
            print("train_time: {0:.5f}".format(step_end_timer - step_timer))

            # 実行周期の調整：loop_timeだけ経つまで，delayさせる
            while step_end_timer - step_timer < config.loop_time:
                step_end_timer = time.perf_counter() - episode_start_time

        env.data_ex.serial_reset() # serial loop_out and close

        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, config.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.perf_counter() - start))
        
        # 終了判定
        if end_sign:
            print('-----nanokon Save Stop-----')
            break

        # ----------------------------------------------
        #      evaluation without exploration noise
        # ----------------------------------------------

        if (episode + 1) % config.eval_interval == 0:
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

                action = policy(obs, 0, training=False)
                obs, reward, done, end_sign = env.step(action, step_timer, train_mode=False)
                
                hsc_image.append(obs)
                recon_image.append(policy.recon_obs)

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
        
        # 終了判定
        if end_sign:
            print('-----nanokon Save Stop-----')
            break

        print('Please push Start(41) or End(42) or Save_Buffer(60).')
        env.data_ex.reset_button_nanoKON()

        save_buffer_tf = True

        while True:
            env.data_ex.catch_nanoKON()
            
            if env.data_ex.nanokon == 41: # 開始
                break

            elif env.data_ex.nanokon == 42: # 終了
                print("Selected End(42)")
                break

            elif env.data_ex.nanokon == 60 and save_buffer_tf: # Save Replay Buffer
                print("Selected Save_Buffer(60)")
                save_buffer_tf = False
                
                replay_buffer.save_buffer(env.data_ex.directory, episode+1)
                env.data_ex.reset_button_nanoKON()

        if env.data_ex.nanokon == 42:
            break


    print('Finish')
    env.data_ex.end_process()
    env.r_hsc.end_hsc()
    print()
    print("Closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='model1')
    parser.add_argument('--exp_info', type=str, default='train')
    parser.add_argument('--bottom_height', type=int, default=82)
    
    parser.add_argument('--save_buffer_epi_interval', type=int, default=0) # not use

    args = parser.parse_args()
    main(args)
