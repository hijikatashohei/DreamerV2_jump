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

from utils import TransitionBuffer, preprocess_obs, preprocess_reward, compute_return, save_eval_numpy
from env import Environment_410, DataExchange, Read2HSCv2
from config import Config


def main(args):
    config = Config(
        exp_info=args.exp_info,
        dir=args.dir,
        bottom_height=args.bottom_height,
        load_buffer=args.load_buffer,
        load_buffer_episode=args.load_buffer_episode,
        load_buffer_idx=args.load_buffer_idx,
        save_buffer_epi_interval=args.save_buffer_epi_interval
    )


    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)

    # define replay buffer
    replay_buffer = TransitionBuffer(config.buffer_capacity, config.observation_space, config.action_space,
                                     config.chunk_length, config.batch_size, config.observation_dtype, config.action_dtype)

    # 続きからデータを集める場合
    if config.load_buffer:
        replay_buffer.load_buffer(config.dir, config.load_buffer_episode, config.load_buffer_idx)
        start_episode = config.load_buffer_episode
    else:
        start_episode = 0

    print()
    print("Load_buffer: ", config.load_buffer)
    print()
    print("Start episode: ", start_episode)
    print("Seed episode: ", config.seed_episodes)
    print("All Steps: ", (config.seed_episodes - start_episode)*config.episode_end_steps)
    print("Save buffer interval steps: ", config.save_buffer_epi_interval*config.episode_end_steps)

    # define env
    r_hsc = Read2HSCv2(height=config.observation_space[1], width=config.observation_space[2])
    data_ex = DataExchange(config.dir, config.exp_info)
    env = Environment_410(r_hsc, data_ex, config.action_discrete, config.bottom_height, config.episode_end_steps)

    env.data_ex.serial_reset() # serial loop_out and close

    # episode終了フラグ
    env.done = False


    # collect seed episodes with random action
    for episode in range(start_episode, config.seed_episodes):
        print("Next episode:", episode+1)

        obs, end_sign = env.preparation_for_next_episode(episode+1)

        # 終了判定
        if end_sign:
            print('-----nanokon Save Stop-----')
            break

        episode_start_time = time.perf_counter()
        step_timer = 0
        step_end_timer = 0
        done = False
        total_reward = 0

        while not done:
            step_timer = time.perf_counter() - episode_start_time
            
            # random action
            if config.action_discrete:
                action_num = random.randrange(config.action_space)
                action_onehot = np.eye(1, M=config.action_space, k=action_num, dtype=np.int8) # N=行,M=列,k=1の要素
                action = np.squeeze(action_onehot, 0)
            else: # continuous control
                raise NotImplementedError

            next_obs, reward, done, end_sign = env.step(action, step_timer)
            replay_buffer.add(obs, action, reward, done)
            obs = next_obs
            
            total_reward += reward

            step_end_timer = time.perf_counter() - episode_start_time
            print("train_time: {0:.5f}".format(step_end_timer - step_timer))

            # 実行周期の調整：loop_timeだけ経つまで，delayさせる
            while step_end_timer - step_timer < config.loop_time:
                step_end_timer = time.perf_counter() - episode_start_time
        
        print('Total test reward at episode [%4d/%4d] is %f' %
                  (episode+1, config.seed_episodes, total_reward))
        
        env.data_ex.serial_reset() # serial loop_out and close

        if (episode+1) % config.save_buffer_epi_interval == 0:
            replay_buffer.save_buffer(env.data_ex.directory, episode+1)

    replay_buffer.save_buffer(env.data_ex.directory, 0)


    print("collected seed episodes with random action")

    print('Finish')
    env.data_ex.end_process()
    env.r_hsc.end_hsc()
    print()
    print("Closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='model1')
    parser.add_argument('--exp_info', type=str, default='run_seed_episode')
    parser.add_argument('--bottom_height', type=int, default=82)
    
    # run_seed_episodeではload_bufferについて，configではなく，ここで定義できる．
    parser.add_argument('--load_buffer', type=bool, default=False)
    parser.add_argument('--load_buffer_episode', type=int, default=40)
    parser.add_argument('--load_buffer_idx', type=int, default=20000)

    parser.add_argument('--save_buffer_epi_interval', type=int, default=10)

    args = parser.parse_args()
    main(args)
