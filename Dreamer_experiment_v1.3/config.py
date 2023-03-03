import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Tuple, Dict


@dataclass
class Config():
    # args setting
    exp_info : str
    dir : str

    bottom_height : float

    save_buffer_epi_interval : int


    pre_train : bool = True
    pre_train_iter : int = 10000
    
    load_buffer : bool = True
    load_buffer_episode : int = 40
    load_buffer_idx : int = 20000


    # DreamerV2 Hyperparameter
    latent_dim : int = 32
    n_atoms : int = 32
    rnn_hidden_dim : int = 512
    hidden_dim : int = 400
    buffer_capacity : int = 100000
    all_episodes : int = 1000
    seed_episodes : int = 40
    collect_interval : int = 100
    batch_size : int = 32
    chunk_length : int = 32
    imagination_horizon : int = 15
    gamma : float = 0.95 # paper=0.995
    lambda_ : float = 0.95 # paper=0.95
    model_lr : float = 1e-4 # paper=2e-4,dmc=3e-4
    action_lr : float = 1e-4 # paper=4e-5,dmc=8e-5
    value_lr : float = 1e-4 # paper=1e-4,dmc=8e-5
    eps : float = 1e-6 # paper=1e-5
    clip_grad_norm : int = 100
    free_nats : int = 0.0
    kl_alpha : float = 0.8
    kl_scale : float = 1.0
    discount_scale : float = 5.0
    actor_ent_weight : float = 1e-3

    target_interval : int = 100
    slow_target : bool = True
    slow_target_fraction : int = 1

    epsilon_begin : float = 0.8
    epsilon_end : float = 0.01
    epsilon_decay : int = 30000

    # env setting
    observation_space = [1, 64, 64]
    action_space : int = 9
    # action_space : int = 6
    # action_space : int = 15
    observation_dtype : np.dtype = np.uint8
    action_dtype : np.dtype = np.float32
    action_discrete : bool = True

    # etc
    loop_time : float = 0.05
    episode_end_steps : int = 500
    # episode_end_steps : int = 200
    seed : int = 0
    eval_interval : int = 10