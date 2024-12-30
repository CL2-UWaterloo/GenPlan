import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import csv
import logging
from xmlrpc.client import boolean
from omegaconf import OmegaConf

# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.trainer_babyai import Trainer, TrainerConfig
from collections import deque
import random
import torch
import pickle

# import gym
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper
import blosc
import argparse
from mingpt.instruction_process import InstructionsPreprocessor
from mingpt.wandb_logger import WandbLogger
import os
import json
from mingpt.utils import AttrDict

config = OmegaConf.load("config/config_GoToObjMazeS4.yaml")

set_seed(config.seed)
if config.stochastic:
    from mingpt.diffusion_transformer_entropy import (
        GENPLAN_GPT,
        GPTConfig,
        token2idx,
        tokens,
    )


# normalize data


def get_rewards(actions):
    rewards = np.zeros_like(actions).astype(float)
    dones = np.zeros_like(actions).astype(float)

    for i in range(actions.shape[0]):
        if actions[i].item() == 5:
            rewards[i] = 0.1
        elif actions[i].item() == 4:
            rewards[i] = 0  # 0.1
        elif actions[i].item() == 3:
            rewards[i] = 0.1
        elif i == actions.shape[0] - 1:
            rewards[i] = 1
            dones[i] = 1
            pass
        else:
            rewards[i] = 0  # -0.1
    return rewards, dones


class filter_trajs:
    def __init__(self, trajs):
        self.trajs = trajs
        self.min = 10000  # float('inf')  # Set to infinity initially
        self.max = 5  # float('-inf') # Set to negative infinity initially

    def filter_trajs(self, threshold=20):
        filtered_trajs = []
        for traj in self.trajs:
            if len(traj[2]) > threshold:
                filtered_trajs.append(traj)
                if len(traj[2]) > self.max:
                    self.max = len(traj[2])
                if len(traj[2]) < self.min:
                    self.min = len(traj[2])
        self.trajs = filtered_trajs
        return self.trajs, self.min, self.max


def reward_to_go(rewards, average: bool = False) -> np.ndarray:
    """Compute the reward to go for each timestep.

    The implementation is iterative because when I wrote a vectorized version, np.cumsum
    cauased numerical instability.
    """

    lengths = rewards.shape[0]
    max_episode_steps = np.max(lengths)

    reverse_reward_to_go = np.inf * np.ones_like(rewards)
    running_reward = 0
    for i, (reward) in enumerate(rewards[::-1]):
        if i == lengths - 1:
            running_reward = 0
        running_reward += reward
        reverse_reward_to_go[i] = running_reward
    cum_reward_to_go = reverse_reward_to_go[::-1].copy()

    avg_reward_to_go = np.inf * np.ones_like(cum_reward_to_go)
    return avg_reward_to_go if average else cum_reward_to_go


def one_hot_encode(actions, num_classes):
    one_hot = np.full((actions.size, num_classes), -1)  # Fill with -1
    one_hot[np.arange(actions.size), actions.flatten()] = 1
    return one_hot


class BERTDataset(Dataset):

    def __init__(self, block_size, dataset_path, env, rate, plan_horizon):
        self.block_size = block_size
        self.inst_preprocessor = InstructionsPreprocessor()
        with open(dataset_path, "rb") as f:

            self.trajs = pickle.load(f)
            self.trajs, mini, maxi = filter_trajs(self.trajs).filter_trajs(plan_horizon//2)
            self.trajs = self.trajs[:1000]

        self.insts = []
        self.max_inst_len = 0
        self.vocab_size = len(tokens)
        lengths = []
        for traj in self.trajs:
            tmp_inst = self.inst_preprocessor(traj[0])
            self.insts.append(tmp_inst)
            self.max_inst_len = max(self.max_inst_len, len(tmp_inst))
            lengths.append(len(traj[3]))
        self.max_inst_len += 1
        self.env = env
        self.rate = rate
        self.plan_horizon = plan_horizon
        self.full_obs_shape = blosc.unpack_array(self.trajs[0][1])[0].shape[0]
        self.state_dim = 7  # len(self.trajs[0][2][0])
        self.count = 0

    def __len__(self):
        return len(self.trajs)

    def get_init_states(self, states):
        return np.copy(states[0])

    def get_full_obs(self, full_image):
        return np.copy(full_image[0])

    def generate_negative_series(length):
        return [-0.1 * i for i in range(1, length + 1)]

    def update_rtgs(self, actions, rtgs):
        action_5_count = 1  #
        rtgs[-1] = 1
        for i in range(actions.shape[0]):
            if actions[i] == 5:
                rtgs[i] = 100 * action_5_count
                action_5_count += 1
            else:
                rtgs[i] = -0.1 * (i + 1)
        rtgs[-1] = (
            rtgs[-1]
            + ((action_5_count - 1) * 100)
            + (-0.1 * (actions.shape[0] - action_5_count))
        )
        return rtgs[-1]

    def __getitem__(self, idx):
        block_size = self.block_size // self.rate

        instruction = self.insts[idx]
        instruction = np.concatenate(
            [np.zeros(self.max_inst_len - len(instruction)), instruction]
        )
        instruction = torch.from_numpy(instruction).to(dtype=torch.long)

        traj = self.trajs[idx]

        si = random.randint(0, len(traj[3]) - plan_horizon//2)

        states = np.array(traj[2])[si : si + block_size]
        states = states.reshape(len(states), -1)
        actions = traj[3]
        actions = np.array([action.value for action in actions]).reshape(-1, 1)
        rewards, dones = get_rewards(actions)  # we don't use it
        rtgs = reward_to_go(rewards)

        actions = actions[si : si + block_size]
        rtgs = rtgs[si : si + block_size]
        rewards = rewards[si : si + block_size]
        dones = dones[si : si + block_size]

        full_image = blosc.unpack_array(traj[1])[si : si + block_size]

        init_state = self.get_init_states(states)
        init_image = self.get_full_obs(full_image)

        tlen = states.shape[0]
        states = np.concatenate(
            [
                states,
                np.ones((block_size - tlen, states.shape[1])) * self.full_obs_shape - 1,
            ],
            axis=0,
        )

        full_image = np.concatenate(
            [
                full_image,
                np.zeros(
                    (
                        block_size - tlen,
                        full_image.shape[1],
                        full_image.shape[2],
                        full_image.shape[3],
                    )
                ),
            ],
            axis=0,
        )
        actions = np.concatenate(
            [actions, token2idx("<-MASK->") * np.ones((block_size - tlen, 1))], axis=0
        )
        rtgs = np.concatenate([rtgs, np.zeros((block_size - tlen, 1))], axis=0)
        rewards = np.concatenate([rewards, np.zeros((block_size - tlen, 1))], axis=0)
        dones = np.concatenate([dones, np.zeros((block_size - tlen, 1))], axis=0)

        msk = random.randint(0, tlen - 1)
        state_msk = np.zeros((tlen, 1))
        action_msk = np.ones((tlen, 1)).astype(boolean)

        masked_action = np.copy(actions)

        state_msk = np.concatenate(
            [state_msk, np.zeros((block_size - tlen, state_msk.shape[1]))], axis=0
        )
        action_msk = np.concatenate(
            [action_msk, np.zeros((block_size - tlen, 1))], axis=0
        )

        states[~action_msk.astype(np.bool_).flatten(), 2] = 0

        states_stats = {
            "min": 0,
            "max": self.full_obs_shape,  # Assuming self.env_size is defined somewhere in your class or script
        }

        actions_stats = {"min": 0, "max": 1}

        actions_one_hot = one_hot_encode(actions.astype(int), self.vocab_size)

        states = torch.from_numpy(states).to(dtype=torch.float32)
        actions = torch.from_numpy(actions).to(dtype=torch.long)
        masked_action = torch.from_numpy(masked_action).to(dtype=torch.long)
        rtgs = torch.from_numpy(rtgs).to(dtype=torch.float32)
        rewards = torch.from_numpy(rewards).to(dtype=torch.float32)
        dones = torch.from_numpy(dones).to(dtype=torch.float32)
        timesteps = torch.tensor([si], dtype=torch.int64).unsqueeze(1)
        state_msk = torch.tensor(state_msk, dtype=torch.bool)
        action_msk = torch.tensor(action_msk, dtype=torch.bool)

        init_state = torch.from_numpy(init_state).to(dtype=torch.long)
        init_image = torch.from_numpy(init_image).to(dtype=torch.float32)
        full_image = torch.from_numpy(full_image).to(dtype=torch.float32)
        actions_one_hot = torch.from_numpy(actions_one_hot).to(dtype=torch.float32)
        return (
            states,
            actions,
            actions_one_hot,
            masked_action,
            full_image,
            state_msk,
            action_msk,
            rewards,
            rtgs,
            timesteps,
            instruction,
            init_state,
            init_image,
            dones,
        )


# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
env_name = "BabyAI-" + config.env + "-v0"


env = gym.make(env_name)
test_env = gym.make(env_name)
test_env = FullyObsWrapper(test_env)
env = FullyObsWrapper(env)
env = SymbolicObsWrapper(env)
test_env = SymbolicObsWrapper(test_env)
print(
    f"env_name {env_name}!!!!!!! mcmc {config.sample_iteration}, horizon {config.horizon}, seed {config.seed}\n"
)
rate = 3 if config.model_type == "reward_conditioned" else 2
max_timesteps = 1024
plan_horizon = config.horizon

dataset_path = "./babyai/demos/" + env_name + "_agent.pkl"
bert_train_dataset = BERTDataset(
    config.context_length * rate, dataset_path, env, rate, plan_horizon
)

mconf = GPTConfig(
    bert_train_dataset.vocab_size,
    bert_train_dataset.block_size,
    noise=config.noise,
    action_horizon=config.action_horizon,
    n_layer=4,
    n_head=4,
    n_embd=128,
    model_type=config.model_type,
    max_timestep=max_timesteps,
    env_size=bert_train_dataset.full_obs_shape,
    state_dim=bert_train_dataset.state_dim,
    sample_iteration=config.sample_iteration,
    horizon=config.horizon,
    extra_config=config,
    env=env,
)
bert_model = GENPLAN_GPT(mconf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(
    max_epochs=config.epochs,
    batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=2 * len(bert_train_dataset) * config.context_length * 3,
    num_workers=4,
    seed=config.seed,
    model_type=config.model_type,
    max_timestep=max_timesteps,
)



namen = "diffusion"
logger = WandbLogger(
    config = OmegaConf.to_container(config, resolve=True),
    project="flow",
    group=f"{env_name}",
    name=namen + f"_{config.horizon}" + f"_{config.action_horizon}_{config.noise}",
    log_dir="./logs",
)
logger.save_config(config=OmegaConf.to_container(config, resolve=True), verbose=False)
trainer = Trainer(
    bert_model,
    bert_train_dataset,
    tconf,
    env,
    test_env,
    env_name,
    rate,
    plan_horizon,
    config.sample_iteration,
    bert_train_dataset.inst_preprocessor,
    bert_train_dataset.full_obs_shape,
    logger,
    config.num_buffers,
    namen + f"_{config.horizon}" + f"_{config.action_horizon}_{config.noise}",
)
trainer.train()
logger.finish()
