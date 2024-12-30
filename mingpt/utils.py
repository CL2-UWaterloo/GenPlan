"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from xmlrpc.client import boolean
import random
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
from mingpt.diffusion_transformer_entropy import token2idx
from matplotlib import pyplot as plt
from torch.distributions.categorical import Categorical
import math

AGENT_ID = 10
AGENT_COLOR = 6


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out




# def reward_to_go(action_sequence , gamma = 0.99):
#     """
#     Calculate the reward-to-go for each action in the action sequence.

#     Parameters:
#     - action_sequence: List or array of actions.
#     - gamma: Discount factor.

#     Returns:
#     - A list of rewards-to-go for each action.
#     """
#     length = len(action_sequence)
#     rewards = []
#     cumulative_reward = 0

#     for i in reversed(range(length)):
#         if action_sequence[i] == 5:
#             reward = 10
#         # elif i == length - 1:
#         #     reward = 1  # Reward when the task is completed
#         else:
#             reward = -0.1  # Negative reward for non-completion actions

#         cumulative_reward = reward + gamma * cumulative_reward
#         rewards.insert(0, [cumulative_reward])

#     return rewards


def generate_state_sequence_tensor_multi_goal(start, goals, size_env, plan_horizon):
    """
    Generate a sequence of states from start to multiple goals within a grid environment using tensors.

    Parameters:
    - start: A tensor representing the starting position as (x, y).
    - goals: A list of tensors, each representing a goal position as (x, y).
    - size_env: The size of the square grid environment.

    Returns:
    - A tensor representing the sequence of states from start through all goals.
    """

    def generate_sequence(start, goal):
        sequence = []
        current = start.clone()

        while not torch.equal(current, goal):
            sequence.append(current.tolist())
            dx = 0
            dy = 0

            # Determine the direction to move in each axis
            if current[0] < goal[0]:
                dx = 1
            elif current[0] > goal[0]:
                dx = -1

            if current[1] < goal[1]:
                dy = 1
            elif current[1] > goal[1]:
                dy = -1

            # Move in the direction with priority given to x-axis to simplify the example
            if dx != 0:
                next_state = current + torch.tensor([dx, 0])
            else:
                next_state = current + torch.tensor([0, dy])

            # Ensure the next state is within bounds
            if 0 <= next_state[0] < size_env and 0 <= next_state[1] < size_env:
                current = next_state
            else:
                break  # Exit if next state is out of bounds

        if torch.equal(current, goal):
            sequence.append(goal.tolist())  # Ensure the goal is included if reached

        return sequence

    # Initialize the sequence with the start position
    total_sequence = []
    current_start = start

    # Generate sequence for each goal
    for goal in goals:
        part_sequence = generate_sequence(current_start, goal)
        if total_sequence:
            part_sequence = part_sequence[
                1:
            ]  # Remove the first element to avoid duplication
        total_sequence.extend(part_sequence)
        current_start = goal
    sequence_tensor = torch.tensor(total_sequence)
    if sequence_tensor.size(0) < plan_horizon:
        # Pad the sequence with zeros
        pad_size = plan_horizon - sequence_tensor.size(0)
        sequence_tensor = F.pad(sequence_tensor, (0, 0, 0, pad_size), "constant", 0)
    elif sequence_tensor.size(0) > plan_horizon:
        # Truncate the sequence
        sequence_tensor = sequence_tensor[:plan_horizon, :]

    return sequence_tensor


def reward_to_go(rewards, average: bool = False) -> np.ndarray:
    """Compute the reward to go for each timestep.

    The implementation is iterative because when I wrote a vectorized version, np.cumsum
    cauased numerical instability.
    """
    # dones = np.logical_or(dataset["terminals"], dataset["timeouts"])
    # _, _, lengths = util.extract_done_markers(dones)

    lengths = len(rewards)
    max_episode_steps = np.max(lengths)
    rewards = np.array(rewards).reshape(-1, 1)

    reverse_reward_to_go = np.inf * np.ones_like(rewards)
    running_reward = 0
    for i, (reward) in enumerate(rewards[::-1]):
        if i == lengths - 1:
            running_reward = 0
        running_reward += reward
        reverse_reward_to_go[i] = running_reward
    cum_reward_to_go = reverse_reward_to_go[::-1].copy()

    avg_reward_to_go = np.inf * np.ones_like(cum_reward_to_go)

    # elapsed_time = 0
    # for i, (cum_reward, done) in enumerate(zip(cum_reward_to_go, dones)):
    #     avg_reward_to_go[i] = cum_reward / (max_episode_steps - elapsed_time)
    #     elapsed_time += 1
    #     if done:
    #         elapsed_time = 0

    return avg_reward_to_go if average else cum_reward_to_go


# @torch.no_grad()
def denoising_sample(
    genplan_model,
    env_name,
    plan_horizon,
    x,
    rtgs,
    timesteps=None,
    insts=None,
    full_obs=None,
    logger=None,
    sample_iteration=1,
    env_size=None,
    ema_net=None,
    is_loop=False,
    stuck_action=-1,
):
    rate = 3  # genplan_model.rate
    batch_size = x.shape[0]
    context = 50  # block_size // rate

    cur_timestep = timesteps.cpu().numpy()[0, 0, 0]
    if env_name == "BabyAI-PickupLoc-v0":
        horizon = random.randint(1, plan_horizon)
    else:
        horizon = plan_horizon
    cur_timestep += horizon
    timesteps = (cur_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to("cuda")

    init_states = torch.clone(x[:, -1]).unsqueeze(1)

    goals = torch.clone(x[:, -1, -4:]).cpu().to(dtype=torch.float32)
    goals = (
        torch.repeat_interleave(torch.Tensor(goals).unsqueeze(1), horizon - 1, dim=1)
        .to(dtype=torch.long)
        .to("cuda")
    )

    sample_states = [[0, 0, 0] for _ in range(horizon - 1)]
    sample_states = torch.repeat_interleave(
        torch.Tensor(sample_states).unsqueeze(0), batch_size, dim=0
    ).to("cuda")

    if horizon > 1:
        sample_states = torch.cat((sample_states, goals), dim=2)
        sample_states = torch.cat((init_states, sample_states), dim=1).to(
            dtype=torch.long
        )
    elif horizon == 1:
        sample_states = init_states

    init_obss = torch.clone(full_obs[:, -1]).cpu()
    sample_obss = (
        torch.repeat_interleave(torch.Tensor(init_obss).unsqueeze(1), horizon, dim=1)
        .to(dtype=torch.float32)
        .to("cuda")
    )

    sample_actions = [[token2idx("<-MASK->")] for i in range(horizon)]

    sample_actions = (
        torch.repeat_interleave(
            torch.Tensor(sample_actions).unsqueeze(0), batch_size, dim=0
        )
        .to(dtype=torch.long)
        .to("cuda")
    )

    rtgs = rtgs[:, -1, :]
    sample_rtgs = (
        torch.repeat_interleave(rtgs, horizon, dim=1).to(dtype=torch.float).to("cuda")
    )

    if is_loop:
        sample_states = x
        sample_obss = full_obs
    for i in range(sample_iteration):
        if i == 0:
            action_masks = np.ones((batch_size, horizon, 1)).astype(boolean)
            action_masks = torch.from_numpy(action_masks).to("cuda")
        else:
            action_masks = np.zeros((batch_size, horizon, 1)).astype(boolean)

        sample_actions[action_masks] = token2idx("<-MASK->")

        states_stats = {"min": 0, "max": env_size}

        action_logits, state, way1, way2 = genplan_model.rollout(
            sample_states,
            insts=insts,
            target_imgs=sample_obss,
            mode="eval",
            is_loop=is_loop,
            logger=logger,
        )

    return action_logits, state, way1, way2
