"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import math
import logging
from tqdm import tqdm
import numpy as np
import pdb
import torch
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict, deque
import inspect
from mingpt.utils import denoising_sample, AGENT_ID, AGENT_COLOR
from collections import deque
import random
import cv2
import torch
from PIL import Image
import logging
from gym.wrappers import RecordVideo
from minigrid.utils.baby_ai_bot import BabyAIBot
import pandas as pd

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm


def configure_optimizers(model, weight_decay: float = 1e-3, device_type="cuda"):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    return optim_groups, extra_args


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 0.50
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.ckpt_path = "actor_close.pkl"


class Trainer:

    def __init__(
        self,
        genplan,
        bert_train_dataset,
        config,
        env,
        test_env,
        env_name,
        rate,
        plan_horizon,
        sample_iteration,
        inst_preprocessor,
        env_size,
        logger,
        buffer,
        save_name,
    ):
        self.genplan = genplan
        self.bert_train_dataset = bert_train_dataset
        self.config = config
        self.env = env
        self.test_env = test_env
        self.env_name = env_name
        self.rate = rate
        self.plan_horizon = plan_horizon
        self.sample_iteration = sample_iteration
        self.inst_preprocessor = inst_preprocessor
        self.env_size = env_size
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.genplan = self.genplan.to(self.device)

        self.logger = logger
        self.buffer = buffer
        self.device = genplan.device
        self.ema = True
        self.save_name = save_name

    def save_checkpoint(self, epoch=0):
        directory = os.path.dirname(self.config.ckpt_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        print(f"saving {self.config.ckpt_path}")

        checkpoint = {
            "train_step": epoch,
            "genplan": self.genplan.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.genplan.stochastic_policy:
            extra_string = "_stochastic"
        else:
            extra_string = ""
        torch.save(
            checkpoint,
            "ckpts_final/"
            + self.save_name
            + "_"
            + self.env_name
            + "_"
            + str(epoch)
            + "_"
            + str(self.plan_horizon)
            + extra_string
            + "_"
            + self.config.ckpt_path,
        )

    def load_checkpoint(self, epoch=0):
        print(f"loading {self.config.ckpt_path}")
        if self.genplan.stochastic_policy:
            extra_string = "_stochastic"
        else:
            extra_string = ""

        # Load the checkpoint
        checkpoint = torch.load(
            "ckpts_final/"
            + self.save_name
            + "_"
            + self.env_name
            + "_"
            + str(epoch)
            + "_"
            + str(self.plan_horizon)
            + extra_string
            + "_"
            + self.config.ckpt_path,
            map_location=self.device,
        )

        # Load the state dictionaries from the checkpoint
        self.genplan.load_state_dict(checkpoint["genplan"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"Checkpoint loaded from epoch {epoch}")

    def train(self):
        num_epochs = self.config.max_epochs
        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        if self.ema:
            self.ema_genplan = EMAModel(
                parameters=self.genplan.parameters(), power=0.75
            )

        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        # params = list(filter(lambda p: p.requires_grad, self.genplan.parameters()))
        params, extra_args = configure_optimizers(self.genplan, weight_decay=8e-3)
        self.optimizer = torch.optim.AdamW(
            params,
            self.genplan.config.extra_config.learning_rate,  # 0.001,  # 0.00008,
            betas=(
                self.genplan.config.extra_config.adam_beta1,
                self.genplan.config.extra_config.adam_beta2,
            ),
            eps=self.genplan.config.extra_config.adam_eps,
            **extra_args,
        )

        # Cosine LR schedule with linear warmup

        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=1,
            num_training_steps=np.ceil(
                len(self.bert_train_dataset) / self.config.batch_size
            )
            * num_epochs,
        )
        if self.genplan.stochastic_policy:
            log_temperature_optimizer = torch.optim.Adam(
                [self.genplan.log_temperature],
                lr=1e-2,  # 1e-3,#
                betas=[0.9, 0.999],
            )

        def run_epoch(split, epoch_num=0):
            is_train = split == "train"
            self.genplan.train(is_train)
            bert_data = self.bert_train_dataset if is_train else None
            bert_loader = DataLoader(
                bert_data,
                shuffle=True,
                pin_memory=True,
                batch_size=self.config.batch_size,
                drop_last=True,
                num_workers=0,
            )  # config.num_workers
            genplan_losses = []

            for it, (
                x,
                y,
                y_one_hot,
                m_y,
                full_imgs,
                msk_x,
                msk_y,
                r_only,
                r,
                t,
                inst,
                init_x,
                init_image,
                dones,
            ) in enumerate(bert_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                m_y = m_y.to(self.device)
                full_imgs = full_imgs.to(self.device)
                msk_x = msk_x.to(self.device)
                msk_y = msk_y.to(self.device)
                r_only = r_only.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                inst = inst.to(self.device)
                dones = dones.to(self.device)

                with torch.autograd.set_detect_anomaly(True):
                    with torch.set_grad_enabled(is_train):
                        genplan_loss, actions_pred, loss_dict = self.genplan(
                            x, y, full_imgs, insts=inst, pad_positions=msk_y
                        )
                        if genplan_loss == 0:
                            continue
                        genplan_loss = genplan_loss.mean()
                        genplan_losses.append(genplan_loss.item())
                        self.logger.log(loss_dict)

                        self.optimizer.zero_grad()

                        genplan_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.genplan.parameters(), max_norm=1.0
                        )
                        self.optimizer.step()

                    if self.genplan.stochastic_policy:
                        log_temperature_optimizer.zero_grad()
                        temp_loss = (
                            self.genplan.temperature()
                            * (
                                self.genplan.entropy.mean()
                                - self.genplan.target_entropy
                            ).detach()
                        )
                        temp_loss.backward()
                        log_temperature_optimizer.step()
                    self.lr_scheduler.step()
                    if self.ema:
                        self.ema_genplan.step(self.genplan.parameters())

                    # report progress
                    print(
                        f"epoch {epoch+1} iter {it}: genplan loss {genplan_loss.item():.5f}, lr: {self.optimizer.param_groups[0]['lr']}."
                    )

        self.tokens = 0  # counter used for learning rate decay
        ########### Train #####################
        if self.ema:
            self.ema_nets_genplan = self.genplan
            self.ema_genplan.copy_to(self.ema_nets_genplan.parameters())

        for epoch in range(self.config.max_epochs):  # config.max_epochs
            run_epoch("train", epoch_num=epoch)
            self.ema_nets_genplan = self.genplan
            if self.ema:
                self.ema_genplan.copy_to(self.ema_nets_genplan.parameters())
            print(f"epoch {epoch}")
            # if epoch % 100 == 0 and epoch > 0:
            #     self.save_checkpoint(epoch)

            if epoch % 100 == 0 and epoch > 0:
                success_rate = self.test_returns(
                    0, self.env_name, self.plan_horizon, test_num=10, final=False
                )

        # self.save_checkpoint(epoch)
        self.test_returns(
            0, self.env_name, self.plan_horizon, test_num=250, final=False
        )

    def test_returns(self, ret, env_name, plan_horizon, test_num=40, final=False):
        self.genplan.train(False)
        env = self.test_env  # self.env
        initial_seed = 1234
        T_rewards = []
        success_count = 0
        fail_count = 0
        if final: # optional - default is False
            pos_change = 5 # this just initializes the agent in different positions in the same map
        else:
            pos_change = 1
        success_per_map = []
        skip_bot = True
        number_of_steps = []
        for test in range(test_num):
            map_success = 0
            for pos in range(pos_change):
                obs, _ = env.reset(seed=initial_seed + test)
                if pos_change > 1:
                    env.agent_pos = env.place_agent(i=0, j=0)
                print("mission:", obs["mission"])
                print(f"agent pos {env.agent_pos}")

                print(f"test {test}")
                goal = [0, 0, 0, 0]
                if not skip_bot:
                    self.bot_advisor_agent = BabyAIBot(env)
                    goal = self.bot_advisor_agent.get_all_goal_state()
                    if len(goal) != 4:
                        goal = goal[:2] + goal[-2:]
                        if len(goal) != 4:
                            pdb.set_trace()

                reward_sum = 500
                done = False

                full_obs = (
                    env.gen_full_obs()
                )  # obs["image"]  # env.gen_full_obs() #obs['image'] #

                full_obs = (
                    torch.from_numpy(full_obs)
                    .to(dtype=torch.float32)
                    .to(self.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                insts = (
                    torch.tensor(
                        self.inst_preprocessor(obs["mission"]), dtype=torch.long
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                state = list(env.agent_pos) + [obs["direction"]] + goal
                cur_state = list(env.agent_pos)

                state = (
                    torch.Tensor(state)
                    .type(torch.long)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )
                rtg = (
                    torch.Tensor([reward_sum])
                    .type(torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )
                sample_actions, states_pred, way1, way2 = denoising_sample(
                    self.genplan,
                    env_name,
                    plan_horizon,
                    state,
                    insts=insts,
                    full_obs=full_obs,
                    rtgs=rtg,
                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
                    logger=self.logger,
                    sample_iteration=self.sample_iteration,
                    ema_net=self.ema_nets_genplan,
                    env_size=self.env_size,
                )

                all_states = state
                all_full_obs = full_obs
                all_rtgs = rtg
                actions = []
                j = 0
                while True:
                    stuck_action = -1
                    # note we unroll action_horizon steps
                    for action in sample_actions:  # [:1]
                        # action = stochastic_action_selection(action)
                        obs, reward, terminated, truncated, info = env.step(action)
                        actions += [action]
                        reward_sum -= 1  # reward
                        # if done or j > max_steps:
                        if terminated or truncated:
                            break
                        j = len(actions)
                        cur_state = list(
                            env.agent_pos
                        )  # env.gen_agent_pos() # list(env.agent_pos)

                        state = cur_state + [obs["direction"]] + goal
                        state = (
                            torch.Tensor(state)
                            .type(torch.long)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        full_obs = (
                            env.gen_full_obs()
                        )  # obs["image"]  # env.gen_full_obs() #obs['image'] #

                        full_obs = (
                            torch.from_numpy(full_obs)
                            .to(dtype=torch.float32)
                            .to(self.device)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        all_states = torch.cat([all_states, state], dim=1)
                        rtg = (
                            torch.Tensor([reward_sum])
                            .type(torch.float32)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        all_rtgs = torch.cat([all_rtgs, rtg], dim=1)
                        all_full_obs = torch.cat([all_full_obs, full_obs], dim=1)
                    if truncated:
                        T_rewards.append(reward_sum)
                        print(f"This round failed {test}")
                        number_of_steps.append(j)
                        fail_count += 1
                        break
                    elif terminated:
                        success_count += 1
                        map_success += 1
                        print(f"This round is a success {test}")
                        number_of_steps.append(j)
                        break
                    sample_actions, states_pred, way1, way2 = denoising_sample(
                        self.genplan,
                        env_name,
                        plan_horizon,
                        all_states[:, -plan_horizon:, :],
                        all_rtgs[:, -plan_horizon:, :],
                        insts=insts,
                        full_obs=all_full_obs[:, -plan_horizon:, :],
                        logger=self.logger,
                        timesteps=(
                            min(j, 512)
                            * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)
                        ),
                        sample_iteration=self.sample_iteration,
                        ema_net=self.ema_nets_genplan,
                        env_size=self.env_size,
                        is_loop=False,
                    )
            success_per_map.append(map_success)
        env.close()
        test_numbers = list(range(1, test_num + 1))
        # data = {
        #     "Test Number": test_numbers,
        #     "Number of Successes": success_per_map,
        #     "Steps": number_of_steps,
        # }
        # dat = {
        #     "Test Number": test_numbers,
        #     "Number of Successes": success_per_map,
        # }
        # df = pd.DataFrame(
        #     {
        #         "Test Number": [data["Test Number"][0]] * len(data["Steps"]),
        #         "Number of Successes": [data["Number of Successes"][0]]
        #         * len(data["Steps"]),
        #         "Steps": data["Steps"],
        #     }
        # )
        # dff = pd.DataFrame(dat)
        # # # Save the DataFrame to a CSV file
        # csv_file_path = f"./successes_per_map_{env_name}_{initial_seed}.csv"
        # csv_file_path_1 = (
        #     f"./successes_per_map_{env_name}_{initial_seed}_succ.csv"
        # )

        # df.to_csv(csv_file_path, index=False)
        # dff.to_csv(csv_file_path_1, index=False)

        test_num = test_num * pos_change
        eval_return = sum(T_rewards) / float(test_num)
        success_rate = success_count / float(test_num)
        msg = f"eval return: {eval_return}, success_rate: {success_rate:.3f}, succes_count: {success_count}, failcount: {fail_count} test_num: {test_num}, {self.config.ckpt_path}"
        print(msg)
        self.logger.log(
            {
                "eval_return": eval_return,
                "success_rate": success_rate,
                "succes_count": success_count,
                "test_num": test_num,
            }
        )
        self.genplan.train(True)
        return success_rate

    # some oracle functions (optional)
    def goal_selection(self, goals, agent):
        min_dist = 100
        min_idx = -1
        for i, goal in enumerate(goals):
            dist = np.sqrt((agent[0] - goal[0]) ** 2 + (agent[1] - goal[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return list(goals[min_idx])


#get plot with bg
def plot_random_colored_heatmap_with_background(
    background, points, goal, waypoints, curr_state, num, num_1, env_size
):
    plt.clf()  # Clear the current figure

    # Plot the background image of the maze
    plt.imshow(background)

    # Assuming the background is 320x320 and env_size is the grid size,
    # Calculate the scale to map grid coordinates to image pixels
    pixel_per_cell = background.shape[0] / env_size

    # Convert environment coordinates to pixel coordinates
    def to_pixel(coord):
        return coord * pixel_per_cell

    # Convert points, goal, waypoints, and curr_state to pixel coordinates
    points_pixel = [(to_pixel(x), to_pixel(y)) for x, y in points]
    goal_pixel = [to_pixel(g) for g in goal]
    mask = [0 for _ in goal]
    mask_pixel = [to_pixel(g) for g in mask]

    waypoints_pixel = [(to_pixel(wx), to_pixel(wy)) for wx, wy in waypoints]
    curr_state_pixel = (to_pixel(curr_state[0]), to_pixel(curr_state[1]))

    # Plot agent's current position in yellow
    # plt.scatter(curr_state_pixel[0], curr_state_pixel[1], color='yellow', s=50, label='Agent Position', zorder=5)
    for px, py in waypoints_pixel:
        plt.scatter(px, py, color="teal", s=100, marker="*", label="Goals", zorder=5)
    # Plot predicted goals in blue
    for px, py in points_pixel:
        plt.scatter(
            px, py, color="blue", s=100, marker="*", label="Waypoints", zorder=5
        )

    plt.scatter(
        goal_pixel[:2],
        goal_pixel[2:],
        color="yellow",
        s=100,
        marker="*",
        label="True Goal",
        zorder=5,
    )

    plt.scatter(
        mask_pixel[:2],
        mask_pixel[2:],
        color="red",
        s=100,
        marker="*",
        label="Mask Position",
        zorder=5,
    )

    # Legend and other settings as before
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.title(f"Goal Generation")
    # if num_1 == 10:
    #     plt.colorbar()

    # Adjust axes to match the background image size
    plt.xlim(0, background.shape[1])
    plt.ylim(background.shape[0], 0)  # Inverted to match the image's origin (top-left)

    plt.axis("off")  # Optionally turn off the axis
    plt.savefig(
        f"some_plots_s7/overlay_waypoint_{num}_{num_1}.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def stochastic_action_selection(action_token): # to emulate stochastic wrapper for env 
    fail_chance = 0.2  # 20% chance to fail executing turn actions
    if action_token in [0, 1] and random.random() < fail_chance:
        # List of alternative actions excluding the original action
        alternative_actions = [i for i in range(6) if i != action_token]
        # Randomly select an alternative action
        action_token = random.choice(alternative_actions)
    return action_token


def plot_grad_flow(named_parameters, step, nam="int"):
    ave_grads = []
    layers = []
    norms = []

    # named_parameters = [n for n, p in list(model.named_parameters())]

    for name, param in named_parameters:
        if (param.requires_grad) and ("bias" not in name):
            layers.append(name)
            # print(n)

            if param.grad is not None:
                # ave_grads.append(param.grad.abs().mean().cpu().detach().numpy())
                norms.append(param.grad.norm().cpu().detach().numpy())
            else:
                # ave_grads.append(0.0)
                norms.append(0.0)
    # if step>20:
    #     breakpoint()
    plt.rcParams["figure.figsize"] = (20, 5)

    plt.plot(norms, alpha=0.3, color="r")
    # plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(norms) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(norms), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(norms))
    # plt.ylim(ymin=-0.01, ymax = 0.01)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f"./grad_norms/{nam}_{step}_grad.png", bbox_inches="tight")
