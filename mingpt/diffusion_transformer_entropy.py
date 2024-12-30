from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np
from mingpt.genplan_utils import *
import pdb
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mingpt.transformer import TransformerForDiffusion

END, PAD, MASK = "<-END->", "<-PAD->", "<-MASK->"
RIGHT = "<-RIGHT->"  # [1,0]
LEFT = "<-LEFT->"  # [-1,0]
UP = "<-UP->"  # [0,-1]
DOWN = "<-DOWN->"  # [0,1]
MOVE = "<-MOVE->"
PICKUP = "<-PICKUP->"
DROP = "<-DROP->"
TOGGLE = "<-TOGGLE->"
DONE = "<-DONE->"
# tokens = [LEFT, RIGHT, MOVE, PICKUP, DROP, TOGGLE, PAD, MASK] #END,
tokens = [LEFT, RIGHT, MOVE, PICKUP, DROP, TOGGLE, MASK]  # END,

_token2idx = dict(zip(tokens, range(len(tokens))))

def generate_random_tensor(shape):
    values = torch.tensor([0.0, 1.0])
    tensor = values[torch.randint(0, len(values), shape)]
    return tensor


# Function to apply the mapping using vectorized operations
def map_values(tensor):
    # Create a tensor to hold the mapped values initially the same as input tensor

    tensor_ = (
        tensor.unique()
    )  # torch.tensor([-1., 0., 2., 4., 5., 6., 10.], device=tensor.device)

    # Define the mapping from original values to class indices
    unique_values = tensor_.unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}

    # Print the mapping to see the conversion
    # print("Mapping:", mapping)
    mapped_tensor = torch.clone(tensor).to(
        torch.long
    )  # Ensure it's a long type for indices
    for old_value, new_value in mapping.items():
        mapped_tensor = torch.where(
            tensor == old_value,
            torch.tensor(new_value, device=tensor.device),
            mapped_tensor,
        )

    # Create a randomized 2D output tensor
    random_mapping = torch.randperm(len(unique_values), device=tensor.device)
    randomized_tensor = torch.clone(tensor).to(
        torch.long
    )  # Ensure it's a long type for indices
    for old_value, random_idx in zip(unique_values, random_mapping):
        randomized_tensor = torch.where(
            tensor == old_value, random_idx, randomized_tensor
        )

    return mapped_tensor, randomized_tensor


def compute_probabilities_image(
    x1, xt, dt, t, num_classes, noise_type="uniform", mask_token=None
):
    # Compute delta p values

    try:
        assert xt.max() < num_classes
        assert x1.max() < num_classes
    except AssertionError:
        xt = torch.where(xt >= num_classes, torch.tensor(0, device=xt.device), xt)

    dt_p_vals = dt_p_xt_g_xt(
        x1, t, num_classes, noise_type=noise_type, mask_token=mask_token
    )  # (B, D, S)
    dt_p_vals_at_xt = dt_p_vals.gather(-1, xt[:, :, :, :, None]).squeeze(-1)  # (B, D)

    # Numerator of R_t^*
    R_t_numer = F.relu(dt_p_vals - dt_p_vals_at_xt[:, :, :, :, None])  # (B, D, S)

    pt_vals = p_xt_g_x1(
        x1, t, num_classes, noise_type=noise_type, mask_token=mask_token
    )  # (B, D, S)
    Z_t = torch.count_nonzero(pt_vals, dim=-1)  # (B, D)
    pt_vals_at_xt = pt_vals.gather(-1, xt[:, :, :, :, None]).squeeze(-1)  # (B, D)

    # Denominator of R_t^*
    R_t_denom = Z_t * pt_vals_at_xt  # (B, D)

    R_t = R_t_numer / R_t_denom[:, :, :, :, None]  # (B, D, S)

    # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero
    R_t[(pt_vals_at_xt == 0.0)[:, :, :, :, None].repeat(1, 1, 1, 1, num_classes)] = 0.0
    R_t[pt_vals == 0.0] = 0.0

    # Calculate the off-diagonal step probabilities
    step_probs = (R_t * dt).clamp(max=1.0)  # (B, D, S)

    # Calculate the on-diagonal step probabilities
    # 1) Zero out the diagonal entries
    step_probs.scatter_(-1, xt[:, :, :, :, None], 0.0)
    # 2) Calculate the diagonal entries such that the probability row sums to 1
    step_probs.scatter_(
        -1,
        xt[:, :, :, :, None],
        (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0),
    )

    xt = Categorical(step_probs).sample()  # (B, D)
    return xt


def compute_probabilities(
    x1, xt, dt, t, num_classes, noise_type="uniform", mask_token=None
):
    # Compute delta p values

    dt_p_vals = dt_p_xt_g_xt(
        x1, t, num_classes, noise_type=noise_type, mask_token=mask_token
    )  # (B, D, S)
    dt_p_vals_at_xt = dt_p_vals.gather(-1, xt[:, :, None]).squeeze(-1)  # (B, D)

    # Numerator of R_t^*
    R_t_numer = F.relu(dt_p_vals - dt_p_vals_at_xt[:, :, None])  # (B, D, S)

    pt_vals = p_xt_g_x1(
        x1, t, num_classes, noise_type=noise_type, mask_token=mask_token
    )  # (B, D, S)
    Z_t = torch.count_nonzero(pt_vals, dim=-1)  # (B, D)
    pt_vals_at_xt = pt_vals.gather(-1, xt[:, :, None]).squeeze(-1)  # (B, D)

    # Denominator of R_t^*
    R_t_denom = Z_t * pt_vals_at_xt  # (B, D)

    R_t = R_t_numer / R_t_denom[:, :, None]  # (B, D, S)

    # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero
    R_t[(pt_vals_at_xt == 0.0)[:, :, None].repeat(1, 1, num_classes)] = 0.0
    R_t[pt_vals == 0.0] = 0.0

    # Calculate the off-diagonal step probabilities
    step_probs = (R_t * dt).clamp(max=1.0)  # (B, D, S)

    # Calculate the on-diagonal step probabilities
    # 1) Zero out the diagonal entries
    step_probs.scatter_(-1, xt[:, :, None], 0.0)
    # 2) Calculate the diagonal entries such that the probability row sums to 1
    step_probs.scatter_(
        -1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    )

    xt = Categorical(step_probs).sample()  # (B, D)
    return xt


def dt_p_xt_g_xt(x1, t, number_classe, noise_type="uniform", mask_token=None):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value

    # uniform
    if noise_type == "uniform":
        x1_onehot = F.one_hot(x1, num_classes=number_classe)  # (B, D, S)
        return x1_onehot - (1 / number_classe)

    # masking
    elif noise_type == "masking":
        x1_onehot = F.one_hot(x1, num_classes=number_classe)  # (B, D, S)
        M_onehot = F.one_hot(torch.tensor([mask_token]), num_classes=number_classe)[
            None, :, :
        ].to(
            x1_onehot.device
        )  # (1, 1, S)

        return x1_onehot - M_onehot


def p_xt_g_x1(x1, t, number_classe, noise_type="uniform", mask_token=None):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value

    # uniform
    if noise_type == "uniform":
        x1_onehot = F.one_hot(x1, num_classes=number_classe)  # (B, D, S)
        return t * x1_onehot + (1 - t) * (1 / number_classe)

    # masking
    elif noise_type == "masking":
        x1_onehot = F.one_hot(x1, num_classes=number_classe)  # (B, D, S)
        M_onehot = F.one_hot(torch.tensor([mask_token]), num_classes=number_classe)[
            None, :, :
        ].to(
            x1_onehot.device
        )  # (1, 1, S)
        return t * x1_onehot + (1 - t) * M_onehot


def optimized_grid_decoder_state(input_tensor, size_env):
    # Get the indices of the max values in the last dimension (predicted class labels)
    # _, indices = torch.max(input_tensor, -1)  # This gives us the flattened grid indices
    indices = input_tensor

    # Convert flattened indices to x, y coordinates in a vectorized manner
    x = indices % size_env
    y = torch.div(indices, size_env, rounding_mode="floor")

    # Stack x and y coordinates along the last dimension
    output_tensor = torch.stack((x, y), dim=-1)

    return output_tensor


def optimized_grid_encoder_state(input_tensor, size_env):
    batch_size, seq_length, _ = input_tensor.shape
    flat_size = size_env * size_env

    # Reshape input_tensor to ensure it's in long format for indexing
    input_tensor = input_tensor.long()

    # Calculate flat indices for each (x, y) pair
    indices = input_tensor[..., 1] * size_env + input_tensor[..., 0]

    # Initialize the output tensor
    output_tensor = torch.zeros(
        batch_size, seq_length, flat_size, device=input_tensor.device
    )

    # Create a range tensor for batch indexing
    batch_indices = (
        torch.arange(batch_size).view(-1, 1).expand(-1, seq_length).reshape(-1)
    )
    seq_indices = torch.arange(seq_length).repeat(batch_size)

    # Use indices to set values to 1
    output_tensor[batch_indices, seq_indices, indices.view(-1)] = 1

    return output_tensor.view(batch_size, seq_length, -1)


def token2idx(token):
    return _token2idx[token]


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    bootstrap = False
    horizon = 100
    extra_config = None
    env = None

    def __init__(self, vocab_size, block_size, noise, action_horizon, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.noise = noise
        self.action_horizon = action_horizon
        for k, v in kwargs.items():
            setattr(self, k, v)


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, num_channels=x.num_features
        ),
    )
    return root_module




class SimpleResidualBlock(nn.Module):
    def __init__(self, input_channel_size, out_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channel_size,
            out_channel_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channel_size,
            out_channel_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel_size)
        if stride == 1:
            if input_channel_size == out_channel_size:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(
                    input_channel_size, out_channel_size, kernel_size=1, stride=stride
                )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channel_size, out_channel_size, kernel_size=1, stride=stride
                ),
                nn.Conv2d(
                    out_channel_size, out_channel_size, kernel_size=1, stride=stride
                ),
            )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out = self.relu2(out + shortcut)
        return out


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=imm_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels,
            out_channels=out_features,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(
            2
        ).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class GENPLAN_GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        pred_horizon = self.config.horizon
        action_dim = 8
        self.device = torch.device("cuda")
        self.stochastic_policy = True

        if self.stochastic_policy:
            self.log_temperature = torch.tensor(np.log(0.01))
            self.log_temperature.requires_grad = True
            self.target_entropy = 0.3
        # create network object
        if config.noise == "uniform":
            self.noise_pred_net = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim - 1,
            output_states_dim=config.env_size * config.env_size+1,
            horizon=pred_horizon,
            n_obs_steps=pred_horizon,
            cond_dim=config.n_embd,
            )
        else:
            self.noise_pred_net = TransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim - 1,
                output_states_dim=config.env_size * config.env_size + 1,
                horizon=pred_horizon,
                n_obs_steps=pred_horizon,
                cond_dim=config.n_embd,
                causal_attn=False,
            )
        

        self.gaussian = False # this is only for testing we don;t parametrize as gaussian dist (see categorical)
        self.aux_loss_module = None # only for devel
        self.action_horizon = config.action_horizon # how many actions we want to unroll from current step

        if self.config.extra_config.exploration_loss == "entropy":
            self.exploration_loss_func = self.entropy_exploration_loss
        elif self.config.extra_config.exploration_loss == "symmetric_kl":
            self.exploration_loss_func = self.symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(
                f"{self.config.extra_config.exploration_loss} not supported!"
            )

        self.action_dim = action_dim
        self.prediction_horizon = pred_horizon
        self.loss_crossent = nn.CrossEntropyLoss()
        self.instr_rnn = nn.GRU( # this is to process the dict on the words of the instruction
            self.config.n_embd, self.config.n_embd, batch_first=True
        )
        self.word_embedding = nn.Embedding(100, self.config.n_embd)
        self.state_embeddings = nn.Sequential( # we discretize the state space to learn better 
            nn.Embedding(
                self.config.env_size * self.config.env_size + 2, self.config.n_embd
            )
        )
        self.direction_embeddings = nn.Sequential(nn.Embedding(5, self.config.n_embd))
        self.state_embeddings_linear = nn.Sequential(
            nn.Linear(self.config.n_embd * 4, self.config.n_embd * 2),
            nn.Tanh(),
            nn.Linear(self.config.n_embd * 2, self.config.n_embd * 2),
            nn.Tanh(),
            nn.Linear(self.config.n_embd * 2, self.config.n_embd),
        )

        if self.config.env_size in [16, 19, 25]: # if you try a new env make sure to adjust here !
            self.state_encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=config.n_embd,
                    kernel_size=(2, 2),
                    padding=1,
                ),
                nn.BatchNorm2d(config.n_embd),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                SimpleResidualBlock(config.n_embd, config.n_embd, 2),
            )
        elif self.config.env_size in [7, 8, 9, 10, 13]:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=config.n_embd,
                    kernel_size=(2, 2),
                    padding=1,
                ),
                nn.BatchNorm2d(config.n_embd),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(
                    in_channels=config.n_embd,
                    out_channels=config.n_embd,
                    kernel_size=(3, 3),
                    padding=1,
                ),
                nn.BatchNorm2d(config.n_embd),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            )
        else:
            pdb.set_trace()

        self.controllers = []
        num_module = 2
        for ni in range(num_module):
            if ni < num_module - 1:
                mod = ExpertControllerFiLM(
                    in_features=self.config.n_embd,
                    out_features=config.n_embd,
                    in_channels=config.n_embd,
                    imm_channels=config.n_embd,
                )
            else:
                mod = ExpertControllerFiLM(
                    in_features=self.config.n_embd,
                    out_features=self.config.n_embd,
                    in_channels=config.n_embd,
                    imm_channels=config.n_embd,
                )
            self.controllers.append(mod)
            self.add_module("FiLM_Controler_" + str(ni), mod)
        self.agent_pos_controllers = (
            []
        )  # fuse the fully observable image with agent goal position and current position

        for ni in range(num_module):  #
            if ni < num_module - 1:
                mod = ExpertControllerFiLM(
                    in_features=self.config.n_embd,
                    out_features=config.n_embd,
                    in_channels=config.n_embd,
                    imm_channels=config.n_embd,
                )
            else:
                mod = ExpertControllerFiLM(
                    in_features=self.config.n_embd,
                    out_features=self.config.n_embd,
                    in_channels=config.n_embd,
                    imm_channels=config.n_embd,
                )
            self.agent_pos_controllers.append(mod)
            self.add_module("FiLM_Controler_" + str(ni + num_module), mod)

        if self.config.env_size in [25]:
            self.film_pool = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 4 * 4, self.config.n_embd), nn.ReLU()
            )
        elif self.config.env_size in [13, 19]:
            self.film_pool = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 3 * 3, self.config.n_embd), nn.ReLU()
            )
        elif self.config.env_size in [7, 8, 9, 10, 16]:
            self.film_pool = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 4, self.config.n_embd), nn.ReLU()
            )
        else:
            pdb.set_trace()

        self.state_encoder = replace_bn_with_gn(self.state_encoder) # stability acc to Chi. etal (Diffusion policy-cont.)

    def _get_instr_embedding(self, instr):
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def compute_energy(
        self,
        target_actions,
        action_logits,
        action_masks,
        use_sum=False,
        use_masks=False,
        use_neg_energy=False,
        pad_pos=None,
    ):
        # action_size = action_logits.shape[2]
        action_masks = action_masks.to(action_logits.device)
        action_masks = action_masks & pad_pos.squeeze(-1)
        if use_masks:
            action_logits = action_logits[action_masks]
            # log_softmax_action_logits = F.log_softmax(action_logits, dim=-1)
            pos_action_targets = target_actions[action_masks]
            pos_action_energy = -torch.gather(action_logits, -1, pos_action_targets)
        else:
            pos_action_energy = -torch.gather(action_logits, -1, target_actions)
        if use_sum:
            pos_energy = pos_action_energy.sum(-1)  # torch.sum(pos_action_energy)
        else:
            pos_energy = pos_action_energy
        energy = torch.clone(pos_energy)
        neg_energy = torch.Tensor([0])

        if use_sum:
            return energy  # , len(action_logits)
        else:
            return energy  # , pos_energy, -neg_energy

    def entropy_exploration_loss(
        self, action_distribution, exclude_last=False, pad_pos=None
    ):
        entropy = action_distribution.entropy()
        if exclude_last:
            entropy = (
                entropy.view(-1, self.config.extra_config.recurrence)[:, :-1]
                .contiguous()
                .view(-1)
            )

        if self.config.extra_config.loss_type == "mean":
            entropy_loss = -self.temperature().detach() * entropy.mean()
        elif self.config.extra_config.loss_type == "sum":
            entropy = entropy.reshape([-1, self.config.extra_config.recurrence])
            entropy = entropy.sum(dim=1)
            entropy_loss = -self.temperature().detach() * entropy.mean()
        elif self.config.extra_config.loss_type == "sum_ori":
            entropy_loss = -self.temperature().detach() * entropy.sum()
        else:
            raise NotImplementedError
        self.entropy = entropy
        return entropy_loss, entropy

    def symmetric_kl_exploration_loss(self, action_distribution): # alt for entropy
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = kl_prior.mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        kl_prior_loss = self.temperature().detach() * kl_prior
        return kl_prior_loss

    def _clip_grads(self, grads):
        norm_type = 2.0
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(grad.detach(), norm_type).to(self.device)
                    for grad in grads
                    if grad is not None
                ]
            ),
            norm_type,
        )
        clip_coef = self.config.extra_config.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                if grad is not None:
                    grad.detach().mul_(clip_coef.to(grad.device))

    def prepare_input(
        self,
        target_imgs,
        true_target_imgs,
        insts,
        target_states,
        true_target_states,
        nstate,
        nway1,
        nway2,
        ndir,
    ):
        target_imgs_ = (
            true_target_imgs.view(
                -1,
                target_imgs.shape[-3],
                target_imgs.shape[-2],
                target_imgs.shape[-1],
            )
            .permute(0, 3, 1, 2)
            .type(torch.float32)
            .contiguous()
        )

        image_embeddings = self.state_encoder(
            target_imgs_
        )  # (batch * block_size, n_embd)

        true_target_imgs = (
            true_target_imgs.view(
                -1, target_imgs.shape[-3], target_imgs.shape[-2], target_imgs.shape[-1]
            )
            .permute(0, 3, 1, 2)
            .type(torch.float32)
            .contiguous()
        )
        true_target_emb = self.state_encoder(true_target_imgs)

        instr_embedding = self._get_instr_embedding(insts)
        instr_embedding = torch.repeat_interleave(
            instr_embedding.unsqueeze(1), target_states.shape[1], dim=1
        )
        instr_embedding = instr_embedding.reshape(-1, instr_embedding.size(2))

        # jointly learn the image and instr
        for controler in self.controllers:
            image_embeddings_ = controler(image_embeddings, instr_embedding)
            true_target_emb = controler(true_target_emb, instr_embedding)

        state_embeddings = self.state_embeddings(nstate.unsqueeze(-1).long())
        goal_embeddings = self.state_embeddings(
            torch.cat((nway1.unsqueeze(-1), nway2.unsqueeze(-1)), dim=-1).long()
        )
        direction_embeddings = self.state_embeddings(ndir.unsqueeze(-1).long())
        state_embeddings = torch.cat(
            (state_embeddings, direction_embeddings, goal_embeddings), dim=2
        ).view(target_states.shape[0] * target_states.shape[1], -1)
        state_embeddings = self.state_embeddings_linear(state_embeddings)

        cat_target_states_ = optimized_grid_encoder_state(
            true_target_states[:, :, :2], self.config.env_size
        )
        cat_waypoint_1_ = optimized_grid_encoder_state(
            true_target_states[:, :, -2:], self.config.env_size
        )
        cat_waypoint_2_ = optimized_grid_encoder_state(
            true_target_states[:, :, -4:-2], self.config.env_size
        )
        true_state_embeddings = self.state_embeddings(
            cat_target_states_.argmax(-1).unsqueeze(-1).long()
        )
        true_goal_embeddings = self.state_embeddings(
            torch.cat(
                (
                    cat_waypoint_1_.argmax(-1).unsqueeze(-1),
                    cat_waypoint_2_.argmax(-1).unsqueeze(-1),
                ),
                dim=-1,
            ).long()
        )
        true_dir_embeddings = self.state_embeddings(
            true_target_states[:, :, 3].long().unsqueeze(-1)
        )
        true_state_embeddings = torch.cat(
            (true_state_embeddings, true_dir_embeddings, true_goal_embeddings), dim=2
        ).view(target_states.shape[0] * target_states.shape[1], -1)
        true_state_embeddings = self.state_embeddings_linear(true_state_embeddings)

        for controller in self.agent_pos_controllers:
            image_embeddings = controller(image_embeddings_, state_embeddings)
            true_target_emb = controller(true_target_emb, true_state_embeddings)

        state_embeddings = self.film_pool(image_embeddings)
        true_target_emb = self.film_pool(true_target_emb)

        state_embeddings = state_embeddings.view(
            target_states.shape[0], target_states.shape[1], self.config.n_embd
        )
        true_target_emb = true_target_emb.view(
            target_states.shape[0], target_states.shape[1], self.config.n_embd
        )

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = state_embeddings  # .unsqueeze(0).flatten(start_dim=1)
        return obs_cond, true_target_emb

    def rollout(
        self,
        target_states,
        insts,
        target_imgs,
        mode,
        is_loop,
        logger=None,
    ):
        # some batching stuff - can be a bit efficient here
        padding_needed = self.prediction_horizon - target_states.shape[1]
        if padding_needed > 0:
            padded_tensor = F.pad(
                target_states, (0, 0, 0, padding_needed), "constant", 0
            )
            target_states = padded_tensor
            pad_up = padding_needed // 2
            pad_down = padding_needed - pad_up
            padding = (0, 0, 0, 0, 0, 0, pad_down, pad_up)
            target_imgs = F.pad(target_imgs, padding, "constant", 0)
        cat_target_states = optimized_grid_encoder_state(
            target_states[:, :, :2], self.config.env_size
        )
        cat_waypoint_1 = optimized_grid_encoder_state(
            target_states[:, :, -2:], self.config.env_size
        )
        cat_waypoint_2 = optimized_grid_encoder_state(
            target_states[:, :, -4:-2], self.config.env_size
        )
        cat_target_states = torch.argmax(cat_target_states, dim=-1)
        cat_waypoint_1 = torch.argmax(cat_waypoint_1, dim=-1)
        cat_waypoint_2 = torch.argmax(cat_waypoint_2, dim=-1)
        true_target_states = torch.zeros_like(target_states)
        true_target_imgs = torch.zeros_like(target_imgs)
        if is_loop: # loop back to the beginning
            true_target_states[:,:self.action_horizon,:] = target_states[:,-self.action_horizon:,:]
            true_target_imgs[:,:self.action_horizon,:,:,:] = target_imgs[:, -self.action_horizon:, :, :, :]
        else:
            true_target_states[:,:self.action_horizon,:] = target_states[:,:self.action_horizon,:]
            true_target_imgs[:,:self.action_horizon,:,:,:] = target_imgs[:, :self.action_horizon, :, :, :]
        if self.config.noise == "masking":
            naction = token2idx("<-MASK->") * torch.ones(
                (target_states.shape[0], self.prediction_horizon),
                dtype=torch.long,
                device=target_states.device,
            )
            nstate = (self.config.env_size**2) * torch.ones(
                (target_states.shape[0], self.prediction_horizon),
                dtype=torch.long,
                device=target_states.device,
            )
            nway1 = (self.config.env_size**2) * torch.ones(
                (target_states.shape[0], self.prediction_horizon),
                dtype=torch.long,
                device=target_states.device,
            )
            nway2 = (self.config.env_size**2) * torch.ones(
                (target_states.shape[0], self.prediction_horizon),
                dtype=torch.long,
                device=target_states.device,
            )
            ndir = 4 * torch.ones(
                (target_states.shape[0], self.prediction_horizon),
                dtype=torch.long,
                device=target_states.device,
            )

            ndir[:, 0] = target_states[:, 0, 2]
            nstate[:, 0] = cat_target_states[:, 0]
            nway1[:, 0] = cat_waypoint_1[:, 0]
            nway2[:, 0] = cat_waypoint_2[:, 0]
            noise = 0.1
            t = 0.0
            dt = 1 / (self.prediction_horizon // 2)

        if self.config.noise == "uniform":
            naction = torch.randint(
                0,
                self.action_dim - 2,
                (target_states.shape[0], self.prediction_horizon),
                device=target_states.device,
            )
            nstate = torch.randint(
                0,
                self.config.env_size**2 - 1,
                (target_states.shape[0], self.prediction_horizon),
                device=target_states.device,
            )
            nway1 = torch.randint(
                0,
                self.config.env_size**2 - 1,
                (target_states.shape[0], self.prediction_horizon),
                device=target_states.device,
            )
            nway2 = torch.randint(
                0,
                self.config.env_size**2 - 1,
                (target_states.shape[0], self.prediction_horizon),
                device=target_states.device,
            )
            ndir = torch.randint(
                0,
                4,
                (target_states.shape[0], self.prediction_horizon),
                device=target_states.device,
            )
            ndir[:, 0] = target_states[:, 0, 2]
            nstate[:, 0] = cat_target_states[:, 0]
            nway1[:, 0] = cat_waypoint_1[:, 0]
            nway2[:, 0] = cat_waypoint_2[:, 0]
            noise = 0.1
            t = 0.0
            dt = 1 / (self.prediction_horizon // 2) # roughly want to flip 2 places each iter - choosen through rate mat/
        while True:
            obs_cond, true_target_emb = self.prepare_input(
                target_imgs,
                true_target_imgs,
                insts,
                target_states,
                true_target_states,
                nstate,
                nway1,
                nway2,
                ndir,
            )
            t = t * torch.ones((target_states.shape[0],), device=target_states.device)
            t = t.to(self.device)
            (
                noise_pred_action,
                noise_state_pred,
                noise_way_1,
                noise_way_2,
                noise_pred_dir,
            ) = self.noise_pred_net(
                sample=naction,
                timestep=t,
                cond=obs_cond,
                true_cond=true_target_emb,
            )
            action_dist_ = CategoricalActionDistribution(noise_pred_action)
            sample_actions, model_log_probs = sample_actions_log_probs(action_dist_)
            # energy = self.compute_energy(sample_actions, model_log_probs, torch.ones_like(action_dist_.sample_max(), dtype=torch.bool).to(action_dist_.sample_max().device), pad_pos=torch.ones_like(action_dist_.sample_max(), dtype=torch.bool).to(action_dist_.sample_max().device))
            energy = -torch.sum(model_log_probs)

            # x1_probs = F.softmax(noise_pr ed_action, dim=-1) # (B, D, S-1)

            x1_probs_state = F.softmax(noise_state_pred, dim=-1)  # (B, D, S-1)

            x1_probs_way_1 = F.softmax(noise_way_1, dim=-1)  # (B, D, S-1)

            x1_probs_way_2 = F.softmax(noise_way_2, dim=-1)  # (B, D, S-1)

            x1_probs_dir = F.softmax(noise_pred_dir, dim=-1)  # (B, D, S-1)

            # x1 = Categorical(x1_probs).sample() # (B, D)

            x1 = sample_actions  # action_dist.sample_max() # (B, D)
            x1_state = Categorical(x1_probs_state).sample()  # (B, D)
            x1_way_1 = Categorical(x1_probs_way_1).sample()  # (B, D)
            x1_way_2 = Categorical(x1_probs_way_2).sample()  # (B, D)
            x1_dir = Categorical(x1_probs_dir).sample()  # (B, D)

            if dt == 1:
                naction = x1
                break

            if self.config.noise == "masking":
                naction = compute_probabilities(
                    x1,
                    naction,
                    dt,
                    t,
                    self.action_dim - 1,
                    noise_type="masking",
                    mask_token=token2idx("<-MASK->"),
                )
                nstate = compute_probabilities(
                    x1_state,
                    nstate,
                    dt,
                    t,
                    self.config.env_size**2 + 1,
                    noise_type="masking",
                    mask_token=self.config.env_size**2,
                )
                nway1 = compute_probabilities(
                    x1_way_1,
                    nway1,
                    dt,
                    t,
                    self.config.env_size**2 + 1,
                    noise_type="masking",
                    mask_token=self.config.env_size**2,
                )
                nway2 = compute_probabilities(
                    x1_way_2,
                    nway2,
                    dt,
                    t,
                    self.config.env_size**2 + 1,
                    noise_type="masking",
                    mask_token=self.config.env_size**2,
                )
                ndir = compute_probabilities(
                    x1_dir, ndir, dt, t, 5, noise_type="masking", mask_token=4
                )
                t += dt
                if t >= 1.0:
                    break

            if self.config.noise == "uniform":
                naction = compute_probabilities(x1, naction, dt, t, self.action_dim - 1)
                nstate = compute_probabilities(
                    x1_state, nstate, dt, t, self.config.env_size**2 + 1
                )
                nway1 = compute_probabilities(
                    x1_way_1, nway1, dt, t, self.config.env_size**2 + 1
                )
                nway2 = compute_probabilities(
                    x1_way_2, nway2, dt, t, self.config.env_size**2 + 1
                )
                ndir = compute_probabilities(x1_dir, ndir, dt, t, 5)

                t += dt
                if t >= 1.0:
                    break

        self.save_naction = naction  # sample_actions#self.best_action_landscape  # x1
        naction = naction[0]

        nstate = optimized_grid_decoder_state(nstate, self.config.env_size)
        nway1 = optimized_grid_decoder_state(nway1, self.config.env_size)
        nway2 = optimized_grid_decoder_state(nway2, self.config.env_size)

        nstate = nstate.detach().to("cpu")  # .numpy()
        nway1 = nway1.detach().to("cpu")  # .numpy()
        nway2 = nway2.detach().to("cpu")  # .numpy()

        if is_loop:
            if not self.action_horizon == 1:
                action = naction.cpu().detach().numpy()[self.action_horizon - 1 :]
            else:
                action = naction.cpu().detach().numpy()  # [:]#[:5]  # [:5]#[49:]#[21:]
        else:
            action = naction.cpu().detach().numpy()  # [:5]#[49:]#[21:]
        state = nstate.numpy()
        way1 = nway1.numpy()
        way2 = nway2.numpy()
        return action, state[0], way1[0], way2[0]

    def generate_time(self, B, some_number):
        intervals = torch.linspace(0, 1, steps=some_number + 1)
        indices = torch.randint(0, len(intervals), (B,))
        result = intervals[indices]

        return result

    def forward(
        self,
        target_states,
        target_actions,
        target_imgs,
        insts=None,
        mode="train",
        pad_positions=None,
    ):

        ########## Initialization ##########
        true_cond = target_imgs.clone().detach()
        # these are used in cross attn.
        target_imgs[:, self.action_horizon :, :, :, :] = 0.0 # by design action_horizon <= horizon
        true_cond[:, self.action_horizon :, :, :, :] = 0.0

        B = target_states.shape[0]
        t = torch.rand((B,)) # can use generate time for uniform splits 
        mask_place = torch.rand((B, target_actions.shape[1])) < (1 - t[:, None])

        target_imgs_ = (
            target_imgs.view(
                -1, target_imgs.shape[-3], target_imgs.shape[-2], target_imgs.shape[-1]
            )
            .permute(0, 3, 1, 2)
            .type(torch.float32)
            .contiguous()
        )
        image_embeddings = self.state_encoder(
            target_imgs_
        )  # (batch * block_size, n_embd)

        true_cond = (
            true_cond.view(
                -1, target_imgs.shape[-3], target_imgs.shape[-2], target_imgs.shape[-1]
            )
            .permute(0, 3, 1, 2)
            .type(torch.float32)
            .contiguous()
        )
        true_cond = self.state_encoder(true_cond)

        instr_embedding = self._get_instr_embedding(insts)
        instr_embedding = torch.repeat_interleave(
            instr_embedding.unsqueeze(1), target_states.shape[1], dim=1
        )
        instr_embedding = instr_embedding.reshape(-1, instr_embedding.size(2))

        cat_target_states = optimized_grid_encoder_state(
            target_states[:, :, :2], self.config.env_size
        )
        cat_waypoint_1 = optimized_grid_encoder_state(
            target_states[:, :, -2:], self.config.env_size
        )
        cat_waypoint_2 = optimized_grid_encoder_state(
            target_states[:, :, -4:-2], self.config.env_size
        )

        cat_target_states = torch.argmax(cat_target_states, dim=-1)
        cat_waypoint_1 = torch.argmax(cat_waypoint_1, dim=-1)
        cat_waypoint_2 = torch.argmax(cat_waypoint_2, dim=-1)

        noisy_actions = target_actions.squeeze(-1).clone()
        noisy_states = cat_target_states.clone()
        noisy_waypoint_1 = torch.zeros_like(cat_waypoint_1)  # cat_waypoint_1.clone()
        noisy_waypoint_2 = torch.zeros_like(cat_waypoint_2)  # cat_waypoint_2.clone()
        dir_gt = target_states[:, :, 2].long().clone().detach()

        #### True embeddings ######
        if self.config.noise == "uniform":
            uniform_noise_actions = torch.randint(
                0,
                self.action_dim - 2,
                (B, target_actions.shape[1]),
                device=target_actions.device,
            )
            noisy_actions[mask_place] = uniform_noise_actions[mask_place]
            uniform_noise_states = torch.randint(
                0,
                self.config.env_size**2 - 1,
                (B, target_actions.shape[1]),
                device=target_actions.device,
            )
            noisy_states[mask_place] = uniform_noise_states[mask_place]
            noisy_waypoint_1[mask_place] = uniform_noise_states[mask_place]
            noisy_waypoint_2[mask_place] = uniform_noise_states[mask_place]
            noisy_dir = target_states[:, :, 2].long().clone().detach()
            noisy_dir_ = torch.randint(
                0, 4, (B, target_actions.shape[1]), device=target_actions.device
            )
            noisy_dir[mask_place] = noisy_dir_[mask_place]

        elif self.config.noise == "masking":
            noisy_actions[mask_place] = token2idx("<-MASK->")
            noisy_states[mask_place] = self.config.env_size**2  # token2idx('<-PAD->')
            noisy_waypoint_1[mask_place] = (
                self.config.env_size**2
            )  # token2idx('<-PAD->')
            noisy_waypoint_2[mask_place] = (
                self.config.env_size**2
            )  # token2idx('<-PAD->')
            noisy_dir = (
                target_states[:, :, 2].long().clone().detach()
            )  # torch.randint(0, 4, (B, target_actions.shape[1]), device=target_actions.device)
            noisy_dir[mask_place] = 4

        ########## update the conditional embeddings ########
        state_embeddings = self.state_embeddings(noisy_states.unsqueeze(-1).long())
        goal_embeddings = self.state_embeddings(
            torch.cat(
                (noisy_waypoint_1.unsqueeze(-1), noisy_waypoint_2.unsqueeze(-1)), dim=-1
            ).long()
        )
        direction_embeddings = self.direction_embeddings(noisy_dir.unsqueeze(-1).long())
        state_embeddings = torch.cat(
            (state_embeddings, direction_embeddings, goal_embeddings), dim=2
        ).view(target_states.shape[0] * target_states.shape[1], -1)
        state_embeddings = self.state_embeddings_linear(state_embeddings)

        true_target_states = target_states.clone().detach()
        true_target_states[:, self.action_horizon :, :] = 0

        # we want to mimic what happens in rollout, this prevents the model from cheating by looking at the waypoint in non masked positions.
        true_target_states[:, :, 3:] = 0

        cat_target_states_ = optimized_grid_encoder_state(
            true_target_states[:, :, :2], self.config.env_size
        )
        cat_waypoint_1_ = optimized_grid_encoder_state(
            true_target_states[:, :, -2:], self.config.env_size
        )
        cat_waypoint_2_ = optimized_grid_encoder_state(
            true_target_states[:, :, -4:-2], self.config.env_size
        )
        true_state_embeddings = self.state_embeddings(
            cat_target_states_.argmax(-1).unsqueeze(-1).long()
        )
        true_goal_embeddings = self.state_embeddings(
            torch.cat(
                (
                    cat_waypoint_1_.argmax(-1).unsqueeze(-1),
                    cat_waypoint_2_.argmax(-1).unsqueeze(-1),
                ),
                dim=-1,
            ).long()
        )
        true_dir_embeddings = self.state_embeddings(
            true_target_states[:, :, 3].long().unsqueeze(-1)
        )
        true_state_embeddings = torch.cat(
            (true_state_embeddings, true_dir_embeddings, true_goal_embeddings), dim=2
        ).view(target_states.shape[0] * target_states.shape[1], -1)
        true_state_embeddings = self.state_embeddings_linear(true_state_embeddings)

        # jointly learn the image and instr
        for controler in self.controllers:
            image_embeddings = controler(image_embeddings, instr_embedding)
            true_cond = controler(true_cond, instr_embedding)

        for controller in self.agent_pos_controllers:
            image_embeddings = controller(image_embeddings, state_embeddings)
            true_cond = controller(true_cond, true_state_embeddings)

        state_embeddings = self.film_pool(image_embeddings)
        true_cond = self.film_pool(true_cond)

        state_embeddings = state_embeddings.view(
            target_states.shape[0], target_states.shape[1], self.config.n_embd
        )
        true_cond = true_cond.view(
            target_states.shape[0], target_states.shape[1], self.config.n_embd
        )

        obs_cond = state_embeddings

        t = t.to(target_actions.device)

        ######### We feed corrupt tokens to transformer ##########
        (
            noise_pred_action,
            noise_pred_state,
            noise_pred_way_1,
            noise_pred_way_2,
            noisy_pred_dir,
        ) = self.noise_pred_net(
            noisy_actions,
            t,
            cond=obs_cond,
            true_cond=true_cond,
        )

        action_dist = CategoricalActionDistribution(noise_pred_action)
        sample_actions, model_log_probs = sample_actions_log_probs(action_dist)
        # only computed at masked positions.
        energy = self.compute_energy(
            target_actions,
            action_dist.log_probs,
            mask_place,
            pad_pos=pad_positions,
            use_masks=True,
        )
        energy_traj = energy.sum(dim=-1).sum(dim=-1)

        if self.config.extra_config.exploration_loss == "entropy":
            exploration_loss, entropy = self.exploration_loss_func(
                action_dist,
                exclude_last=False,
                pad_pos=pad_positions,
            )
        elif self.config.extra_config.exploration_loss == "symmetric_kl":
            exploration_loss = self.exploration_loss_func(action_dist)
            entropy = action_dist.entropy()

        ####### Noise contrastive Loss (Optional) - this is to see if NCE offers better results ########
        # uniform_noise_actions = torch.randint(
        #     0,
        #     self.action_dim - 2,
        #     (B, target_actions.shape[1], 1),
        #     device=target_actions.device,
        # )

        # cont_action_15 = target_actions.clone()

        # # noise - 15% # flip 1-2 indices alone 
        # random_seq = torch.rand((B, target_actions.shape[1]))
        # mask_25 = random_seq < 0.15

        # cont_action_15[mask_25] = uniform_noise_actions[mask_25]

        # contast_action = torch.cat([target_actions, cont_action_15], dim=0)
        # t_ = torch.cat([t, t], dim=0)
        # contrast_cond = torch.cat([obs_cond, obs_cond], dim=0)
        # contrast_true_cond = torch.cat([true_cond, true_cond], dim=0)
        # if self.config.noise == "masking":
        #     mask_full_uncond_actions = torch.full_like(
        #         contast_action, token2idx("<-MASK->")
        #     )
        # else:
        #     mask_full_uncond_actions = torch.randint(
        #         0,
        #         self.action_dim - 2,
        #         (contast_action.shape[0], target_actions.shape[1], 1),
        #         device=target_actions.device,
        #     )
        # noise_pred_action_con, _, _, _, _ = self.noise_pred_net(
        #     mask_full_uncond_actions.squeeze(-1),  # all masked input!
        #     t_,
        #     cond=contrast_cond,
        #     true_cond=contrast_true_cond,
        # )

        # action_dist_contrast = CategoricalActionDistribution(noise_pred_action_con)

        # energy_contrast = self.compute_energy(
        #     contast_action,
        #     action_dist_contrast.log_probs,
        #     torch.cat(
        #         [
        #             torch.ones_like(mask_place, dtype=bool).to(mask_place.device),
        #             torch.ones_like(mask_place, dtype=bool).to(mask_place.device),
        #         ],
        #         dim=0,
        #     ),
        #     pad_pos=torch.cat([pad_positions, pad_positions], dim=0),
        #     use_masks=False,
        #     use_sum=True,
        # )

        # energy_contrast = energy_contrast.sum(-1)  # .sum(-1)

        # energy_real, energy_fake = torch.chunk(energy_contrast, 2, 0)
        # energy_stack = torch.cat(
        #     [energy_real.unsqueeze(-1), energy_fake.unsqueeze(-1)], dim=-1
        # )

        # target = torch.zeros(energy_real.size(0)).to(energy_stack.device)  # (B)
        # loss_energy = F.cross_entropy(
        #     -1 * energy_stack, target.long(), reduction="none"
        # )[:, None]

        loss_here_only = mask_place.to(pad_positions.device) & pad_positions.squeeze(
            -1
        )  # don't calculate loss for the padded positions and non masked positions

        loss_dir = F.cross_entropy(
            noisy_pred_dir[loss_here_only],
            dir_gt[loss_here_only],
            reduction="none",
            ignore_index=-1,
        )
        loss_action = self.loss_crossent(
            noise_pred_action[loss_here_only], target_actions[loss_here_only].view(-1)
        )

        loss_state = F.cross_entropy(
            noise_pred_state[loss_here_only],
            cat_target_states[loss_here_only],
            reduction="none",
            ignore_index=-1,
        )
        loss_waypoint_1 = F.cross_entropy(
            noise_pred_way_1[loss_here_only],
            cat_waypoint_1[loss_here_only],
            reduction="none",
            ignore_index=-1,
        )
        loss_waypoint_2 = F.cross_entropy(
            noise_pred_way_2[loss_here_only],
            cat_waypoint_2[loss_here_only],
            reduction="none",
            ignore_index=-1,
        )

        loss_dir = loss_dir.mean()
        loss_action = loss_action.mean()
        loss_state = loss_state.mean()
        loss_waypoint_1 = loss_waypoint_1.mean()
        loss_waypoint_2 = loss_waypoint_2.mean()

        loss_final = (
            loss_action
            + loss_dir
            + loss_state
            + loss_waypoint_1
            + loss_waypoint_2
            + 0.01 * energy_traj.mean() # large scalar often - depends on length of trajectory
            + exploration_loss.mean()
        )

        action_correct_rate = torch.sum(
            target_actions.squeeze(-1)[loss_here_only] == sample_actions[loss_here_only]
        ) / torch.sum(loss_here_only)

        loss_dict = {
            "loss": loss_final,
            "loss_action": loss_action,
            "loss_state": loss_state,
            "loss_waypoint_1": loss_waypoint_1,
            "loss_waypoint_2": loss_waypoint_2,
            "energy": energy.mean(),
            "loss_dir": loss_dir,
            "exploration_loss": exploration_loss.mean(),
            "entropy": entropy.mean(),
            "action_correct_rate": action_correct_rate,
            # "loss_energy": loss_energy.mean(),
            "energy_traj": energy_traj.mean(),
            "temperature": self.temperature(),
            # "energy_real": energy_real.mean(),
            # "energy_fake": energy_fake.mean(),
        }

        return loss_final, noise_pred_action, loss_dict
