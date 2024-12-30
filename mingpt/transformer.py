from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from mingpt.module_attr_mixin import ModuleAttrMixin
from torch import distributions as pyd
import math
import torch.nn.functional as F
import numpy as np
from mingpt.conv2d_components import Conv2dBlock
logger = logging.getLogger(__name__)

class ResidualBlock2D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        )
        
        # make sure dimensions compatible
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x in_channels x H x W ]
            returns:
            out : [ batch_size x out_channels x H x W ]
        '''
        out = self.blocks(x)
        out = out + self.residual_conv(x)
        return out
    
class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
    
class SeqToImgDecoder(nn.Module):
    def __init__(self, input_dim=256, output_shape=(10, 10, 7), seq_len=125):
        super().__init__()
        self.seq_len = seq_len
        self.output_shape = output_shape
        
        # Project to a higher channel count suitable for reshaping into spatial dimensions
        self.proj = nn.Linear(input_dim, output_shape[0] * output_shape[1] * 8)  # Increase channels before reshaping
        
        # Define 2D residual blocks for processing image-like data
        self.res_block1 = ResidualBlock2D(8, output_shape[2], kernel_size=3, n_groups=output_shape[2])  # Upsampled channel number
        
        # self.res_block2 = ResidualBlock2D(16, output_shape[2], kernel_size=3, n_groups=2)  # Output channel number
        
        # Upsampling layers
        # self.upsample1 = Upsample2d(16)
        # self.upsample2 = Upsample2d(output_shape[2])

    def forward(self, x):
        batch_size = x.size(0)
        
        
        # Linear projection and reshape to image-like structure
        x = self.proj(x)  # Shape after projection: (batch_size, seq_len, H*W*8)
        x = x.view(batch_size, self.seq_len, 8, self.output_shape[0], self.output_shape[1])  # Reshape: (batch, seq_len, C, H, W)

        # Process each sequence element independently
        # x = x.permute(0, 2, 1, 3, 4)  # Rearrange to (batch, C, seq_len, H, W)
        # x = x.reshape(-1, self.seq_len, self.output_shape[0], self.output_shape[1])  # Flatten batch and channels: (batch*C, seq_len, H, W)
        x = x.reshape(batch_size * self.seq_len, 8, self.output_shape[0], self.output_shape[0])
        # Apply residual blocks and upsampling
        x = self.res_block1(x)
        # x = self.upsample1(x)
        # x = self.res_block2(x)
        # x = self.upsample2(x)

        # Final reshape and permute to get back to the required dimension
        x = x.view(batch_size, self.output_shape[2], self.seq_len, self.output_shape[0], self.output_shape[1])  # Assuming upsample scales by 2
        x = x.permute(0, 2, 3, 4, 1)  # Rearrange to (batch, seq_len, H, W, C)

        return x
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # assumes timesteps is in the range 0 to 1000

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
    
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[0.0, 6.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)
    

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_emb, config.n_emb)
        self.query = nn.Linear(config.n_emb, config.n_emb)
        self.value = nn.Linear(config.n_emb, config.n_emb)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.register_buffer("mask", torch.ones(config.block_size + 1, config.block_size + 1)
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None, is_debug=False):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_emb, 4 * config.n_emb),
            GELU(),
            nn.Linear(4 * config.n_emb, config.n_emb),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, is_debug):
        x = x + self.attn(self.ln1(x), is_debug=is_debug)
        x = x + self.mlp(self.ln2(x))
        return x
    
class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            output_states_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 4,
            n_head: int = 4,
            n_emb: int = 128,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=False,
            obs_as_cond: bool=True,
            n_cond_layers: int = 0,
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        self.causal_attn = causal_attn
        
        T = horizon*2
        T_cond = 1
        if not time_as_cond:
            # T += 1
            T_cond -= 1
        # obs_as_cond = cond_dim > 0
        # if obs_as_cond:
        #     assert time_as_cond
        #     T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Embedding(input_dim, n_emb)  #nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        # self.time_emb = transformer_timestep_embedding(time * 1000, n_emb) #SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        self.n_emb = n_emb
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb, bias=False)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if horizon > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, horizon+1, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
            # self.blocks = [] #nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
            # config.n_emb = n_emb
            # for i in range(config.n_layer):
            #     block = Block(config)
            #     self.blocks.append(block) 
            #     self.add_module('block_' + str(i), block)

        else:
            # encoder only BERT
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )
        
        # self.attn_scores = {"self_attn": [], "cross_attn": []}

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    # indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Sequential(nn.Linear(n_emb, n_emb, bias=False),
                                  nn.ReLU(),
                                    nn.Linear(n_emb, output_dim, bias=False))
        self.state_head = nn.Linear(n_emb, output_states_dim,  bias=False)
        self.ways_head_1 = nn.Linear(n_emb, output_states_dim,  bias=False)
        self.ways_head_2 = nn.Linear(n_emb, output_states_dim,  bias=False)
        self.dir_head = nn.Linear(n_emb, 4,  bias=False)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        # init
        # self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def loss_fn(
        self,
        a_hat_dist,
        a,
        # attention_mask,
        # entropy_reg,
        ):
        # a_hat is a SquashedNormal Distribution
        target_actions = a.float()
        normalized_tensor = (target_actions - target_actions.mean()) / target_actions.std()
        # Apply tanh to squash values to [-1, 1]
        squashed_tensor = torch.tanh(normalized_tensor)
        log_likelihood = a_hat_dist.log_likelihood(squashed_tensor).mean()
        
        entropy = a_hat_dist.entropy().mean()
        loss = -(log_likelihood + self.temperature().detach() * entropy)
        self.entropy = entropy
        return (
            loss,
            # -log_likelihood,
            # entropy,
        )
    
    def temperature(self):
        return self.log_temperature.exp()
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        true_cond: Optional[torch.Tensor]=None, 
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep

        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        
        time_emb = transformer_timestep_embedding(timesteps * 1000, self.n_emb).unsqueeze(1) #self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            # token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            token_embeddings = torch.zeros((input_emb.shape[0], input_emb.shape[1]*2, input_emb.shape[-1]), dtype=torch.float32, device=input_emb.device)
            token_embeddings[:,::2,:] = cond
            token_embeddings[:,1::2,:] = input_emb[:,-cond.shape[1]:,:]

            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings + time_emb)
            # (B,T+1,n_emb)
            x = self.encoder(src=x,)
            # # (B,T+1,n_emb)
            # x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            # if self.obs_as_cond:
            cond_obs_emb = self.cond_obs_emb(true_cond)
                # (B,To,n_emb)
            cond_obs_emb = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_obs_emb.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_obs_emb + position_embeddings + time_emb)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)


            
            # decoder
            
            cond = self.cond_obs_emb(cond)

            token_embeddings = torch.zeros((input_emb.shape[0], input_emb.shape[1]*2, input_emb.shape[-1]), dtype=torch.float32, device=input_emb.device)

            token_embeddings[:,::2,:] =  cond # state = film(image, inst, state) - corrupt
            token_embeddings[:,1::2,:] = input_emb[:,-cond.shape[1]:,:] # actions - corrupt


            # token_embeddings = input_emb

            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings + time_emb)
            # (B,T,n_emb)

            # #### get some maps ##### # if you want to visualize attn
            # for layer in self.decoder.layers:
            #     x = layer(x, memory, tgt_mask=None, memory_mask=None)
            #     self.attn_scores["self_attn"].append(layer.self_attn_scores)
            #     self.attn_scores["cross_attn"].append(layer.cross_attn_scores)


            if self.causal_attn:
                x = self.decoder(
                    tgt=x,
                    memory=memory,
                    tgt_mask=self.mask,
                    memory_mask=self.memory_mask,
                )
            else:
                x = self.decoder(
                    tgt=x,
                    memory=memory,
                    tgt_mask=None,
                    memory_mask=None
                )
            # (B,T,n_emb)
            # for block in self.blocks:
            #     x = block(x, False) 
        
        
        # head
        x_dec = self.ln_f(x)

        x_state = self.state_head(x_dec[:, 1::2, :])
        x_action = self.head(x_dec[:, ::2, :])
        x_way_1 = self.ways_head_1(x_dec)
        x_way_2 = self.ways_head_2(x_dec) # we can add more waypoint predictors as required for custom tasks

        x_dir = self.dir_head(x_dec)
        
        return x_action, x_state, x_way_1[:, 1::2, :], x_way_2[:, 1::2, :], x_dir[:, 1::2, :]
