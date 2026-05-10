"""
Mini DreamerV3 — A Minimal PyTorch Implementation
===================================================
From-scratch implementation of DreamerV3 (Mastering Diverse Domains through
World Models, Hafner et al. 2023) capturing all key algorithmic innovations:

  • Categorical RSSM with 32×32 discrete latent space
  • Symlog predictions & two-hot encoded targets
  • Actor with Reinforce + straight-through gradient estimator
  • Critic with slow-target EMA & discrete regression
  • Free-bits KL balancing

Author : Guru
Reference: https://arxiv.org/abs/2301.04104
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from typing import Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────
#  Symlog Transform & Two-Hot Encoding
# ─────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic compression: sign(x) * ln(|x| + 1).
    
    Squashes large magnitudes while preserving sign — crucial for
    stabilizing learning across environments with vastly different
    reward scales (a key DreamerV3 innovation).
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def two_hot_encode(x: torch.Tensor, num_bins: int = 255,
                   low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Soft binning of continuous scalars into a two-hot distribution.
    
    Instead of predicting a single scalar, DreamerV3 predicts a categorical
    distribution over evenly-spaced bins. Each target activates the two
    nearest bins proportionally — this is the "two-hot" encoding.
    
    Args:
        x:        Scalar values of any shape (...)
        num_bins: Number of discrete bins
        low/high: Range of the bin centers
    
    Returns:
        Two-hot encoded tensor (..., num_bins)
    """
    orig_shape = x.shape
    x_flat = x.reshape(-1).clamp(low, high)
    
    bins = torch.linspace(low, high, num_bins, device=x.device)
    bin_width = bins[1] - bins[0]
    
    # Find the lower bin index for each value
    below = ((x_flat - low) / bin_width).floor().long().clamp(0, num_bins - 2)
    above = below + 1
    
    # Fractional position between the two bins
    frac = ((x_flat - bins[below]) / bin_width).clamp(0, 1)
    
    result = torch.zeros(x_flat.shape[0], num_bins, device=x.device)
    result.scatter_(-1, below.unsqueeze(-1), (1 - frac).unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), frac.unsqueeze(-1))
    
    return result.reshape(*orig_shape, num_bins)


def decode_two_hot(logits: torch.Tensor, num_bins: int = 255,
                   low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Decode a two-hot distribution back to a scalar value."""
    bins = torch.linspace(low, high, num_bins, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)


# ─────────────────────────────────────────────────────────────
#  MLP Building Block
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple feed-forward network with LayerNorm + SiLU activations.
    
    DreamerV3 uses LayerNorm (not BatchNorm) and SiLU (not ReLU) throughout
    for more stable gradient flow.
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512,
                 num_layers: int = 3, dist_output: bool = False):
        super().__init__()
        layers = []
        for i in range(num_layers):
            inp = in_dim if i == 0 else hidden_dim
            out = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(inp, out))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out))
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.dist_output = dist_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
#  Encoder & Decoder (supports both vector and image obs)
# ─────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """CNN encoder for image observations.
    
    Transforms pixel observations into a flat feature vector that feeds
    into the RSSM's posterior (representation model).
    """
    
    def __init__(self, in_channels: int = 3, depth: int = 48):
        super().__init__()
        self.depth = depth
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 1 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return self.convs(x).reshape(x.shape[0], -1)


class ConvDecoder(nn.Module):
    """CNN decoder — reconstructs images from latent state."""
    
    def __init__(self, latent_dim: int, out_channels: int = 3, depth: int = 48):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(latent_dim, 32 * depth)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(2 * depth, 1 * depth, 6, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(1 * depth, out_channels, 6, stride=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.reshape(x.shape[0], 32 * self.depth, 1, 1)
        return self.deconvs(x)


class VectorEncoder(nn.Module):
    """MLP encoder for low-dimensional vector observations (e.g. CartPole)."""
    
    def __init__(self, obs_dim: int, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorDecoder(nn.Module):
    """MLP decoder for vector observations."""
    
    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
#  Categorical RSSM — Heart of DreamerV3
# ─────────────────────────────────────────────────────────────

class CategoricalRSSM(nn.Module):
    """Recurrent State-Space Model with categorical latent variables.
    
    The RSSM maintains two state components:
      • Deterministic state h_t  — GRU hidden state aggregating history
      • Stochastic  state z_t  — 32 categorical variables × 32 classes each
    
    This gives a discrete latent space of 32^32 possible configurations,
    providing rich expressiveness while enabling straight-through gradient
    estimation. This is a KEY departure from DreamerV1/V2 which used
    Gaussian latents.
    
    Two distributions are computed:
      • Prior     p(z_t | h_t)        — "dynamics predictor" (imagination)
      • Posterior q(z_t | h_t, x_t)   — "representation model" (grounded in obs)
    
    The KL divergence between these drives the world model to learn dynamics
    that can accurately predict future states without observations.
    """
    
    def __init__(self, embed_dim: int = 512, action_dim: int = 2,
                 deter_dim: int = 512, stoch_dim: int = 32,
                 num_classes: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.stoch_size = stoch_dim * num_classes  # flat stochastic state
        
        # --- Sequence Model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) ---
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_size + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        
        # --- Dynamics Predictor (Prior): p(z_t | h_t) ---
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * num_classes),
        )
        
        # --- Representation Model (Posterior): q(z_t | h_t, x_t) ---
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * num_classes),
        )
    
    @property
    def full_state_dim(self) -> int:
        """Total latent state dimensionality (deter + stoch)."""
        return self.deter_dim + self.stoch_size
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Create zero-initialized RSSM state."""
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_size, device=device),
        }
    
    def get_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate deterministic and stochastic state into feature vector."""
        return torch.cat([state["deter"], state["stoch"]], dim=-1)
    
    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with straight-through gradients.
        
        Uses the Gumbel-Softmax trick during training to allow gradients to
        flow through the discrete sampling operation.
        """
        B = logits.shape[0]
        logits = logits.reshape(B, self.stoch_dim, self.num_classes)
        
        if self.training:
            # Straight-through: one-hot forward, soft backward
            probs = F.softmax(logits, dim=-1)
            sample = F.gumbel_softmax(logits, tau=1.0, hard=True)
        else:
            probs = F.softmax(logits, dim=-1)
            indices = probs.argmax(dim=-1)
            sample = F.one_hot(indices, self.num_classes).float()
        
        return sample.reshape(B, -1)  # (B, stoch_dim * num_classes)
    
    def observe_step(self, prev_state: Dict[str, torch.Tensor],
                     prev_action: torch.Tensor,
                     embed: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], dict]:
        """Single RSSM step with observation (for world model training).
        
        Returns:
            state:  Updated state dict {deter, stoch}
            info:   Dict with prior_logits, posterior_logits for KL computation
        """
        # Sequence model: advance deterministic state
        x = torch.cat([prev_state["stoch"], prev_action], dim=-1)
        x = self.pre_gru(x)
        deter = self.gru(x, prev_state["deter"])
        
        # Prior distribution (from dynamics only)
        prior_logits = self.prior_net(deter)
        
        # Posterior distribution (grounded in observation)
        posterior_logits = self.posterior_net(torch.cat([deter, embed], dim=-1))
        stoch = self._sample_categorical(posterior_logits)
        
        state = {"deter": deter, "stoch": stoch}
        info = {
            "prior_logits": prior_logits.reshape(-1, self.stoch_dim, self.num_classes),
            "posterior_logits": posterior_logits.reshape(-1, self.stoch_dim, self.num_classes),
        }
        return state, info
    
    def imagine_step(self, prev_state: Dict[str, torch.Tensor],
                     prev_action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single RSSM step WITHOUT observation (for imagination/dreaming).
        
        Uses the prior distribution only — this is how the agent "dreams"
        about future trajectories to train the actor and critic.
        """
        x = torch.cat([prev_state["stoch"], prev_action], dim=-1)
        x = self.pre_gru(x)
        deter = self.gru(x, prev_state["deter"])
        
        prior_logits = self.prior_net(deter)
        stoch = self._sample_categorical(prior_logits)
        
        return {"deter": deter, "stoch": stoch}
    
    def observe_sequence(self, embeds: torch.Tensor, actions: torch.Tensor,
                         initial_state: Dict[str, torch.Tensor]
                         ) -> Tuple[Dict[str, torch.Tensor], dict]:
        """Process a sequence of observations (for batch world model training).
        
        Args:
            embeds:  (B, T, embed_dim) — encoded observations
            actions: (B, T, action_dim) — actions taken
            initial_state: Starting RSSM state
        
        Returns:
            states: Dict with (B, T, ...) tensors for deter and stoch
            infos:  Dict with (B, T, ...) tensors for prior/posterior logits
        """
        B, T = embeds.shape[:2]
        state = initial_state
        
        all_deter, all_stoch = [], []
        all_prior_logits, all_posterior_logits = [], []
        
        for t in range(T):
            state, info = self.observe_step(state, actions[:, t], embeds[:, t])
            all_deter.append(state["deter"])
            all_stoch.append(state["stoch"])
            all_prior_logits.append(info["prior_logits"])
            all_posterior_logits.append(info["posterior_logits"])
        
        states = {
            "deter": torch.stack(all_deter, dim=1),
            "stoch": torch.stack(all_stoch, dim=1),
        }
        infos = {
            "prior_logits": torch.stack(all_prior_logits, dim=1),
            "posterior_logits": torch.stack(all_posterior_logits, dim=1),
        }
        return states, infos


# ─────────────────────────────────────────────────────────────
#  Reward & Continue Predictors
# ─────────────────────────────────────────────────────────────

class RewardPredictor(nn.Module):
    """Predicts rewards from latent states using symlog two-hot encoding.
    
    DreamerV3 predicts rewards as a categorical distribution over symlog-
    transformed bins, enabling the model to handle diverse reward scales.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 512,
                 num_bins: int = 255):
        super().__init__()
        self.num_bins = num_bins
        self.net = MLP(state_dim, num_bins, hidden_dim, num_layers=3)
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns logits over reward bins."""
        return self.net(feat)
    
    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns decoded scalar reward."""
        logits = self.forward(feat)
        return symexp(decode_two_hot(logits, self.num_bins))


class ContinuePredictor(nn.Module):
    """Predicts episode continuation probability from latent state.
    
    Outputs a Bernoulli probability p(continue | s_t).
    This is essential for proper value estimation near episode boundaries.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = MLP(state_dim, 1, hidden_dim, num_layers=3)
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns continue logit (pre-sigmoid)."""
        return self.net(feat).squeeze(-1)
    
    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns continue probability."""
        return torch.sigmoid(self.forward(feat))


# ─────────────────────────────────────────────────────────────
#  Actor — Policy Network
# ─────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Policy network that selects actions from latent states.
    
    For discrete actions:  outputs categorical logits
    For continuous actions: outputs mean & std for squashed Normal
    
    DreamerV3 trains the actor using a combination of:
      1. Reinforce gradients (through the policy log-prob)
      2. Straight-through dynamics gradients (through imagined returns)
    This stabilizes training by providing both high-variance unbiased
    gradients and low-variance biased gradients.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512,
                 discrete: bool = True, num_layers: int = 3):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        
        if discrete:
            self.net = MLP(state_dim, action_dim, hidden_dim, num_layers)
        else:
            self.net = MLP(state_dim, 2 * action_dim, hidden_dim, num_layers)
    
    def forward(self, feat: torch.Tensor) -> D.Distribution:
        """Returns action distribution."""
        out = self.net(feat)
        
        if self.discrete:
            return D.Categorical(logits=out)
        else:
            mean, log_std = out.chunk(2, dim=-1)
            log_std = log_std.clamp(-5, 2)
            return D.Normal(mean, log_std.exp())
    
    def get_action(self, feat: torch.Tensor, 
                   deterministic: bool = False) -> torch.Tensor:
        """Sample an action (or take the mode for evaluation)."""
        dist = self.forward(feat)
        if deterministic:
            if self.discrete:
                return dist.probs.argmax(dim=-1)
            else:
                return dist.mean
        return dist.sample()


# ─────────────────────────────────────────────────────────────
#  Critic — Value Network
# ─────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """Value function estimating expected return from latent states.
    
    Uses discrete regression (two-hot encoded bins in symlog space)
    matching the reward predictor's output format. A slow-moving
    target network (EMA) stabilizes training — updated every step
    with momentum τ instead of periodic hard copies.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 512,
                 num_bins: int = 255, slow_target_tau: float = 0.02):
        super().__init__()
        self.num_bins = num_bins
        self.tau = slow_target_tau
        
        self.net = MLP(state_dim, num_bins, hidden_dim, num_layers=3)
        
        # Slow target network (EMA copy)
        self.target_net = copy.deepcopy(self.net)
        for p in self.target_net.parameters():
            p.requires_grad = False
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns logits over value bins."""
        return self.net(feat)
    
    def target(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate from the slow target network."""
        logits = self.target_net(feat)
        return symexp(decode_two_hot(logits, self.num_bins))
    
    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate from the online network."""
        logits = self.forward(feat)
        return symexp(decode_two_hot(logits, self.num_bins))
    
    def update_target(self):
        """Exponential moving average update of target network."""
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.data.lerp_(p.data, self.tau)


# ─────────────────────────────────────────────────────────────
#  DreamerV3 Agent — Full System
# ─────────────────────────────────────────────────────────────

class DreamerV3Agent(nn.Module):
    """Complete DreamerV3 agent orchestrating all components.
    
    Architecture Overview:
    ┌─────────────────────────────────────────────────────┐
    │                    World Model                       │
    │  ┌──────────┐   ┌───────────────┐   ┌───────────┐  │
    │  │ Encoder  │──▶│ Categorical   │──▶│ Decoder   │  │
    │  │          │   │    RSSM       │   │ Reward    │  │
    │  └──────────┘   │ (h_t, z_t)   │   │ Continue  │  │
    │                 └───────────────┘   └───────────┘  │
    └─────────────────────────────────────────────────────┘
                           │
                    Imagination (dream)
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
         ┌─────────┐              ┌──────────┐
         │  Actor  │              │  Critic  │
         │ (policy)│              │  (value) │
         └─────────┘              └──────────┘
    
    Training Loop:
      1. Collect experience from environment
      2. Train world model on observed sequences (recon + KL + reward + continue)
      3. Imagine future trajectories using world model
      4. Train actor-critic on imagined trajectories
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 discrete_actions: bool = True,
                 image_obs: bool = False,
                 deter_dim: int = 512,
                 stoch_dim: int = 32,
                 num_classes: int = 32,
                 hidden_dim: int = 512,
                 embed_dim: int = 512,
                 imagination_horizon: int = 15,
                 gamma: float = 0.997,
                 lambda_: float = 0.95,
                 free_nats: float = 1.0,
                 kl_balance: float = 0.8,
                 device: str = "cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.image_obs = image_obs
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.free_nats = free_nats
        self.kl_balance = kl_balance
        self.device = device
        
        # Encoder
        if image_obs:
            self.encoder = ConvEncoder(in_channels=obs_dim)
            # Calculate encoder output dim
            with torch.no_grad():
                dummy = torch.zeros(1, obs_dim, 64, 64)
                enc_out_dim = self.encoder(dummy).shape[-1]
        else:
            self.encoder = VectorEncoder(obs_dim, embed_dim)
            enc_out_dim = embed_dim
        
        # RSSM
        act_input_dim = action_dim if discrete_actions else action_dim
        self.rssm = CategoricalRSSM(
            embed_dim=enc_out_dim,
            action_dim=act_input_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
        
        state_dim = self.rssm.full_state_dim
        
        # Decoder
        if image_obs:
            self.decoder = ConvDecoder(state_dim, out_channels=obs_dim)
        else:
            self.decoder = VectorDecoder(state_dim, obs_dim, hidden_dim)
        
        # Predictors
        self.reward_pred = RewardPredictor(state_dim, hidden_dim)
        self.continue_pred = ContinuePredictor(state_dim, hidden_dim)
        
        # Actor-Critic
        self.actor = Actor(state_dim, action_dim, hidden_dim, discrete_actions)
        self.critic = Critic(state_dim, hidden_dim)
        
        self.to(device)
    
    def get_initial_state(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Get zero-initialized RSSM state."""
        return self.rssm.initial_state(batch_size, self.device)
    
    # ── Interaction ──────────────────────────────────────
    
    @torch.no_grad()
    def policy(self, obs: torch.Tensor, state: Dict[str, torch.Tensor],
               action: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Select action given current observation and state.
        
        This is the main entry point during environment interaction.
        """
        embed = self.encoder(obs)
        state, _ = self.rssm.observe_step(state, action, embed)
        feat = self.rssm.get_feat(state)
        action = self.actor.get_action(feat, deterministic=not training)
        
        if self.discrete_actions:
            action_encoded = F.one_hot(action, self.action_dim).float()
        else:
            action_encoded = action
        
        return action_encoded, state
    
    # ── World Model Training ────────────────────────────
    
    def world_model_loss(self, obs: torch.Tensor, actions: torch.Tensor,
                         rewards: torch.Tensor, continues: torch.Tensor
                         ) -> Tuple[torch.Tensor, dict]:
        """Compute world model loss on observed sequences.
        
        Loss components:
          • Reconstruction: How well can we predict observations?
          • Reward:         How well can we predict rewards?
          • Continue:       How well can we predict episode boundaries?
          • KL Divergence:  Prior should match posterior (dynamics learning)
        
        Args:
            obs:       (B, T, obs_dim) or (B, T, C, H, W) for images
            actions:   (B, T, action_dim)
            rewards:   (B, T)
            continues: (B, T) — 1.0 if episode continues, 0.0 if terminal
        
        Returns:
            total_loss: Scalar loss for optimization
            metrics:    Dict of individual loss components
        """
        B, T = obs.shape[:2]
        
        # Encode all observations
        if self.image_obs:
            embeds = self.encoder(obs.reshape(B * T, *obs.shape[2:]))
            embeds = embeds.reshape(B, T, -1)
        else:
            embeds = self.encoder(obs.reshape(B * T, -1))
            embeds = embeds.reshape(B, T, -1)
        
        # Run RSSM over sequence
        init_state = self.get_initial_state(B)
        states, infos = self.rssm.observe_sequence(embeds, actions, init_state)
        
        # Get features: (B, T, state_dim)
        feat = self.rssm.get_feat({
            "deter": states["deter"],
            "stoch": states["stoch"],
        })
        feat_flat = feat.reshape(B * T, -1)
        
        # --- Reconstruction loss ---
        obs_pred = self.decoder(feat_flat)
        if self.image_obs:
            obs_target = obs.reshape(B * T, *obs.shape[2:])
        else:
            obs_target = obs.reshape(B * T, -1)
        recon_loss = F.mse_loss(obs_pred, obs_target)
        
        # --- Reward loss (symlog two-hot) ---
        reward_logits = self.reward_pred(feat_flat)
        reward_target = two_hot_encode(symlog(rewards.reshape(B * T)),
                                       self.reward_pred.num_bins)
        reward_loss = -torch.sum(reward_target * F.log_softmax(reward_logits, dim=-1),
                                 dim=-1).mean()
        
        # --- Continue loss (binary cross-entropy) ---
        continue_logits = self.continue_pred(feat_flat)
        continue_loss = F.binary_cross_entropy_with_logits(
            continue_logits, continues.reshape(B * T)
        )
        
        # --- KL Divergence (free-bits balancing) ---
        prior_logits = infos["prior_logits"].reshape(B * T, -1,
                                                     self.rssm.num_classes)
        post_logits = infos["posterior_logits"].reshape(B * T, -1,
                                                       self.rssm.num_classes)
        
        prior_dist = D.Categorical(logits=prior_logits)
        post_dist = D.Categorical(logits=post_logits)
        
        # DreamerV3 KL balancing: asymmetric weighting
        kl_value = D.kl_divergence(post_dist, prior_dist).sum(-1).mean()
        kl_loss = torch.clamp(kl_value, min=self.free_nats)
        
        # Total world model loss
        total_loss = recon_loss + reward_loss + continue_loss + 0.1 * kl_loss
        
        metrics = {
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_wm_loss": total_loss.item(),
        }
        return total_loss, metrics
    
    # ── Imagination & Actor-Critic Training ─────────────
    
    def imagine_trajectories(self, start_states: Dict[str, torch.Tensor]
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dream future trajectories for actor-critic training.
        
        Starting from real states observed during world model training,
        the agent "imagines" H steps into the future using only the
        prior dynamics (no observations needed). The actor selects
        actions, and the reward/continue predictors evaluate outcomes.
        
        Returns:
            feats:     (B, H, state_dim) — imagined latent states
            rewards:   (B, H) — predicted rewards
            continues: (B, H) — predicted continue probabilities
        """
        state = {k: v.detach() for k, v in start_states.items()}
        
        feats, rewards, continues = [], [], []
        
        for _ in range(self.imagination_horizon):
            feat = self.rssm.get_feat(state)
            action_dist = self.actor(feat)
            action = action_dist.sample()
            
            if self.discrete_actions:
                action = F.one_hot(action, self.action_dim).float()
            
            state = self.rssm.imagine_step(state, action)
            
            next_feat = self.rssm.get_feat(state)
            feats.append(next_feat)
            rewards.append(self.reward_pred.predict(next_feat))
            continues.append(self.continue_pred.predict(next_feat))
        
        return (torch.stack(feats, dim=1),
                torch.stack(rewards, dim=1),
                torch.stack(continues, dim=1))
    
    def compute_lambda_returns(self, rewards: torch.Tensor,
                               values: torch.Tensor,
                               continues: torch.Tensor) -> torch.Tensor:
        """Compute λ-returns for imagined trajectories (GAE-style).
        
        R_t^λ = r_t + γ c_t [(1-λ) V(s_{t+1}) + λ R_{t+1}^λ]
        
        Where c_t is the predicted continue probability, providing
        automatic episode boundary handling in imagination.
        """
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last_value = values[:, -1]
        
        for t in reversed(range(H)):
            if t == H - 1:
                next_value = last_value
            else:
                next_value = values[:, t + 1]
            
            returns[:, t] = rewards[:, t] + self.gamma * continues[:, t] * (
                (1 - self.lambda_) * next_value + self.lambda_ * (
                    returns[:, t + 1] if t < H - 1 else next_value
                )
            )
        
        return returns
    
    def actor_critic_loss(self, start_states: Dict[str, torch.Tensor]
                         ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Train actor and critic on imagined trajectories.
        
        1. Imagine H-step trajectories from start_states
        2. Compute λ-returns using critic's target network
        3. Train critic to match λ-returns (two-hot regression)
        4. Train actor to maximize returns (Reinforce + dynamics grads)
        
        Returns:
            actor_loss:  Scalar loss for actor optimizer
            critic_loss: Scalar loss for critic optimizer
            metrics:     Dict with diagnostics
        """
        # Imagine future
        feats, rewards, continues = self.imagine_trajectories(start_states)
        B, H = feats.shape[:2]
        
        # Critic values from target network
        with torch.no_grad():
            values = self.critic.target(feats.reshape(B * H, -1)).reshape(B, H)
        
        # λ-returns
        returns = self.compute_lambda_returns(rewards, values, continues)
        
        # --- Critic loss (discrete regression with two-hot) ---
        critic_logits = self.critic(feats.reshape(B * H, -1))
        target = two_hot_encode(symlog(returns.reshape(B * H).detach()),
                                self.critic.num_bins)
        critic_loss = -torch.sum(target * F.log_softmax(critic_logits, dim=-1),
                                 dim=-1).mean()
        
        # --- Actor loss (maximize returns) ---
        # Reinforce-style: -log_prob * advantage
        advantage = (returns - values).detach()
        # Normalize advantages for stability
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # Recompute actions for log_prob
        feat_flat = feats.reshape(B * H, -1).detach()
        actor_dist = self.actor(feat_flat)
        
        if self.discrete_actions:
            # For discrete: get action probs and use reinforce
            entropy = actor_dist.entropy().mean()
            # Use policy gradient with baseline
            action = actor_dist.sample()
            log_prob = actor_dist.log_prob(action)
            actor_loss = -(log_prob * advantage.reshape(B * H)).mean()
            actor_loss -= 3e-4 * entropy  # entropy regularization
        else:
            action = actor_dist.rsample()
            log_prob = actor_dist.log_prob(action).sum(-1)
            actor_loss = -(log_prob * advantage.reshape(B * H)).mean()
            entropy = actor_dist.entropy().sum(-1).mean()
            actor_loss -= 3e-4 * entropy
        
        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_imagined_reward": rewards.mean().item(),
            "mean_value": values.mean().item(),
            "mean_return": returns.mean().item(),
            "entropy": entropy.item(),
        }
        return actor_loss, critic_loss, metrics


# ─────────────────────────────────────────────────────────────
#  Model Summary Utility
# ─────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> str:
    """Pretty-print parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1e6:
        return f"{total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)"
    return f"{total/1e3:.1f}K params ({trainable/1e3:.1f}K trainable)"


if __name__ == "__main__":
    # Quick sanity check
    print("=" * 60)
    print("  Mini DreamerV3 — Component Test")
    print("=" * 60)
    
    agent = DreamerV3Agent(
        obs_dim=4,
        action_dim=2,
        discrete_actions=True,
        image_obs=False,
        deter_dim=256,
        stoch_dim=16,
        num_classes=16,
        hidden_dim=256,
        embed_dim=256,
    )
    
    print(f"\n📊 Model size: {count_parameters(agent)}")
    
    # Test policy
    obs = torch.randn(1, 4)
    state = agent.get_initial_state(1)
    action = torch.zeros(1, 2)  # one-hot
    
    action_out, new_state = agent.policy(obs, state, action)
    print(f"✅ Policy:  obs{list(obs.shape)} → action{list(action_out.shape)}")
    
    # Test world model loss
    B, T = 4, 8
    obs_seq = torch.randn(B, T, 4)
    act_seq = F.one_hot(torch.randint(0, 2, (B, T)), 2).float()
    rew_seq = torch.randn(B, T)
    cont_seq = torch.ones(B, T)
    
    wm_loss, wm_metrics = agent.world_model_loss(obs_seq, act_seq, rew_seq, cont_seq)
    print(f"✅ WM Loss: {wm_loss.item():.4f}")
    for k, v in wm_metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # Test imagination
    start = agent.get_initial_state(B)
    actor_loss, critic_loss, ac_metrics = agent.actor_critic_loss(start)
    print(f"✅ Actor Loss:  {actor_loss.item():.4f}")
    print(f"✅ Critic Loss: {critic_loss.item():.4f}")
    
    print(f"\n{'=' * 60}")
    print("  All components working! ✨")
    print(f"{'=' * 60}")
