<div align="center">

# 🏎️ DreamerV3: Self-Driving World Models

<div align="center">
  <img src="demo.gif" alt="DreamerV3 Self Driving Agent Dreaming" width="400"/>
</div>

**An Advanced PyTorch Implementation of DreamerV3 for Continuous Control & Physical AI**

*World model reinforcement learning agent that learns environment dynamics and plans by dreaming directly from pixel observations.*

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2301.04104-b31b1b.svg)](https://arxiv.org/abs/2301.04104)

</div>

---

## 🏁 Physical AI Core: Self-Driving from Pixels

This project is specifically engineered to showcase **Physical AI** readiness by training a self-driving agent (`CarRacing-v3`) entirely from image pixels. 

**DreamerV3** ([Hafner et al., 2023](https://arxiv.org/abs/2301.04104)) is a model-based reinforcement learning algorithm that learns a *world model* of the environment and then trains its policy entirely through *imagination*.

Our implementation excels in autonomous vehicular control:
1. **Pixel-to-Latent Compression**: The CNN encoder transforms `(3, 64, 64)` pixel observations into a categorical latent space, isolating structural semantic features (the track, the car).
2. **Dreaming the Road**: The agent rolls out future driving scenarios entirely inside its neural network (imagination) without querying the simulator, resulting in highly efficient sample complexity.
3. **Continuous Control**: Steering, braking, and accelerating are handled natively via stochastic latent state estimations.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      WORLD MODEL                         │
│                                                          │
│  Observation ──▶ [Encoder] ──▶ ┌─────────────────────┐   │
│   (Pixels)                     │  Categorical RSSM   │   │
│                                │  h_t = GRU(h,z,a)   │   │
│  Action ─────────────────────▶ │  z_t ~ Cat(32×32)   │   │
│                                └──────────┬──────────┘   │
│                                           │              │
│                 ┌─────────────────────────┼──────┐       │
│                 ▼             ▼            ▼      ▼      │
│            [Decoder]    [Reward]    [Continue]           │
│            (recon)      (symlog)    (Bernoulli)          │
└──────────────────────────────────────────────────────────┘
                            │
                     Imagination (H steps)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
         ┌──────────┐             ┌──────────┐
         │  ACTOR   │             │  CRITIC  │
         │ Reinforce│             │ Slow EMA │
         │ + STE    │             │ Two-Hot  │
         └──────────┘             └──────────┘
```

---

## ✨ Key Algorithmic Implementations

| Innovation | Description | Value for Robotics |
|---|---|---|
| **Categorical RSSM** | 32×32 discrete latent space instead of Gaussian | Richer expressiveness, prevents posterior collapse in pixel domains |
| **Symlog Transform** | `sign(x)·ln(\|x\|+1)` for predictions | Handles massive reward scaling dynamically |
| **Two-Hot Encoding** | Soft binning for scalar regression | More stable than direct MSE regression for highly dynamic environments |
| **Free-Bits KL** | KL divergence clamped with free nats | Ensures representation model doesn't overfit to noisy sensor data |
| **Slow Target EMA** | Exponential moving average critic target | Stable value estimation against changing environments |

---

## 🚀 Quick Start

### 1. Install Dependencies

You'll need `gymnasium[box2d]` and `swig` to support the self-driving physical AI environments.

```bash
pip install -r requirements.txt swig
```

### 2. Run Self-Driving Training

```bash
# Train on CarRacing-v3 from pixels (default)
python train.py --total-steps 100000

# Custom architecture scaling
python train.py --deter-dim 512 --stoch-dim 32 --num-classes 32 --hidden-dim 512
```

### 3. Visualizes Outputs

The system automatically generates `.mp4` inference videos over the course of training into the `videos/` folder.

---

## ⚙️ Configuration

| Argument | Default | Description |
|---|---|---|
| `--env` | `CarRacing-v3` | Gymnasium environment ID |
| `--total-steps` | `50000` | Total environment steps |
| `--deter-dim` | `256` | GRU hidden state size |
| `--stoch-dim` | `16` | Categorical variables |
| `--num-classes` | `16` | Classes per variable |
| `--hidden-dim` | `256` | MLP hidden layer size |
| `--lr` | `3e-4` | Learning rate |
| `--batch-size` | `16` | Training batch size |
| `--seq-len` | `32` | Sequence length for RSSM |

---

## 📚 References
- **DreamerV3:** [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) — Hafner et al., 2023
- **RSSM:** [Learning Latent Dynamics for Planning](https://arxiv.org/abs/1811.04551) — Hafner et al., 2019

## 📄 License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

<div align="center">
<b>Built for demonstrating state-of-the-art capability in continuous control and physical AI.</b>
</div>
