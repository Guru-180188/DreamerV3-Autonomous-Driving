"""
Training Script for Mini DreamerV3
=====================================
Trains the DreamerV3 agent on Gymnasium environments (default: CartPole-v1).

Usage:
    python train.py                        # Default: CartPole-v1
    python train.py --env LunarLander-v3   # Any Gymnasium env with vector obs
    python train.py --steps 50000          # Custom number of steps

The training loop follows the DreamerV3 paper:
  1. Interact with environment, store transitions in replay buffer
  2. Sample batches → train world model (RSSM + decoders)
  3. Imagine trajectories from world model → train actor-critic
  4. Repeat
"""

import argparse
import random
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

class DreamerEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        if len(env.observation_space.shape) == 3:
            env = gym.wrappers.ResizeObservation(env, (64, 64))
            super().__init__(env)
            self.is_image_obs = True
            shape = (env.observation_space.shape[-1], 64, 64)
            self.observation_space = gym.spaces.Box(
                low=-0.5, high=0.5, shape=shape, dtype=np.float32
            )
        else:
            super().__init__(env)
            self.is_image_obs = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_image_obs:
            obs = np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0 - 0.5
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_image_obs:
            obs = np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0 - 0.5
        return obs, info
import torch.optim as optim
import cv2
from torch.cuda.amp import autocast, GradScaler

from dreamer_v3 import DreamerV3Agent, count_parameters


# ─────────────────────────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """Stores a single episode of experience."""
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    continues: List[float] = field(default_factory=list)


class ReplayBuffer:
    """Simple episode-based replay buffer.
    
    Stores complete episodes and samples fixed-length sub-sequences
    for training. This is important for DreamerV3 because the RSSM
    needs sequential context to learn temporal dynamics.
    """
    
    def __init__(self, capacity: int = 1000):
        self.episodes: deque = deque(maxlen=capacity)
    
    def add_episode(self, episode: Episode):
        """Store a completed episode."""
        if len(episode.obs) > 1:
            self.episodes.append(episode)
    
    def sample(self, batch_size: int, seq_len: int,
               device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Sample random sub-sequences from stored episodes.
        
        Returns:
            obs:       (B, T, obs_dim)
            actions:   (B, T, action_dim)
            rewards:   (B, T)
            continues: (B, T)
        """
        obs_batch, act_batch, rew_batch, cont_batch = [], [], [], []
        
        for _ in range(batch_size):
            # Pick a random episode
            ep = random.choice(self.episodes)
            max_start = max(0, len(ep.obs) - seq_len - 1)
            start = random.randint(0, max_start)
            end = min(start + seq_len, len(ep.obs) - 1)
            actual_len = end - start
            
            obs_seq = np.array(ep.obs[start:end])
            act_seq = np.array(ep.actions[start:end])
            rew_seq = np.array(ep.rewards[start:end])
            cont_seq = np.array(ep.continues[start:end])
            
            # Pad if needed
            if actual_len < seq_len:
                pad_len = seq_len - actual_len
                obs_seq = np.concatenate([obs_seq, np.zeros((pad_len, *obs_seq.shape[1:]))])
                act_seq = np.concatenate([act_seq, np.zeros((pad_len, *act_seq.shape[1:]))])
                rew_seq = np.concatenate([rew_seq, np.zeros(pad_len)])
                cont_seq = np.concatenate([cont_seq, np.zeros(pad_len)])
            
            obs_batch.append(obs_seq)
            act_batch.append(act_seq)
            rew_batch.append(rew_seq)
            cont_batch.append(cont_seq)
        
        return (
            torch.tensor(np.array(obs_batch), dtype=torch.float32, device=device),
            torch.tensor(np.array(act_batch), dtype=torch.float32, device=device),
            torch.tensor(np.array(rew_batch), dtype=torch.float32, device=device),
            torch.tensor(np.array(cont_batch), dtype=torch.float32, device=device),
        )
    
    def __len__(self):
        return len(self.episodes)


# ─────────────────────────────────────────────────────────────
#  Video Recording
# ─────────────────────────────────────────────────────────────

def record_video_episode(agent: DreamerV3Agent, env: gym.Env, device: torch.device,
                        discrete: bool, action_dim: int, video_path: str,
                        fps: int = 30) -> Tuple[float, List[np.ndarray]]:
    """Record a single episode to video and return the episode return.
    
    Args:
        agent:      DreamerV3Agent instance
        env:        Gymnasium environment
        device:     torch device
        discrete:   Whether action space is discrete
        action_dim: Action dimension
        video_path: Path to save the video file
        fps:        Frames per second for video
        
    Returns:
        ep_return:  Total episode return
        frames:     List of recorded frames
    """
    obs, _ = env.reset()
    state = agent.get_initial_state(1)
    prev_action = torch.zeros(1, action_dim, device=device)
    ep_return = 0.0
    frames = []
    
    done = False
    step = 0
    while not done:
        # Render and capture frame
        frame = env.render()
        if frame is not None:
            # Ensure frame is uint8 RGB
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frames.append(frame)
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32,
                                  device=device).unsqueeze(0)
        
        with torch.no_grad():
            action_encoded, state = agent.policy(
                obs_tensor, state, prev_action, training=False
            )
        
        if discrete:
            action = action_encoded.argmax(dim=-1).item()
            action_onehot = action_encoded.squeeze(0).cpu().numpy()
        else:
            action = action_encoded.squeeze(0).cpu().numpy()
            action_onehot = action
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        obs = next_obs
        prev_action = action_encoded
        ep_return += reward
        step += 1
    
    # Save video if frames were captured
    if frames:
        save_frames_to_video(frames, video_path, fps)
    else:
        # Fallback: Create a simple test frame if no frames were captured
        print(f"  ⚠️  Warning: No frames captured. Creating placeholder video...")
        h, w = 480, 640
        test_frames = [np.ones((h, w, 3), dtype=np.uint8) * i for i in range(30, 100, 10)]
        save_frames_to_video(test_frames, video_path, fps)
    
    return ep_return, frames


def record_video_with_visualization(agent: DreamerV3Agent, env: gym.Env, device: torch.device,
                                   discrete: bool, action_dim: int, video_path: str,
                                   env_name: str, fps: int = 30) -> Tuple[float, List[np.ndarray]]:
    """Record episode with visualization frames showing stats and agent behavior.
    
    Args:
        agent:      DreamerV3Agent instance
        env:        Gymnasium environment
        device:     torch device
        discrete:   Whether action space is discrete
        action_dim: Action dimension
        video_path: Path to save the video file
        env_name:   Name of environment for display
        fps:        Frames per second for video
        
    Returns:
        ep_return:  Total episode return
        frames:     List of visualization frames
    """
    obs, _ = env.reset()
    state = agent.get_initial_state(1)
    prev_action = torch.zeros(1, action_dim, device=device)
    ep_return = 0.0
    frames = []
    
    done = False
    step = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32,
                                  device=device).unsqueeze(0)
        
        with torch.no_grad():
            action_encoded, state = agent.policy(
                obs_tensor, state, prev_action, training=False
            )
        
        if discrete:
            action = action_encoded.argmax(dim=-1).item()
            action_onehot = action_encoded.squeeze(0).cpu().numpy()
        else:
            action = action_encoded.squeeze(0).cpu().numpy()
            action_onehot = action
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Create visualization frame
        vis_frame = create_visualization_frame(obs, action, reward, step, env_name)
        frames.append(vis_frame)
        
        obs = next_obs
        prev_action = action_encoded
        ep_return += reward
        step += 1
    
    # Save video
    save_frames_to_video(frames, video_path, fps)
    
    return ep_return, frames


def create_visualization_frame(obs: np.ndarray, action: int or float, reward: float, 
                              step: int, env_name: str, width: int = 640, height: int = 480) -> np.ndarray:
    """Create a visualization frame showing observation, action, and reward.
    
    Args:
        obs:       Observation vector
        action:    Action taken
        reward:    Reward received
        step:      Step number
        env_name:  Environment name
        width:     Frame width
        height:    Frame height
        
    Returns:
        Visualization frame as RGB numpy array
    """
    frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add text using simple ASCII rendering
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (20, 20, 20)  # Dark text
    thickness = 2
    
    y_offset = 50
    cv2.putText(frame, f"Environment: {env_name}", (30, y_offset), font, font_scale, color, thickness)
    cv2.putText(frame, f"Step: {step}", (30, y_offset + 40), font, font_scale, color, thickness)
    cv2.putText(frame, f"Action: {action}", (30, y_offset + 80), font, font_scale, color, thickness)
    cv2.putText(frame, f"Reward: {reward:.3f}", (30, y_offset + 120), font, font_scale, color, thickness)
    
    # Show observation stats
    cv2.putText(frame, f"Obs Mean: {np.mean(obs):.3f}", (30, y_offset + 160), font, font_scale, color, thickness)
    cv2.putText(frame, f"Obs Std: {np.std(obs):.3f}", (30, y_offset + 200), font, font_scale, color, thickness)
    
    # Add progress bar
    bar_y = height - 80
    bar_width = 500
    bar_x = 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (50, 50, 50), 2)
    progress = min(step / 500, 1.0)  # Assuming 500 steps max
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 30), (50, 200, 50), -1)
    cv2.putText(frame, f"Progress: {progress*100:.0f}%", (bar_x + 150, bar_y + 50), font, 0.6, color, 1)
    
    return frame


def save_frames_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """Save a list of frames to an MP4 video file.
    
    Args:
        frames:     List of RGB numpy arrays (H, W, 3)
        output_path: Path to save the video
        fps:        Frames per second
    """
    if not frames:
        print(f"  ⚠️  No frames to save to {output_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    height, width = frames[0].shape[:2]
    
    # Use MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"  ❌ Failed to open VideoWriter for {output_path}")
        return
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"  ✅ Video saved: {output_path}")



def train(args):
    """Main training loop."""
    
    print("=" * 65)
    print("  🌙 Mini DreamerV3 — Training")
    print("=" * 65)
    
    # ── Setup ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:      {device}")
    
    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"  GPU Name:    {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    print(f"  Environment: {args.env}")
    
    env = DreamerEnvWrapper(gym.make(args.env))
    obs_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
    else:
        action_dim = env.action_space.shape[0]
        discrete = False
    
    print(f"  Obs dim:     {obs_dim}")
    print(f"  Action dim:  {action_dim} ({'discrete' if discrete else 'continuous'})")
    print(f"  Mixed Prec:  {'enabled' if args.use_amp else 'disabled'}")
    
    # ── Agent ────────────────────────────────────────────
    agent = DreamerV3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        discrete_actions=discrete,
        image_obs=env.is_image_obs,
        deter_dim=args.deter_dim,
        stoch_dim=args.stoch_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        imagination_horizon=args.imagination_horizon,
        gamma=args.gamma,
        device=str(device),
    )
    print(f"  Model:       {count_parameters(agent)}")
    
    # ── Optimizers ───────────────────────────────────────
    wm_params = (
        list(agent.encoder.parameters()) +
        list(agent.rssm.parameters()) +
        list(agent.decoder.parameters()) +
        list(agent.reward_pred.parameters()) +
        list(agent.continue_pred.parameters())
    )
    wm_optimizer = optim.Adam(wm_params, lr=args.lr, eps=1e-8)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.lr * 0.3, eps=1e-8)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.lr, eps=1e-8)
    
    # ── Mixed Precision Scaler ───────────────────────────
    scaler = GradScaler() if args.use_amp else None
    
    # ── Replay Buffer ───────────────────────────────────
    buffer = ReplayBuffer(capacity=args.buffer_size)
    
    # ── Metrics ──────────────────────────────────────────
    episode_returns = []
    best_return = -float("inf")
    
    print(f"\n{'─' * 65}")
    print(f"  {'Step':>8}  {'Episode':>7}  {'Return':>8}  {'WM Loss':>9}  "
          f"{'Actor':>8}  {'Critic':>8}")
    print(f"{'─' * 65}")
    
    # ── Collect Initial Experience ───────────────────────
    total_steps = 0
    episode_count = 0
    
    while total_steps < args.total_steps:
        # ── Collect one episode ──────────────────────────
        obs, _ = env.reset()
        episode = Episode()
        state = agent.get_initial_state(1)
        prev_action = torch.zeros(1, action_dim, device=device)
        ep_return = 0.0
        
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32,
                                      device=device).unsqueeze(0)
            
            with torch.no_grad():
                action_encoded, state = agent.policy(
                    obs_tensor, state, prev_action, training=True
                )
            
            if discrete:
                action = action_encoded.argmax(dim=-1).item()
                action_onehot = action_encoded.squeeze(0).cpu().numpy()
            else:
                action = action_encoded.squeeze(0).cpu().numpy()
                action_onehot = action
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode.obs.append(obs)
            episode.actions.append(action_onehot)
            episode.rewards.append(reward)
            episode.continues.append(0.0 if terminated else 1.0)
            
            obs = next_obs
            prev_action = action_encoded
            ep_return += reward
            total_steps += 1
        
        # Add terminal observation
        episode.obs.append(obs)
        buffer.add_episode(episode)
        episode_returns.append(ep_return)
        episode_count += 1
        
        if ep_return > best_return:
            best_return = ep_return
        
        # ── Train ────────────────────────────────────────
        wm_loss_val = 0.0
        actor_loss_val = 0.0
        critic_loss_val = 0.0
        
        if len(buffer) >= args.min_episodes:
            for _ in range(args.train_ratio):
                # Sample batch
                obs_b, act_b, rew_b, cont_b = buffer.sample(
                    args.batch_size, args.seq_len, device
                )
                
                # Train world model with mixed precision
                wm_optimizer.zero_grad()
                with autocast(enabled=args.use_amp):
                    wm_loss, wm_metrics = agent.world_model_loss(
                        obs_b, act_b, rew_b, cont_b
                    )
                if args.use_amp:
                    scaler.scale(wm_loss).backward()
                    scaler.unscale_(wm_optimizer)
                else:
                    wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(wm_params, args.grad_clip)
                if args.use_amp:
                    scaler.step(wm_optimizer)
                else:
                    wm_optimizer.step()
                wm_loss_val = wm_metrics["total_wm_loss"]
                
                # Imagination-based actor-critic training
                with torch.no_grad():
                    B = obs_b.shape[0]
                    init_state = agent.get_initial_state(B)
                    embeds = agent.encoder(obs_b[:, 0])
                    start_state, _ = agent.rssm.observe_step(
                        init_state, act_b[:, 0], embeds
                    )
                
                with autocast(enabled=args.use_amp):
                    actor_loss, critic_loss, ac_metrics = agent.actor_critic_loss(
                        start_state
                    )
                
                # Update actor
                actor_optimizer.zero_grad()
                if args.use_amp:
                    scaler.scale(actor_loss).backward()
                    scaler.unscale_(actor_optimizer)
                else:
                    actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),
                                               args.grad_clip)
                if args.use_amp:
                    scaler.step(actor_optimizer)
                else:
                    actor_optimizer.step()
                actor_loss_val = ac_metrics["actor_loss"]
                
                # Update critic
                critic_optimizer.zero_grad()
                if args.use_amp:
                    scaler.scale(critic_loss).backward()
                    scaler.unscale_(critic_optimizer)
                else:
                    critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in agent.critic.parameters() if p.requires_grad],
                    args.grad_clip
                )
                if args.use_amp:
                    scaler.step(critic_optimizer)
                else:
                    critic_optimizer.step()
                critic_loss_val = ac_metrics["critic_loss"]
                
                # Update scaler for next iteration
                if args.use_amp:
                    scaler.update()
                
                # Update slow target
                agent.critic.update_target()
        
        # ── Logging ──────────────────────────────────────
        if episode_count % args.log_every == 0:
            avg_return = np.mean(episode_returns[-args.log_every:])
            print(f"  {total_steps:>8d}  {episode_count:>7d}  "
                  f"{avg_return:>8.1f}  {wm_loss_val:>9.4f}  "
                  f"{actor_loss_val:>8.4f}  {critic_loss_val:>8.4f}")
        
        # ── Video Recording ──────────────────────────────
        if args.record_video and episode_count % args.record_every == 0 and episode_count > 0:
            print(f"\n  🎥 Recording evaluation videos (episode {episode_count})...")
            video_subdir = os.path.join(args.video_dir, f"episode_{episode_count}")
            os.makedirs(video_subdir, exist_ok=True)
            
            for eval_i in range(args.num_eval_episodes):
                video_path = os.path.join(video_subdir, f"eval_{eval_i:03d}.mp4")
                eval_env = DreamerEnvWrapper(gym.make(args.env, render_mode="rgb_array"))
                eval_return, _ = record_video_episode(
                    agent, eval_env, device, discrete, action_dim, video_path
                )
                eval_env.close()
                print(f"    Episode {eval_i}: return = {eval_return:.1f}")
            print()
        
        # ── Checkpoint ───────────────────────────────────
        if episode_count % args.save_every == 0 and episode_count > 0:
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "wm_optimizer": wm_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "episode": episode_count,
                "total_steps": total_steps,
            }, f"checkpoint_{args.env}.pt")
            print(f"  💾 Checkpoint saved (episode {episode_count})")
    
    # ── Final Summary ────────────────────────────────────
    env.close()
    avg_final = np.mean(episode_returns[-50:]) if episode_returns else 0
    
    print(f"\n{'=' * 65}")
    print(f"  Training Complete!")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Episodes:      {episode_count:,}")
    print(f"  Best return:   {best_return:.1f}")
    print(f"  Avg (last 50): {avg_final:.1f}")
    
    # GPU Memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  Peak GPU Mem:  {peak_memory:.1f} GB")
        torch.cuda.empty_cache()
    print(f"{'=' * 65}")
    
    # Save final model
    torch.save(agent.state_dict(), f"dreamer_v3_{args.env}.pt")
    print(f"  💾 Final model saved: dreamer_v3_{args.env}.pt")
    
    # ── Record Final Videos ──────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  🎬 Recording Final Agent Performance Videos")
    print(f"{'=' * 65}")
    
    final_video_dir = os.path.join(args.video_dir, "final_performance")
    os.makedirs(final_video_dir, exist_ok=True)
    
    final_returns = []
    for eval_i in range(args.num_final_videos):
        video_path = os.path.join(final_video_dir, f"final_{eval_i:03d}.mp4")
        eval_env = DreamerEnvWrapper(gym.make(args.env, render_mode="rgb_array"))
        
        # Try rendering with environment first, fallback to visualization
        eval_return, frames = record_video_episode(
            agent, eval_env, device, discrete, action_dim, video_path, fps=30
        )
        
        # If no frames captured, use visualization approach
        if not frames or len(frames) < 10:
            print(f"    → Retrying with visualization mode...")
            eval_env.close()
            eval_env = DreamerEnvWrapper(gym.make(args.env))
            video_path_vis = video_path.replace(".mp4", "_vis.mp4")
            eval_return, _ = record_video_with_visualization(
                agent, eval_env, device, discrete, action_dim, video_path_vis, args.env, fps=30
            )
            video_path = video_path_vis
        
        eval_env.close()
        final_returns.append(eval_return)
        print(f"  ✓ Episode {eval_i}: return = {eval_return:.1f}")
    
    avg_final_return = np.mean(final_returns)
    print(f"\n  📊 Final Videos Average Return: {avg_final_return:.1f}")
    print(f"  📁 Saved to: {final_video_dir}/")
    print(f"{'=' * 65}\n")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mini DreamerV3 on Gymnasium environments"
    )
    
    # Environment
    parser.add_argument("--env", type=str, default="CarRacing-v3",
                        help="Gymnasium environment ID (e.g. CarRacing-v3 for Physical AI)")
    parser.add_argument("--total-steps", type=int, default=50_000,
                        help="Total environment steps")
    
    # Model architecture
    parser.add_argument("--deter-dim", type=int, default=256)
    parser.add_argument("--stoch-dim", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--imagination-horizon", type=int, default=15)
    
    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--train-ratio", type=int, default=2,
                        help="Training iterations per episode")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--buffer-size", type=int, default=500)
    parser.add_argument("--min-episodes", type=int, default=5,
                        help="Min episodes before training starts")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    # Video Recording
    parser.add_argument("--record-video", action="store_true",
                        help="Enable video recording during training")
    parser.add_argument("--record-every", type=int, default=100,
                        help="Record video every N episodes")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")
    parser.add_argument("--num-eval-episodes", type=int, default=1,
                        help="Number of evaluation episodes to record")
    parser.add_argument("--num-final-videos", type=int, default=3,
                        help="Number of final performance videos to record after training")
    
    # GPU/Performance
    parser.add_argument("--use-amp", action="store_true",
                        help="Use mixed precision training (faster GPU training)")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    train(args)
