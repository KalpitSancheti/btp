# per_buffer.py
"""
Prioritized Experience Replay Buffer for SAC
Based on the paper: "Prioritized Experience Replay" by Schaul et al.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
import random


class SumTree:
    """
    Sum Tree data structure for efficient sampling with priorities
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.pending_idx + self.capacity - 1
        self.data[self.pending_idx] = data
        self.update(idx, priority)

        self.pending_idx += 1
        if self.pending_idx >= self.capacity:
            self.pending_idx = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer for SAC
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        max_priority: float = 1.0,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        
        # PER parameters
        self.alpha = alpha  # How much prioritization is used (0 = uniform, 1 = full prioritization)
        self.beta = beta    # Importance sampling correction (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small constant to avoid zero priorities
        self.max_priority = max_priority
        
        # Sum tree for efficient sampling
        self.tree = SumTree(buffer_size)
        
        # Track which experiences are important for tunnel navigation
        self.collision_bonus = 2.0      # Higher priority for collision experiences
        self.checkpoint_bonus = 1.5     # Higher priority for checkpoint achievements
        self.near_miss_bonus = 1.3      # Higher priority for near-collision experiences
        self.goal_bonus = 3.0          # Highest priority for goal achievements

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict],
    ) -> None:
        """
        Add experience to buffer with priority based on tunnel navigation context
        """
        # Calculate initial priority based on experience type
        priority = self._calculate_initial_priority(reward, done, infos)
        
        # Store in regular buffer
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Add to priority tree
        experience = {
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'done': done,
            'infos': infos
        }
        self.tree.add(priority, experience)

    def _calculate_initial_priority(self, reward: np.ndarray, done: np.ndarray, infos: List[Dict]) -> float:
        """
        Calculate initial priority based on experience importance for tunnel navigation
        """
        base_priority = self.max_priority
        
        for i, info in enumerate(infos):
            # High priority for collisions (learn to avoid)
            if info.get('termination_reason') == 'collision':
                base_priority *= self.collision_bonus
            
            # High priority for goal achievements (learn successful behavior)
            elif info.get('termination_reason') == 'goal_reached':
                base_priority *= self.goal_bonus
            
            # Medium priority for checkpoint achievements
            elif 'new_checkpoints' in info and info['new_checkpoints']:
                base_priority *= self.checkpoint_bonus
            
            # Medium priority for near-misses (close to obstacles)
            elif info.get('min_clear_above') is not None:
                clear_above = info['min_clear_above']
                if clear_above < 1.0:  # Very close to ceiling
                    base_priority *= self.near_miss_bonus
            
            # Higher priority for high absolute rewards (both positive and negative)
            if abs(reward[i]) > 50:  # Significant reward/penalty
                base_priority *= 1.2
        
        return base_priority

    def sample(self, batch_size: int, env: VecNormalize = None):
        """
        Sample batch with prioritized sampling
        """
        # Update beta (importance sampling correction)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample indices based on priorities
        indices = []
        priorities = []
        experiences = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, experience = self.tree.get(s)
            
            indices.append(idx)
            priorities.append(priority)
            experiences.append(experience)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.buffer_size * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Convert experiences to batch format
        batch_obs = np.array([exp['obs'] for exp in experiences])
        batch_next_obs = np.array([exp['next_obs'] for exp in experiences])
        batch_actions = np.array([exp['action'] for exp in experiences])
        batch_rewards = np.array([exp['reward'] for exp in experiences])
        batch_dones = np.array([exp['done'] for exp in experiences])
        
        # Convert to torch tensors
        batch_obs = torch.as_tensor(batch_obs, device=self.device).float()
        batch_next_obs = torch.as_tensor(batch_next_obs, device=self.device).float()
        batch_actions = torch.as_tensor(batch_actions, device=self.device).float()
        batch_rewards = torch.as_tensor(batch_rewards, device=self.device).float()
        batch_dones = torch.as_tensor(batch_dones, device=self.device).float()
        batch_weights = torch.as_tensor(weights, device=self.device).float()
        
        return (
            batch_obs,
            batch_actions,
            batch_next_obs,
            batch_dones,
            batch_rewards,
            batch_weights,
            indices
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities of sampled experiences based on TD errors
        """
        for idx, priority in zip(indices, priorities):
            # Add epsilon to avoid zero priorities and apply alpha
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class PERSACWrapper:
    """
    Wrapper to integrate PER with SAC training
    """
    def __init__(self, model, per_buffer):
        self.model = model
        self.per_buffer = per_buffer
        self.update_count = 0

    def learn_step(self):
        """
        Custom learning step with PER
        """
        if self.per_buffer.size() < self.model.learning_starts:
            return

        # Sample from PER buffer
        batch = self.per_buffer.sample(self.model.batch_size)
        obs, actions, next_obs, dones, rewards, weights, indices = batch

        # Compute TD errors for priority updates
        with torch.no_grad():
            # Get current Q-values
            current_q1, current_q2 = self.model.critic(obs, actions)
            
            # Get target Q-values
            next_actions, next_log_probs = self.model.actor.action_log_prob(next_obs)
            target_q1, target_q2 = self.model.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.model.ent_coef * next_log_probs
            target_q = rewards + (1 - dones) * self.model.gamma * target_q
            
            # Compute TD errors
            td_error1 = abs(current_q1.squeeze() - target_q.squeeze())
            td_error2 = abs(current_q2.squeeze() - target_q.squeeze())
            td_errors = torch.max(td_error1, td_error2).cpu().numpy()

        # Update priorities
        self.per_buffer.update_priorities(indices, td_errors)

        # Perform weighted gradient update
        # (This would require modifying SAC's internal training step to use weights)
        # For now, we'll use the standard SAC update but with prioritized sampling
        
        self.update_count += 1
        return td_errors.mean()


def create_per_sac_model(env, **sac_kwargs):
    """
    Create SAC model with PER buffer
    """
    # Create PER buffer
    per_buffer = PrioritizedReplayBuffer(
        buffer_size=sac_kwargs.get('buffer_size', int(1e6)),
        observation_space=env.observation_space,
        action_space=env.action_space,
        alpha=0.6,      # Prioritization strength
        beta=0.4,       # Importance sampling correction
        beta_increment=0.001,
    )
    
    # Create SAC model (we'll need to modify it to use PER)
    from stable_baselines3 import SAC
    
    model = SAC(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        **sac_kwargs
    )
    
    # Replace the replay buffer with PER buffer
    model.replay_buffer = per_buffer
    
    return model, per_buffer
