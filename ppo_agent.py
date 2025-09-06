# ppo_train_improved.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import re
from collections import deque

# Import your custom modules
from airsim_env import Env
import config_ppo as config

# ----------------- Utility helpers -----------------
class RunningMeanStd:
"""Simple running mean/std estimator for low-dim vectors (Welford online)."""
def __init__(self, shape=(), epsilon=1e-4):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = epsilon

def update(self, x):
    x = np.array(x, dtype=np.float64)
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    batch_count = x.shape[0] if x.ndim > 0 else 1.0

    # Welford-style update
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * (batch_count / tot_count)
    m_a = self.var * (self.count)
    m_b = batch_var * (batch_count)
    M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
    new_var = M2 / (tot_count)

    self.mean = new_mean
    self.var = np.maximum(new_var, 1e-6)
    self.count = tot_count

def normalize(self, x):
    return (np.array(x, dtype=np.float32) - self.mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8)


# --- Helper function for plotting and saving stats ---
def save_and_plot_metrics(episode, scores, actor_losses, critic_losses, steps, y_positions):
num_episodes = episode + 1

# Ensure all arrays have the same length by padding with NaN if necessary
max_len = max(len(scores), len(actor_losses), len(critic_losses), len(steps), len(y_positions))
actual_episodes = min(num_episodes, max_len)

# Pad arrays to match the actual number of episodes
def pad_array(arr, target_len):
    if len(arr) < target_len:
        return arr + [np.nan] * (target_len - len(arr))
    return arr[:target_len]

scores_padded = pad_array(scores, actual_episodes)
actor_losses_padded = pad_array(actor_losses, actual_episodes)
critic_losses_padded = pad_array(critic_losses, actual_episodes)
steps_padded = pad_array(steps, actual_episodes)
y_positions_padded = pad_array(y_positions, actual_episodes)

df = pd.DataFrame({
    'episode': range(1, actual_episodes + 1),
    'score': scores_padded,
    'actor_loss': actor_losses_padded,
    'critic_loss': critic_losses_padded,
    'steps': steps_padded,
    'y_position': y_positions_padded
})
# Ensure the save directory exists
if not os.path.exists(config.SAVE_PATH):
    os.makedirs(config.SAVE_PATH)
df.to_csv(os.path.join(config.SAVE_PATH, 'ppo_stat.csv'), index=False)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('PPO Training Progress', fontsize=16)
axes[0, 0].plot(scores, label='Total Reward')
axes[0, 0].set_title('Total Reward per Episode')
axes[0, 0].grid(True)
axes[0, 1].plot(actor_losses, label='Actor Loss', color='orange')
axes[0, 1].set_title('Average Actor Loss per Episode')
axes[0, 1].grid(True)
axes[1, 0].plot(critic_losses, label='Critic Loss', color='green')
axes[1, 0].set_title('Average Critic Loss per Episode')
axes[1, 0].grid(True)
axes[1, 1].plot(steps, label='Steps', color='red')
axes[1, 1].set_title('Steps per Episode')
axes[1, 1].grid(True)
axes[2, 0].plot(y_positions, label='Y Position', color='purple')
axes[2, 0].set_title('Final Y Position per Episode')
axes[2, 0].grid(True)
axes[2, 1].axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
if not os.path.exists(config.GRAPH_PATH):
    os.makedirs(config.GRAPH_PATH)
plt.savefig(os.path.join(config.GRAPH_PATH, 'ppo_training_progress.png'))
plt.close()
print(f"--- Metrics and plots SAVED for episode {episode + 1} ---")


# --- New robust image processing function with smoothing support & normalization ---
def _process_image(response, depth_max=100.0, clip_min=0.0, clip_max=None):
"""
- response: single airsim image response (DepthPlanner float)
- depth_max: value to normalize against (meters). taken from config.DEPTH_MAX if available.
Returns a (H,W,C) float32 image normalized to roughly [-1,1] or [0,1] depending on config.
"""
if not response:
    return np.zeros(config.STATE_IMAGE_SHAPE, dtype=np.float32)

# image_data_float is expected for DepthPlanner; handle missing or malformed cases
if not hasattr(response, 'image_data_float') or response.image_data_float is None:
    return np.zeros(config.STATE_IMAGE_SHAPE, dtype=np.float32)

if not hasattr(response, 'height') or not hasattr(response, 'width') or response.height <= 0 or response.width <= 0:
    print("Warning: Invalid image dims; returning zeros.")
    return np.zeros(config.STATE_IMAGE_SHAPE, dtype=np.float32)

img_raw = np.array(response.image_data_float, dtype=np.float32)

expected_size = response.height * response.width
if img_raw.size != expected_size:
    # If DepthPlanner returns extra metadata, try to slice; otherwise return zeros
    if img_raw.size > expected_size:
        img_raw = img_raw[:expected_size]
    else:
        print(f"Warning: Image size mismatch. expected {expected_size}, got {img_raw.size}. Returning zeros.")
        return np.zeros(config.STATE_IMAGE_SHAPE, dtype=np.float32)

try:
    img_reshaped = img_raw.reshape(response.height, response.width, 1)
    # resize to expected shape
    target_h, target_w, target_c = config.STATE_IMAGE_SHAPE
    img_resized = tf.image.resize(img_reshaped, [target_h, target_w]).numpy().astype(np.float32)

    # Normalize with a fixed cap (prefer config.DEPTH_MAX if present)
    # If config provides DEPTH_MAX, use it; otherwise use passed depth_max
    depth_cap = getattr(config, 'DEPTH_MAX', depth_max)
    if depth_cap <= 0:
        depth_cap = depth_max

    # clip to a reasonable positive range and scale to [0,1]
    if clip_max is None:
        clip_max_effective = depth_cap
    else:
        clip_max_effective = clip_max

    img_clipped = np.clip(img_resized, clip_min, clip_max_effective)
    img_norm = img_clipped / float(clip_max_effective + 1e-8)

    # Optionally scale to zero-mean if desired; for now keep 0-1 since conv nets often prefer that
    return img_norm.astype(np.float32)
except Exception as e:
    print(f"Warning: Error while processing image: {e}. Returning zeros.")
    return np.zeros(config.STATE_IMAGE_SHAPE, dtype=np.float32)


# ---------------- PPO Agent Implementation ----------------
class PPOAgent:
def __init__(self, state_image_shape, state_velocity_shape, action_dim, action_bound):
    self.action_dim = action_dim
    self.action_bound = action_bound
    self.optimizer = Adam(learning_rate=config.LEARNING_RATE, clipnorm=config.MAX_GRAD_NORM)
    self.log_std = tf.Variable(np.zeros(action_dim, dtype=np.float32), trainable=True, name='log_std')
    self.model = self._build_actor_critic(state_image_shape, state_velocity_shape)

    # small convenience: running action mean for diagnostics (not used for training)
    self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

def _build_actor_critic(self, image_shape, velocity_shape):
    image_input = Input(shape=image_shape, dtype=tf.float32)
    conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_input)
    conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
    conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
    image_flatten = Flatten()(conv)

    velocity_input = Input(shape=velocity_shape, dtype=tf.float32)
    concat = Concatenate()([image_flatten, velocity_input])
    shared = Dense(512, activation='relu')(concat)

    actor_dense = Dense(256, activation='relu')(shared)
    mu = Dense(self.action_dim, activation='tanh')(actor_dense)
    # scale to action bounds (assume symmetric bounds)
    mu = tf.keras.layers.Lambda(lambda x: x * self.action_bound)(mu)

    critic_dense = Dense(256, activation='relu')(shared)
    value = Dense(1)(critic_dense)

    return Model(inputs=[image_input, velocity_input], outputs=[mu, value])

def select_action(self, state):
    """
    state: [image (H,W,C), velocity (D,)]
    returns: action (applied), log_prob (for training), value (state value)
    """
    image_state = tf.expand_dims(tf.convert_to_tensor(state[0], dtype=tf.float32), 0)  # (1,H,W,C)
    velocity_state = tf.expand_dims(tf.convert_to_tensor(state[1], dtype=tf.float32), 0)  # (1,D)
    mu, value = self.model([image_state, velocity_state], training=False)

    # compute std, make sure shape broadcasts with mu
    std = tf.exp(self.log_std)  # shape (action_dim,)
    # ensure std has batch dim for broadcasting
    std_b = tf.reshape(std, (1, self.action_dim))

    dist = tfp.distributions.Normal(loc=mu, scale=std_b)
    sampled = dist.sample()
    log_prob = tf.reduce_sum(dist.log_prob(sampled), axis=-1)

    # clip action to bounds and convert to numpy
    action = tf.clip_by_value(sampled, -self.action_bound, self.action_bound)
    action_np = action[0].numpy().astype(np.float32)
    log_prob_np = log_prob[0].numpy().astype(np.float32)
    value_np = value[0][0].numpy().astype(np.float32)

    # update prev_action for diagnostics
    self.prev_action = action_np
    return action_np, log_prob_np, value_np

def train(self, states, actions, log_probs_old, returns, advantages):
    """
    states: [images_array, velocities_array]
    actions: (N, action_dim)
    log_probs_old: (N,)
    returns: (N,)
    advantages: (N,)
    """
    actor_losses, critic_losses = [], []
    N = len(states[0])
    # convert to tensors once here
    images_all = tf.convert_to_tensor(states[0], dtype=tf.float32)
    vels_all = tf.convert_to_tensor(states[1], dtype=tf.float32)
    actions_all = tf.convert_to_tensor(actions, dtype=tf.float32)
    logp_old_all = tf.convert_to_tensor(log_probs_old, dtype=tf.float32)
    returns_all = tf.convert_to_tensor(returns, dtype=tf.float32)
    adv_all = tf.convert_to_tensor(advantages, dtype=tf.float32)

    for _ in range(config.PPO_EPOCHS):
        # shuffle indices deterministically for each epoch improves mixing
        idxs = np.random.permutation(N)
        for start in range(0, N, config.BATCH_SIZE):
            end = start + config.BATCH_SIZE
            batch_idx = idxs[start:end]

            batch_images = tf.gather(images_all, batch_idx)
            batch_vels = tf.gather(vels_all, batch_idx)
            batch_actions = tf.gather(actions_all, batch_idx)
            batch_logp_old = tf.gather(logp_old_all, batch_idx)
            batch_returns = tf.gather(returns_all, batch_idx)
            batch_adv = tf.gather(adv_all, batch_idx)

            with tf.GradientTape() as tape:
                mu, values = self.model([batch_images, batch_vels], training=True)
                values = tf.squeeze(values, axis=-1)

                # critic loss (MSE)
                critic_loss = tf.reduce_mean(tf.square(batch_returns - values))
                # distribution
                std = tf.exp(self.log_std)
                std_b = tf.reshape(std, (1, self.action_dim))
                dist = tfp.distributions.Normal(loc=mu, scale=std_b)

                logp_new = tf.reduce_sum(dist.log_prob(batch_actions), axis=-1)
                ratio = tf.exp(logp_new - batch_logp_old)
                surr1 = ratio * batch_adv
                surr2 = tf.clip_by_value(ratio, 1.0 - config.PPO_CLIP, 1.0 + config.PPO_CLIP) * batch_adv
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                entropy_loss = -tf.reduce_mean(dist.entropy())

                total_loss = actor_loss + config.CRITIC_COEF * critic_loss + config.ENTROPY_COEF * entropy_loss

            trainable_vars = self.model.trainable_variables + [self.log_std]
            grads = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))

            actor_losses.append(actor_loss.numpy().astype(np.float32))
            critic_losses.append(critic_loss.numpy().astype(np.float32))

    # return average losses
    return float(np.mean(actor_losses) if actor_losses else 0.0), float(np.mean(critic_losses) if critic_losses else 0.0)


def compute_advantages_and_returns(rewards, values, dones, next_value):
values = np.append(values, next_value)
advantages = np.zeros_like(rewards, dtype=np.float32)
last_gae_lam = 0
for t in reversed(range(len(rewards))):
    mask = 1.0 - float(dones[t])
    delta = rewards[t] + config.GAMMA * values[t + 1] * mask - values[t]
    advantages[t] = last_gae_lam = delta + config.GAMMA * config.GAE_LAMBDA * mask * last_gae_lam
returns = advantages + values[:-1]
# normalize advantages here (already in your code but keep it stable)
adv_mean = np.mean(advantages) if advantages.size > 0 else 0.0
adv_std = np.std(advantages) if advantages.size > 0 else 1.0
if adv_std < 1e-8:
    adv_std = 1.0
advantages = (advantages - adv_mean) / (adv_std + 1e-8)
return returns, advantages


# ----------------- Main training loop -----------------
if __name__ == '__main__':
# set seeds for reproducibility where possible
np.random.seed(getattr(config, 'SEED', 0))
tf.random.set_seed(getattr(config, 'SEED', 0))

env = Env()
agent = PPOAgent(config.STATE_IMAGE_SHAPE, config.STATE_VELOCITY_SHAPE, config.ACTION_DIM, config.ACTION_BOUND)
summary_writer = tf.summary.create_file_writer(config.TENSORBOARD_PATH)
start_episode = 0
all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = [], [], [], [], []

recent_scores = deque(maxlen=100)
recent_steps = deque(maxlen=100)

# running normalizer for velocity (helps scale velocity inputs)
vel_rms = RunningMeanStd(shape=config.STATE_VELOCITY_SHAPE)

# image smoothing (EMA)
image_ema_alpha = getattr(config, 'IMAGE_EMA_ALPHA', 0.6)  # higher alpha -> smoother (0..1)

# depth normalization cap
depth_cap = getattr(config, 'DEPTH_MAX', 100.0)

if os.path.exists(config.MODEL_PATH):
    model_files = [f for f in os.listdir(config.MODEL_PATH) if f.startswith('ppo_model_ep')]
    if model_files:
        latest_episode = max([int(re.search(r'(\d+)', f).group(1)) for f in model_files])
        model_path = os.path.join(config.MODEL_PATH, f"ppo_model_ep{latest_episode}.weights.h5")
        if os.path.exists(model_path):
            print(f"Resuming training from episode {latest_episode}...")
            dummy_img = np.zeros((1, *config.STATE_IMAGE_SHAPE), dtype=np.float32)
            dummy_vel = np.zeros((1, *config.STATE_VELOCITY_SHAPE), dtype=np.float32)
            agent.model([dummy_img, dummy_vel])  # build
            agent.model.load_weights(model_path)
            start_episode = latest_episode
            stats_path = os.path.join(config.SAVE_PATH, 'ppo_stat.csv')
            if os.path.exists(stats_path):
                df = pd.read_csv(stats_path)
                all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = \
                    df['score'].tolist(), df['actor_loss'].tolist(), df['critic_loss'].tolist(), \
                    df['steps'].tolist(), df['y_position'].tolist()
                recent_scores.extend(all_scores[-100:])
                recent_steps.extend(all_steps[-100:])
                print("Loaded previous training statistics.")

episode_count = start_episode
total_timesteps = 0

# helper: apply image smoothing EMA
prev_image_ema = None
prev_vel_ema = None

while total_timesteps < config.TOTAL_TIMESTEPS and episode_count < config.MAX_EPISODES:
    states_img, states_vel, actions, rewards, dones, log_probs, values = [], [], [], [], [], [], []
    state = env.reset()
    current_episode_reward = 0.0

    # reset EMAs for a fresh episode
    prev_image_ema = None
    prev_vel_ema = None

    for t in range(config.UPDATE_TIMESTEPS):
        total_timesteps += 1

        # process image
        # state[0] is responses (list), but your env returns [responses, vel] where responses is a list
        response = state[0][0] if isinstance(state[0], (list, tuple)) else state[0]
        image_state = _process_image(response, depth_max=depth_cap)
        velocity_state = np.array(state[1], dtype=np.float32)

        # initialize EMA values if needed
        if prev_image_ema is None:
            prev_image_ema = image_state.copy()
        else:
            prev_image_ema = image_ema_alpha * prev_image_ema + (1.0 - image_ema_alpha) * image_state

        if prev_vel_ema is None:
            prev_vel_ema = velocity_state.copy()
        else:
            prev_vel_ema = image_ema_alpha * prev_vel_ema + (1.0 - image_ema_alpha) * velocity_state

        # normalize velocity with running mean/std
        vel_rms.update(prev_vel_ema.reshape((1, -1)))
        vel_norm = vel_rms.normalize(prev_vel_ema)

        # final processed state (same shape as before — we didn't change config.STATE_IMAGE_SHAPE)
        state_processed = [prev_image_ema.astype(np.float32), vel_norm.astype(np.float32)]

        # select action
        action, log_prob, value = agent.select_action(state_processed)
        next_state, reward, done, info = env.step(action)

        # store transition data (store processed states that we actually used to sample)
        states_img.append(state_processed[0])
        states_vel.append(state_processed[1])
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        log_probs.append(float(log_prob))
        values.append(float(value))

        state = next_state
        current_episode_reward += float(reward)

        if done:
            episode_steps = info.get('steps', t + 1)
            recent_scores.append(current_episode_reward)
            recent_steps.append(episode_steps)
            avg_reward_100 = np.mean(recent_scores) if len(recent_scores) > 0 else 0.0
            avg_steps_100 = np.mean(recent_steps) if len(recent_steps) > 0 else 0.0

            print("\n" + "=" * 80)
            print(f"|| EPISODE {episode_count + 1} FINISHED! || Status: {info.get('status', 'N/A')}")
            print(f"|| Steps: {episode_steps: <4} | Reward: {current_episode_reward: <8.2f} | Avg Reward (Last 100): {avg_reward_100: <8.2f} ||")
            print("=" * 80 + "\n")

            all_scores.append(current_episode_reward)
            all_steps.append(episode_steps)
            all_y_positions.append(info.get('Y', 0))

            with summary_writer.as_default():
                tf.summary.scalar('Total Reward', current_episode_reward, step=episode_count)
                tf.summary.scalar('Average Reward (Last 100)', avg_reward_100, step=episode_count)
                tf.summary.scalar('Average Steps (Last 100)', avg_steps_100, step=episode_count)

            episode_count += 1

            # Save model & stats according to your existing schedule
            if (episode_count) % config.SAVE_FREQ == 0 and episode_count > start_episode:
                if not os.path.exists(config.MODEL_PATH):
                    os.makedirs(config.MODEL_PATH)
                save_path = os.path.join(config.MODEL_PATH, f"ppo_model_ep{episode_count}.weights.h5")
                agent.model.save_weights(save_path)
                save_and_plot_metrics(episode_count - 1, all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions)
                print(f"--- Model saved at {save_path} ---")

            # start next episode
            state = env.reset()
            current_episode_reward = 0.0

    # compute bootstrap value for last state
    response = state[0][0] if isinstance(state[0], (list, tuple)) else state[0]
    next_image_state = _process_image(response, depth_max=depth_cap)
    # apply same EMA smoothing as during rollout (use previous ema if present)
    if prev_image_ema is None:
        next_image_ema = next_image_state
    else:
        next_image_ema = image_ema_alpha * prev_image_ema + (1.0 - image_ema_alpha) * next_image_state

    next_velocity_state = np.array(state[1], dtype=np.float32)
    if prev_vel_ema is None:
        next_vel_ema = next_velocity_state
    else:
        next_vel_ema = image_ema_alpha * prev_vel_ema + (1.0 - image_ema_alpha) * next_velocity_state

    vel_rms.update(next_vel_ema.reshape((1, -1)))
    next_vel_norm = vel_rms.normalize(next_vel_ema)

    _, _, next_value = agent.select_action([next_image_ema.astype(np.float32), next_vel_norm.astype(np.float32)])

    # prepare arrays for training
    states_img_arr = np.array(states_img, dtype=np.float32)
    states_vel_arr = np.array(states_vel, dtype=np.float32)
    actions_arr = np.array(actions, dtype=np.float32)
    log_probs_arr = np.array(log_probs, dtype=np.float32)
    values_arr = np.array(values, dtype=np.float32)
    returns, advantages = compute_advantages_and_returns(rewards, values_arr, dones, next_value)

    # train agent (keeps same save behavior)
    actor_loss, critic_loss = agent.train([states_img_arr, states_vel_arr],
                                        actions_arr, log_probs_arr, returns, advantages)

    all_actor_losses.append(actor_loss)
    all_critic_losses.append(critic_loss)
    print(f"--- PPO Update Complete. Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f} ---")

    with summary_writer.as_default():
        tf.summary.scalar('Average Actor Loss', actor_loss, step=episode_count)
        tf.summary.scalar('Average Critic Loss', critic_loss, step=episode_count)

env.disconnect()
