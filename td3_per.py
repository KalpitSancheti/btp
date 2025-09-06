# td3_per.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import re
from collections import deque

# Import your custom modules
from airsim_env2 import Env
from PER import PrioritizedReplayBuffer
import config

# --- Robust Image Processing ---
def _process_image(response):
    """
    Robust image -> single-channel float32 array in shape config.STATE_IMAGE_SHAPE.
    Handles both image_data_uint8 (RGB) and image_data_float (depth).
    """
    # safe empty/default
    out_shape = tuple(config.STATE_IMAGE_SHAPE)
    zeros = np.zeros(out_shape, dtype=np.float32)

    if response is None:
        return zeros

    # 1) Prefer uint8 (RGB) if present
    try:
        if hasattr(response, "image_data_uint8") and response.image_data_uint8:
            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            # some AirSim responses include padding; guard with expected size:
            expected = int(response.height) * int(response.width) * 3
            if img.size != expected:
                return zeros
            img = img.reshape(response.height, response.width, 3)
            # convert to grayscale, resize, normalize
            img_tf = tf.image.rgb_to_grayscale(img)
            img_tf = tf.image.resize(img_tf, [out_shape[0], out_shape[1]])
            img_np = img_tf.numpy().astype(np.float32) / 255.0
            return img_np
    except Exception:
        # fall through to float processing or return zeros
        pass

    # 2) Fallback: float image (depth / depthVis) -> single channel
    try:
        if hasattr(response, "image_data_float") and response.image_data_float:
            arr = np.array(response.image_data_float, dtype=np.float32)
            expected = int(response.height) * int(response.width)
            if arr.size != expected:
                return zeros
            arr = arr.reshape(response.height, response.width, 1)
            # resize
            img_tf = tf.image.resize(arr, [out_shape[0], out_shape[1]])
            img_np = img_tf.numpy().astype(np.float32)
            # normalize depth to [0,1] in a numerically stable way
            mn, mx = img_np.min(), img_np.max()
            if mx > mn:
                img_np = (img_np - mn) / (mx - mn)
            else:
                img_np = np.zeros_like(img_np, dtype=np.float32)
            return img_np
    except Exception:
        pass

    # if nothing worked, return zeros
    return zeros


# --- Helper function for plotting and saving stats ---
def save_and_plot_metrics(episode, scores, actor_losses, critic_losses, steps, y_positions):
    df = pd.DataFrame({
        'episode': range(1, len(scores) + 1), 'score': scores, 'actor_loss': actor_losses,
        'critic_loss': critic_losses, 'steps': steps, 'y_position': y_positions
    })
    if not os.path.exists(config.SAVE_PATH): os.makedirs(config.SAVE_PATH)
    df.to_csv(os.path.join(config.SAVE_PATH, 'td3_per_stat.csv'), index=False)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('TD3+PER Training Progress', fontsize=16)

    if len(scores) > 1:
        p05 = np.percentile(scores, 5)
        p95 = np.percentile(scores, 95)
        clipped_scores = np.clip(scores, p05, p95)
    else:
        clipped_scores = scores

    axes[0, 0].plot(clipped_scores, label='Total Reward (Clipped)')
    axes[0, 0].set_title('Total Reward per Episode (Clipped for Clarity)')
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
    if not os.path.exists(config.GRAPH_PATH): os.makedirs(config.GRAPH_PATH)
    plt.savefig(os.path.join(config.GRAPH_PATH, 'td3_per_training_progress.png'))
    plt.close()
    print(f"--- Metrics and plots SAVED for episode {episode + 1} ---")

# --- Ornstein-Uhlenbeck Noise for Exploration ---
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta, self.mean, self.std_dev, self.dt, self.x_initial = theta, mean, std_deviation, dt, x_initial
        self.reset()
    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +
             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x
    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

# --- TD3 Agent Implementation ---
class TD3Agent:
    def __init__(self):
        self.actor = self._build_actor()
        self.critic_1, self.critic_2 = self._build_critic(), self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic_1, self.target_critic_2 = self._build_critic(), self._build_critic()

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = Adam(learning_rate=config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=config.CRITIC_LEARNING_RATE)

        self.buffer = PrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.PER_ALPHA)
        self.noise = OUActionNoise(mean=np.zeros(config.ACTION_DIM), std_deviation=float(config.EXPLORATION_NOISE) * np.ones(config.ACTION_DIM))

    def _build_actor(self):
        image_input = Input(shape=config.STATE_IMAGE_SHAPE)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_input)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        image_flatten = Flatten()(conv)
        velocity_input = Input(shape=config.STATE_VELOCITY_SHAPE)
        concat = Concatenate()([image_flatten, velocity_input])
        dense = Dense(512, activation='relu')(concat) # Increased layer size
        dense = Dense(512, activation='relu')(dense) # Increased layer size
        output = Dense(config.ACTION_DIM, activation='tanh')(dense)
        scaled_output = output * config.ACTION_BOUND
        return Model(inputs=[image_input, velocity_input], outputs=scaled_output)

    def _build_critic(self):
        image_input = Input(shape=config.STATE_IMAGE_SHAPE)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_input)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        image_flatten = Flatten()(conv)
        velocity_input = Input(shape=config.STATE_VELOCITY_SHAPE)
        state_concat = Concatenate()([image_flatten, velocity_input])
        state_dense = Dense(256, activation='relu')(state_concat) # Increased layer size
        action_input = Input(shape=(config.ACTION_DIM,))
        action_dense = Dense(256, activation='relu')(action_input) # Increased layer size
        concat = Concatenate()([state_dense, action_dense])
        dense = Dense(512, activation='relu')(concat) # Increased layer size
        dense = Dense(512, activation='relu')(dense) # Increased layer size
        output = Dense(1)(dense)
        return Model(inputs=[image_input, velocity_input, action_input], outputs=output)

    def select_action(self, state):
        image_state = tf.expand_dims(tf.convert_to_tensor(state[0]), 0)
        velocity_state = tf.expand_dims(tf.convert_to_tensor(state[1]), 0)
        action = self.actor([image_state, velocity_state])[0].numpy()
        action += self.noise()
        return np.clip(action, -config.ACTION_BOUND, config.ACTION_BOUND)

    def train(self, total_steps):
        # Replay buffer warm-up: skip training until buffer has enough samples
        if len(self.buffer) < config.BUFFER_WARMUP:
            if total_steps % 1000 == 0:
                print(f"Warmup: {len(self.buffer)}/{config.BUFFER_WARMUP} transitions collected")
            return None, None

        beta = min(config.PER_BETA_END, config.PER_BETA_START + (total_steps * (config.PER_BETA_END - config.PER_BETA_START) / config.PER_BETA_FRAMES))

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(config.BATCH_SIZE, beta=beta)

        state_images = tf.convert_to_tensor(np.array([s[0] for s in states]), dtype=tf.float32)
        state_velocities = tf.convert_to_tensor(np.array([s[1] for s in states]), dtype=tf.float32)
        next_state_images = tf.convert_to_tensor(np.array([s[0] for s in next_states]), dtype=tf.float32)
        next_state_velocities = tf.convert_to_tensor(np.array([s[1] for s in next_states]), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        actor_loss_val, critic_loss_val = None, None

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor([next_state_images, next_state_velocities])
            noise = tf.clip_by_value(tf.random.normal(shape=tf.shape(target_actions), stddev=config.POLICY_NOISE),
                                    -config.NOISE_CLIP, config.NOISE_CLIP)
            target_actions = tf.clip_by_value(target_actions + noise, -config.ACTION_BOUND, config.ACTION_BOUND)

            target_q1 = self.target_critic_1([next_state_images, next_state_velocities, target_actions])
            target_q2 = self.target_critic_2([next_state_images, next_state_velocities, target_actions])
            target_q = tf.minimum(target_q1, target_q2)

            y = rewards + config.GAMMA * (1 - dones) * tf.squeeze(target_q)

            q1 = self.critic_1([state_images, state_velocities, actions])
            q2 = self.critic_2([state_images, state_velocities, actions])

            td_errors = y - tf.squeeze(q1)
            critic_1_loss = tf.reduce_mean(weights * tf.square(y - tf.squeeze(q1)))
            critic_2_loss = tf.reduce_mean(weights * tf.square(y - tf.squeeze(q2)))
            critic_loss_val = critic_1_loss + critic_2_loss

        critic_params = self.critic_1.trainable_variables + self.critic_2.trainable_variables
        critic_grad = tape.gradient(critic_loss_val, critic_params)
        critic_grad = [g if g is not None else tf.zeros_like(v) for g, v in zip(critic_grad, critic_params)]
        critic_grad, _ = tf.clip_by_global_norm(critic_grad, 40.0)
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic_params))

        self.buffer.update_priorities(indices, np.abs(td_errors.numpy()))

        if total_steps % config.POLICY_UPDATE_FREQUENCY == 0:
            with tf.GradientTape() as tape2:
                new_actions = self.actor([state_images, state_velocities])
                q1_new = self.critic_1([state_images, state_velocities, new_actions])
                actor_loss_val = -tf.reduce_mean(q1_new)

            actor_grad = tape2.gradient(actor_loss_val, self.actor.trainable_variables)
            actor_grad = [g if g is not None else tf.zeros_like(v) for g, v in zip(actor_grad, self.actor.trainable_variables)]
            actor_grad, _ = tf.clip_by_global_norm(actor_grad, 40.0)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            self._soft_update_target_networks()

        del tape
        return actor_loss_val, critic_loss_val


    def _soft_update_target_networks(self):
        for target, main in zip(self.target_actor.weights, self.actor.weights):
            target.assign(config.TAU * main + (1 - config.TAU) * target)
        for target, main in zip(self.target_critic_1.weights, self.critic_1.weights):
            target.assign(config.TAU * main + (1 - config.TAU) * target)
        for target, main in zip(self.target_critic_2.weights, self.critic_2.weights):
            target.assign(config.TAU * main + (1 - config.TAU) * target)

# --- Main Training Loop ---
if __name__ == '__main__':
    env = Env()
    agent = TD3Agent()
    summary_writer = tf.summary.create_file_writer(config.TENSORBOARD_PATH)
    start_episode = 0
    all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = [], [], [], [], []
    recent_scores = deque(maxlen=100)

    current_noise_level = config.EXPLORATION_NOISE

    # --- Resume Logic ---
    if os.path.exists(config.MODEL_PATH):
        model_files = [f for f in os.listdir(config.MODEL_PATH) if f.startswith('actor_ep')]
        if model_files:
            latest_episode = max([int(re.search(r'(\d+)', f).group(1)) for f in model_files])
            actor_path = os.path.join(config.MODEL_PATH, f"actor_ep{latest_episode}.h5")
            if os.path.exists(actor_path):
                print(f"Resuming training from episode {latest_episode}...")
                agent.actor.load_weights(actor_path)
                agent.critic_1.load_weights(os.path.join(config.MODEL_PATH, f"critic1_ep{latest_episode}.h5"))
                agent.critic_2.load_weights(os.path.join(config.MODEL_PATH, f"critic2_ep{latest_episode}.h5"))
                start_episode = latest_episode

                current_noise_level = config.EXPLORATION_NOISE * (config.NOISE_DECAY ** start_episode)

                stats_path = os.path.join(config.SAVE_PATH, 'td3_per_stat.csv')
                if os.path.exists(stats_path):
                    df = pd.read_csv(stats_path)
                    all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = \
                        df['score'].tolist(), df['actor_loss'].tolist(), df['critic_loss'].tolist(), \
                        df['steps'].tolist(), df['y_position'].tolist()
                    recent_scores.extend(all_scores[-100:])
                    print("Loaded previous training statistics.")

    total_steps = 0
    for e in range(start_episode, config.MAX_EPISODES):
        state = env.reset()
        image_state = _process_image(state[0][0])
        state = [image_state, state[1]]

        total_reward, episode_steps = 0, 0
        actor_losses, critic_losses = [], []

        agent.noise.std_dev = max(config.MIN_EXPLORATION_NOISE, current_noise_level)

        consecutive_collisions = 0
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            total_steps += 1
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            next_image_state = _process_image(next_state[0][0])
            next_state_processed = [next_image_state, next_state[1]]

            # Track consecutive collisions for early termination
            if info.get('status') == 'collision':
                consecutive_collisions += 1
                if consecutive_collisions >= 3:  # End episode after 3 consecutive collisions
                    done = True
            else:
                consecutive_collisions = 0

            agent.buffer.add(state, action, reward, next_state_processed, done)
            actor_loss, critic_loss = agent.train(total_steps)

            if actor_loss is not None: actor_losses.append(actor_loss.numpy())
            if critic_loss is not None: critic_losses.append(critic_loss.numpy())

            state = next_state_processed
            total_reward += reward
            episode_steps = info.get('steps', step + 1)

            if done: break

        current_noise_level *= config.NOISE_DECAY

        recent_scores.append(total_reward)
        avg_reward_100 = np.mean(recent_scores) if recent_scores else 0

        all_scores.append(total_reward)
        all_actor_losses.append(np.mean(actor_losses) if actor_losses else 0)
        all_critic_losses.append(np.mean(critic_losses) if critic_losses else 0)
        all_steps.append(episode_steps)
        all_y_positions.append(info.get('Y', 0))

       # This is the new, modified line
        print(f"Ep: {e+1}, Steps: {episode_steps}, Y: {info.get('Y', 0):.2f}, Reward: {total_reward:.2f}, Avg R(100): {avg_reward_100:.2f}, Noise: {agent.noise.std_dev:.3f}, Status: {info.get('status')}")

        with summary_writer.as_default():
            tf.summary.scalar('Total Reward', total_reward, step=e)
            tf.summary.scalar('Average Reward (Last 100)', avg_reward_100, step=e)
            tf.summary.scalar('Steps per Episode', episode_steps, step=e)
            tf.summary.scalar('Exploration Noise', agent.noise.std_dev, step=e)

        if (e + 1) % config.SAVE_FREQ == 0:
            if not os.path.exists(config.MODEL_PATH): os.makedirs(config.MODEL_PATH)
            agent.actor.save(os.path.join(config.MODEL_PATH, f"actor_ep{e+1}.h5"))
            agent.critic_1.save(os.path.join(config.MODEL_PATH, f"critic1_ep{e+1}.h5"))
            agent.critic_2.save(os.path.join(config.MODEL_PATH, f"critic2_ep{e+1}.h5"))
            save_and_plot_metrics(e, all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions)

    env.disconnect()