# rddpg_train.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (Input, Dense, Conv2D, Flatten, TimeDistributed,
                                     LSTM, Concatenate, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from airsim_env import Env
import config_rdpg as config  # NOTE: add RDDPG params to this config module or create a new config_rddpg

# ----------------- Helpers (plotting + image processing) -----------------
def save_and_plot_metrics(episode, scores, actor_losses, critic_losses, steps, y_positions):
    num_episodes = episode + 1
    df = pd.DataFrame({
        'episode': range(1, num_episodes + 1),
        'score': scores[:num_episodes],
        'actor_loss': actor_losses[:num_episodes],
        'critic_loss': critic_losses[:num_episodes],
        'steps': steps[:num_episodes],
        'y_position': y_positions[:num_episodes]
    })
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    df.to_csv(os.path.join(config.SAVE_PATH, 'rddpg_stat.csv'), index=False)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('R-DDPG Training Progress', fontsize=16)

    axes[0, 0].plot(scores, label='Total Reward')
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].grid(True)

    axes[0, 1].plot(actor_losses, label='Actor Loss')
    axes[0, 1].set_title('Actor Loss per Episode')
    axes[0, 1].grid(True)

    axes[1, 0].plot(critic_losses, label='Critic Loss')
    axes[1, 0].set_title('Critic Loss per Episode')
    axes[1, 0].grid(True)

    axes[1, 1].plot(steps, label='Steps')
    axes[1, 1].set_title('Steps per Episode')
    axes[1, 1].grid(True)

    axes[2, 0].plot(y_positions, label='Y Position')
    axes[2, 0].set_title('Final Y Position per Episode')
    axes[2, 0].grid(True)

    axes[2, 1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if not os.path.exists(config.GRAPH_PATH):
        os.makedirs(config.GRAPH_PATH)
    plt.savefig(os.path.join(config.GRAPH_PATH, 'rddpg_training_progress.png'))
    plt.close()
    print(f"Metrics and plots saved for episode {episode + 1}")

def _process_image(response):
    """Converts AirSim depth image response to a processed numpy array."""
    if not response or not response.image_data_float:
        return np.zeros(config.STATE_IMAGE_SHAPE, dtype=np.float32)
    img_raw = np.array(response.image_data_float, dtype=np.float32)
    img_reshaped = img_raw.reshape(response.height, response.width, 1)
    img_resized = tf.image.resize(img_reshaped, [config.STATE_IMAGE_SHAPE[0], config.STATE_IMAGE_SHAPE[1]])
    return img_resized.numpy().astype(np.float32)

# ----------------- Replay Buffer for sequences -----------------
class SequenceReplayBuffer:
    def __init__(self, max_size, seq_len):
        self.max_size = max_size
        self.seq_len = seq_len
        self.ptr = 0
        self.size = 0
        # store sequences as lists and convert to numpy on sample
        self.images = [None] * max_size    # each entry: np.array shape (seq_len, H, W, C)
        self.vels = [None] * max_size      # each entry: np.array shape (seq_len, vel_dim)
        self.actions = [None] * max_size   # each entry: np.array shape (seq_len, action_dim) or actions per timestep
        self.rewards = [None] * max_size   # shape (seq_len,)
        self.dones = [None] * max_size     # shape (seq_len,)
        self.next_images = [None] * max_size
        self.next_vels = [None] * max_size

    def add(self, seq_imgs, seq_vels, seq_actions, seq_rewards, seq_dones, seq_next_imgs, seq_next_vels):
        """Add a full sequence of length seq_len. All are numpy arrays."""
        self.images[self.ptr] = seq_imgs
        self.vels[self.ptr] = seq_vels
        self.actions[self.ptr] = seq_actions
        self.rewards[self.ptr] = seq_rewards
        self.dones[self.ptr] = seq_dones
        self.next_images[self.ptr] = seq_next_imgs
        self.next_vels[self.ptr] = seq_next_vels

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        imgs = np.array([self.images[i] for i in idxs], dtype=np.float32)
        vels = np.array([self.vels[i] for i in idxs], dtype=np.float32)
        actions = np.array([self.actions[i] for i in idxs], dtype=np.float32)
        rewards = np.array([self.rewards[i] for i in idxs], dtype=np.float32)
        dones = np.array([self.dones[i] for i in idxs], dtype=np.float32)
        next_imgs = np.array([self.next_images[i] for i in idxs], dtype=np.float32)
        next_vels = np.array([self.next_vels[i] for i in idxs], dtype=np.float32)
        return imgs, vels, actions, rewards, dones, next_imgs, next_vels

# ----------------- OU Noise -----------------
class OUActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = np.zeros_like(self.mu) if x0 is None else x0

    def __call__(self):
        x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

# ----------------- Recurrent Actor & Critic -----------------
def build_cnn_lstm_encoder(image_shape, vel_dim, seq_len):
    """
    Returns two model functions:
      - observation_encoder: inputs (seq_images, seq_vels) -> encoded vector (batch, encoding_dim)
      - encoder_model for TimeDistributed architecture (used inside actor/critic)
    """
    # Image input: (seq_len, H, W, C)
    seq_img_in = Input(shape=(seq_len, *image_shape), name='seq_images')
    seq_vel_in = Input(shape=(seq_len, vel_dim), name='seq_vels')

    # TimeDistributed CNN
    def make_cnn():
        img = Input(shape=image_shape)
        x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(img)
        x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
        x = Flatten()(x)
        return Model(img, x, name='cnn_encoder')

    cnn = make_cnn()
    td = TimeDistributed(cnn)(seq_img_in)  # (batch, seq_len, feat_dim)

    # Concatenate velocity per timestep and reduce with LSTM
    combined = Concatenate(axis=-1)([td, seq_vel_in])  # (batch, seq_len, feat + vel_dim)
    lstm_out = LSTM(512, return_sequences=False, name='encoder_lstm')(combined)  # final hidden

    encoder_model = Model(inputs=[seq_img_in, seq_vel_in], outputs=lstm_out, name='obs_encoder')
    return encoder_model

def build_actor(image_shape, vel_dim, seq_len, action_dim, action_bound):
    seq_img_in = Input(shape=(seq_len, *image_shape), name='actor_seq_images')
    seq_vel_in = Input(shape=(seq_len, vel_dim), name='actor_seq_vels')
    encoder = build_cnn_lstm_encoder(image_shape, vel_dim, seq_len)
    encoded = encoder([seq_img_in, seq_vel_in])
    x = Dense(256, activation='relu')(encoded)
    x = Dense(256, activation='relu')(x)
    mu = Dense(action_dim, activation='tanh')(x)
    scaled = tf.keras.layers.Lambda(lambda z: z * action_bound)(mu)
    model = Model(inputs=[seq_img_in, seq_vel_in], outputs=scaled, name='recurrent_actor')
    return model, encoder  # return encoder for reuse

def build_critic(image_shape, vel_dim, seq_len, action_dim):
    seq_img_in = Input(shape=(seq_len, *image_shape), name='critic_seq_images')
    seq_vel_in = Input(shape=(seq_len, vel_dim), name='critic_seq_vels')
    action_in = Input(shape=(action_dim,), name='critic_action')  # action at last timestep
    encoder = build_cnn_lstm_encoder(image_shape, vel_dim, seq_len)
    encoded = encoder([seq_img_in, seq_vel_in])
    x = Concatenate()([encoded, action_in])
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    q = Dense(1)(x)
    model = Model(inputs=[seq_img_in, seq_vel_in, action_in], outputs=q, name='recurrent_critic')
    return model, encoder

# ----------------- RDDPG Agent -----------------
class RDDPGAgent:
    def __init__(self, image_shape, vel_shape, action_dim, action_bound):
        self.image_shape = image_shape
        self.vel_shape = vel_shape
        self.seq_len = config.SEQ_LEN
        self.action_dim = action_dim
        self.action_bound = action_bound

        # Actor & Critic + targets
        self.actor, self.actor_encoder = build_actor(image_shape, vel_shape[0], self.seq_len, action_dim, action_bound)
        self.critic, self.critic_encoder = build_critic(image_shape, vel_shape[0], self.seq_len, action_dim)

        self.actor_target, _ = build_actor(image_shape, vel_shape[0], self.seq_len, action_dim, action_bound)
        self.critic_target, _ = build_critic(image_shape, vel_shape[0], self.seq_len, action_dim)

        # Initialize target weights
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.actor_optimizer = Adam(learning_rate=config.ACTOR_LR, clipnorm=config.MAX_GRAD_NORM)
        self.critic_optimizer = Adam(learning_rate=config.CRITIC_LR, clipnorm=config.MAX_GRAD_NORM)

        self.ou_noise = OUActionNoise(mu=np.zeros(self.action_dim),
                                      sigma=config.OU_SIGMA,
                                      theta=config.OU_THETA,
                                      dt=1.0/config.OU_DT)

    def get_action(self, seq_img, seq_vel, add_noise=True):
        """seq_img: (seq_len, H, W, C) ; seq_vel: (seq_len, vel_dim)"""
        s_img = np.expand_dims(seq_img, axis=0).astype(np.float32)
        s_vel = np.expand_dims(seq_vel, axis=0).astype(np.float32)
        action = self.actor.predict([s_img, s_vel], verbose=0)[0]
        if add_noise:
            noise = self.ou_noise()
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return action

    def train(self, buffer: SequenceReplayBuffer, batch_size):
        if buffer.size < batch_size:
            return 0.0, 0.0

        imgs, vels, actions_seq, rewards_seq, dones_seq, next_imgs, next_vels = buffer.sample(batch_size)
        # We'll train on the last timestep of each sequence
        # shape: imgs (B, seq_len, H,W,C) ; actions_seq (B, seq_len, action_dim)
        last_actions = actions_seq[:, -1, :]
        last_rewards = rewards_seq[:, -1]
        last_dones = dones_seq[:, -1].astype(np.float32)

        # compute target actions from actor_target using next sequences
        next_actions = self.actor_target.predict([next_imgs, next_vels], verbose=0)
        # compute target Q
        target_q = self.critic_target.predict([next_imgs, next_vels, next_actions], verbose=0)[:, 0]
        y = last_rewards + config.GAMMA * (1.0 - last_dones) * target_q

        # Train critic
        with tf.GradientTape() as tape:
            q_values = self.critic([imgs, vels, last_actions], training=True)[:, 0]
            critic_loss = tf.reduce_mean(tf.square(y - q_values))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Train actor (policy gradient via critic)
        with tf.GradientTape() as tape:
            pred_actions = self.actor([imgs, vels], training=True)
            # actor loss: maximize Q -> minimize -Q
            actor_q = self.critic([imgs, vels, pred_actions], training=False)[:, 0]
            actor_loss = -tf.reduce_mean(actor_q)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update targets
        self._soft_update(self.actor_target, self.actor, config.TAU)
        self._soft_update(self.critic_target, self.critic, config.TAU)

        return actor_loss.numpy(), critic_loss.numpy()

    def _soft_update(self, target_net, source_net, tau):
        target_weights = target_net.get_weights()
        source_weights = source_net.get_weights()
        new_weights = [tau * s + (1.0 - tau) * t for s, t in zip(source_weights, target_weights)]
        target_net.set_weights(new_weights)

    def save_weights(self, path, episode):
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.save_weights(os.path.join(path, f"rddpg_actor_ep{episode}.h5"))
        self.critic.save_weights(os.path.join(path, f"rddpg_critic_ep{episode}.h5"))
        self.actor_target.save_weights(os.path.join(path, f"rddpg_actor_target_ep{episode}.h5"))
        self.critic_target.save_weights(os.path.join(path, f"rddpg_critic_target_ep{episode}.h5"))

    def load_weights(self, path, episode):
        actor_p = os.path.join(path, f"rddpg_actor_ep{episode}.h5")
        critic_p = os.path.join(path, f"rddpg_critic_ep{episode}.h5")
        if os.path.exists(actor_p) and os.path.exists(critic_p):
            # build networks by calling once (if not already built)
            dummy_img = np.zeros((1, config.SEQ_LEN, *config.STATE_IMAGE_SHAPE), dtype=np.float32)
            dummy_vel = np.zeros((1, config.SEQ_LEN, *config.STATE_VELOCITY_SHAPE), dtype=np.float32)
            dummy_act = np.zeros((1, config.ACTION_DIM), dtype=np.float32)
            self.actor.predict([dummy_img, dummy_vel])
            self.critic.predict([dummy_img, dummy_vel, dummy_act])
            self.actor.load_weights(actor_p)
            self.critic.load_weights(critic_p)
            # copy to targets
            self.actor_target.set_weights(self.actor.get_weights())
            self.critic_target.set_weights(self.critic.get_weights())
            print(f"Loaded RDDPG weights from episode {episode}.")
            return True
        return False

# ----------------- Main training loop -----------------
if __name__ == '__main__':
    env = Env()
    agent = RDDPGAgent(config.STATE_IMAGE_SHAPE, config.STATE_VELOCITY_SHAPE, config.ACTION_DIM, config.ACTION_BOUND)

    buffer = SequenceReplayBuffer(max_size=config.REPLAY_BUFFER_SIZE, seq_len=config.SEQ_LEN)

    # metrics
    all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = [], [], [], [], []

    start_episode = 0
    if os.path.exists(config.MODEL_PATH):
        model_files = [f for f in os.listdir(config.MODEL_PATH) if f.startswith('rddpg_actor_ep')]
        if model_files:
            latest_episode = max([int(''.join(filter(str.isdigit, f))) for f in model_files])
            if agent.load_weights(config.MODEL_PATH, latest_episode):
                start_episode = latest_episode

        stats_path = os.path.join(config.SAVE_PATH, 'rddpg_stat.csv')
        if os.path.exists(stats_path):
            df = pd.read_csv(stats_path)
            all_scores = df['score'].tolist()
            all_actor_losses = df['actor_loss'].tolist()
            all_critic_losses = df['critic_loss'].tolist()
            all_steps = df['steps'].tolist()
            all_y_positions = df['y_position'].tolist()
            print("Loaded previous RDDPG stats.")

    episode_count = start_episode
    total_timesteps = 0

    # We'll build temporary deque buffers to accumulate seq_len steps
    from collections import deque
    seq_img_buf = deque(maxlen=config.SEQ_LEN)
    seq_vel_buf = deque(maxlen=config.SEQ_LEN)
    seq_act_buf = deque(maxlen=config.SEQ_LEN)
    seq_reward_buf = deque(maxlen=config.SEQ_LEN)
    seq_done_buf = deque(maxlen=config.SEQ_LEN)

    while total_timesteps < config.TOTAL_TIMESTEPS and episode_count < config.MAX_EPISODES:
        state = env.reset()
        seq_img_buf.clear(); seq_vel_buf.clear(); seq_act_buf.clear(); seq_reward_buf.clear(); seq_done_buf.clear()
        episode_reward = 0
        step = 0
        agent.ou_noise.reset()

        # initialize sequence buffers with first frame replicated if needed
        first_img = _process_image(state[0][0])
        first_vel = state[1]
        for _ in range(config.SEQ_LEN):
            seq_img_buf.append(first_img)
            seq_vel_buf.append(first_vel)
            seq_act_buf.append(np.zeros(config.ACTION_DIM, dtype=np.float32))
            seq_reward_buf.append(0.0)
            seq_done_buf.append(False)

        done = False
        info = {}
        while not done and total_timesteps < config.TOTAL_TIMESTEPS:
            # select action from past seq
            seq_imgs_np = np.array(list(seq_img_buf), dtype=np.float32)
            seq_vels_np = np.array(list(seq_vel_buf), dtype=np.float32)

            action = agent.get_action(seq_imgs_np, seq_vels_np, add_noise=True)

            next_state, reward, done, info = env.step(action)
            next_img = _process_image(next_state[0][0])
            next_vel = next_state[1]

            # append step to seq buffers (we store current step as last in sequence)
            seq_act_buf.append(action)
            seq_reward_buf.append(reward)
            seq_done_buf.append(done)

            # For storing into replay buffer we need:
            # - current sequence (seq_len) of images, vels, actions, rewards, dones
            # - next sequence: shift by one and append next_img/next_vel
            curr_imgs = np.array(list(seq_img_buf), dtype=np.float32)
            curr_vels = np.array(list(seq_vel_buf), dtype=np.float32)
            curr_actions = np.array(list(seq_act_buf), dtype=np.float32)
            curr_rewards = np.array(list(seq_reward_buf), dtype=np.float32)
            curr_dones = np.array(list(seq_done_buf), dtype=np.float32)

            # build next sequences
            next_seq_imgs = np.array(list(seq_img_buf)[1:] + [next_img], dtype=np.float32)
            next_seq_vels = np.array(list(seq_vel_buf)[1:] + [next_vel], dtype=np.float32)

            buffer.add(curr_imgs, curr_vels, curr_actions, curr_rewards, curr_dones, next_seq_imgs, next_seq_vels)

            # move main sequence forward
            seq_img_buf.append(next_img)
            seq_vel_buf.append(next_vel)

            episode_reward += reward
            total_timesteps += 1
            step += 1

            # training step: perform N updates per environment step if buffer has enough
            if buffer.size >= config.MIN_BUFFER_TO_LEARN:
                for _ in range(config.UPDATES_PER_STEP):
                    a_loss, c_loss = agent.train(buffer, config.BATCH_SIZE)
                all_actor_losses.append(a_loss)
                all_critic_losses.append(c_loss)

            if done:
                print(f"Episode {episode_count+1} done. Steps: {step}, Reward: {episode_reward:.2f}, Info: {info}")
                all_scores.append(episode_reward)
                all_steps.append(step)
                all_y_positions.append(info.get('Y', 0))
                with tf.summary.create_file_writer(config.TENSORBOARD_PATH).as_default():
                    tf.summary.scalar('Total Reward', episode_reward, step=episode_count)

                episode_count += 1
                break

        # periodically save models & stats
        if episode_count % config.SAVE_FREQ == 0 and episode_count > start_episode:
            agent.save_weights(config.MODEL_PATH, episode_count)
            save_and_plot_metrics(episode_count-1, all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions)

    env.disconnect()
