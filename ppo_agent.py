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

# Import your custom modules
from airsim_env import Env
import config_ppo as config # Use the new PPO config

# --- Helper function for plotting and saving stats (re-used from TD3 script) ---
def save_and_plot_metrics(episode, scores, actor_losses, critic_losses, steps, y_positions):
    """Saves metrics to a CSV and generates plots of the training progress."""
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
    print(f"Metrics and plots saved for episode {episode + 1}")

# --- New robust image processing function ---
def _process_image(response):
    """Converts a single AirSim depth image response to a processed numpy array."""
    if not response or not response.image_data_float:
        # Return a zero image if the response is invalid
        return np.zeros(config.STATE_IMAGE_SHAPE, dtype=np.float32)
        
    # **FIX**: Use np.array for lists of floats, not np.frombuffer
    img_raw = np.array(response.image_data_float, dtype=np.float32)
    
    # Reshape to a single-channel image
    img_reshaped = img_raw.reshape(response.height, response.width, 1)
    
    # Resize and return
    img_resized = tf.image.resize(img_reshaped, [config.STATE_IMAGE_SHAPE[0], config.STATE_IMAGE_SHAPE[1]])
    return img_resized.numpy()

# --- PPO Agent Implementation ---
class PPOAgent:
    def __init__(self, state_image_shape, state_velocity_shape, action_dim, action_bound):
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.optimizer = Adam(learning_rate=config.LEARNING_RATE, clipnorm=config.MAX_GRAD_NORM)
        self.log_std = tf.Variable(np.zeros(action_dim, dtype=np.float32), trainable=True, name='log_std')
        self.model = self._build_actor_critic(state_image_shape, state_velocity_shape)

    def _build_actor_critic(self, image_shape, velocity_shape):
        image_input = Input(shape=image_shape)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_input)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        image_flatten = Flatten()(conv)

        velocity_input = Input(shape=velocity_shape)
        concat = Concatenate()([image_flatten, velocity_input])
        shared = Dense(512, activation='relu')(concat)

        actor_dense = Dense(256, activation='relu')(shared)
        mu = Dense(self.action_dim, activation='tanh')(actor_dense)
        mu = mu * self.action_bound

        critic_dense = Dense(256, activation='relu')(shared)
        value = Dense(1)(critic_dense)

        return Model(inputs=[image_input, velocity_input], outputs=[mu, value])

    def select_action(self, state):
        image_state = tf.expand_dims(tf.convert_to_tensor(state[0]), 0)
        velocity_state = tf.expand_dims(tf.convert_to_tensor(state[1]), 0)
        
        mu, value = self.model([image_state, velocity_state])
        std = tf.exp(self.log_std)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        
        action = dist.sample()
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        
        return action[0].numpy(), log_prob[0].numpy(), value[0].numpy()

    def train(self, states, actions, log_probs_old, returns, advantages):
        actor_losses, critic_losses = [], []
        
        for _ in range(config.PPO_EPOCHS):
            indices = np.random.permutation(len(states[0]))
            for start in range(0, len(states[0]), config.BATCH_SIZE):
                end = start + config.BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states_img = tf.convert_to_tensor(states[0][batch_indices])
                batch_states_vel = tf.convert_to_tensor(states[1][batch_indices])
                batch_actions = tf.convert_to_tensor(actions[batch_indices])
                batch_log_probs_old = tf.convert_to_tensor(log_probs_old[batch_indices])
                batch_returns = tf.convert_to_tensor(returns[batch_indices])
                batch_advantages = tf.convert_to_tensor(advantages[batch_indices])
                
                with tf.GradientTape() as tape:
                    mu, values = self.model([batch_states_img, batch_states_vel], training=True)
                    
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values))
                    
                    std = tf.exp(self.log_std)
                    dist = tfp.distributions.Normal(loc=mu, scale=std)
                    log_probs_new = tf.reduce_sum(dist.log_prob(batch_actions), axis=-1)
                    
                    ratio = tf.exp(log_probs_new - batch_log_probs_old)
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1.0 - config.PPO_CLIP, 1.0 + config.PPO_CLIP) * batch_advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    
                    entropy_loss = -tf.reduce_mean(dist.entropy())
                    
                    total_loss = (actor_loss + 
                                  config.CRITIC_COEF * critic_loss + 
                                  config.ENTROPY_COEF * entropy_loss)

                trainable_vars = self.model.trainable_variables + [self.log_std]
                grads = tape.gradient(total_loss, trainable_vars)
                self.optimizer.apply_gradients(zip(grads, trainable_vars))
                
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())

        return np.mean(actor_losses), np.mean(critic_losses)

# --- Generalized Advantage Estimation (GAE) ---
def compute_advantages_and_returns(rewards, values, dones, next_value):
    values = np.append(values, next_value)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + config.GAMMA * values[t+1] * (1-dones[t]) - values[t]
        advantages[t] = last_gae_lam = delta + config.GAMMA * config.GAE_LAMBDA * (1-dones[t]) * last_gae_lam
    returns = advantages + values[:-1]
    return returns, (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

# --- Main Training Loop ---
if __name__ == '__main__':
    env = Env()
    agent = PPOAgent(config.STATE_IMAGE_SHAPE, config.STATE_VELOCITY_SHAPE, config.ACTION_DIM, config.ACTION_BOUND)
    
    summary_writer = tf.summary.create_file_writer(config.TENSORBOARD_PATH)

    start_episode = 0
    all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = [], [], [], [], []

    if os.path.exists(config.MODEL_PATH):
        model_files = [f for f in os.listdir(config.MODEL_PATH) if f.startswith('ppo_model_ep')]
        if model_files:
            latest_episode = max([int(re.search(r'(\d+)', f).group(1)) for f in model_files])
            model_path = os.path.join(config.MODEL_PATH, f"ppo_model_ep{latest_episode}.h5")
            if os.path.exists(model_path):
                print(f"Resuming training from episode {latest_episode}...")
                dummy_img = np.zeros((1, *config.STATE_IMAGE_SHAPE))
                dummy_vel = np.zeros((1, *config.STATE_VELOCITY_SHAPE))
                agent.model([dummy_img, dummy_vel])
                agent.model.load_weights(model_path)
                start_episode = latest_episode
                
                stats_path = os.path.join(config.SAVE_PATH, 'ppo_stat.csv')
                if os.path.exists(stats_path):
                    df = pd.read_csv(stats_path)
                    all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = \
                        df['score'].tolist(), df['actor_loss'].tolist(), df['critic_loss'].tolist(), \
                        df['steps'].tolist(), df['y_position'].tolist()
                    print("Loaded previous training statistics.")
    
    episode_count = start_episode
    total_timesteps = 0
    
    while total_timesteps < config.TOTAL_TIMESTEPS and episode_count < config.MAX_EPISODES:
        states_img, states_vel, actions, rewards, dones, log_probs, values = [], [], [], [], [], [], []
        
        state = env.reset()
        current_episode_reward = 0
        
        for t in range(config.UPDATE_TIMESTEPS):
            total_timesteps += 1
            
            # Use the helper function for cleaner and safer processing
            image_state = _process_image(state[0][0])
            velocity_state = state[1]
            state_processed = [image_state, velocity_state]
            
            action, log_prob, value = agent.select_action(state_processed)
            next_state, reward, done, info = env.step(action)
            
            states_img.append(state_processed[0])
            states_vel.append(state_processed[1])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            
            state = next_state
            current_episode_reward += reward
            
            if done:
                print(f"Episode: {episode_count+1}, Timesteps: {total_timesteps}, Reward: {current_episode_reward:.2f}, Info: {info}")
                all_scores.append(current_episode_reward)
                all_steps.append(t+1)
                all_y_positions.append(info.get('Y', 0))
                with summary_writer.as_default():
                    tf.summary.scalar('Total Reward', current_episode_reward, step=episode_count)
                
                episode_count += 1
                state = env.reset()
                current_episode_reward = 0

        # Use the helper function for the final state of the rollout
        next_image_state = _process_image(state[0][0])
        next_velocity_state = state[1]
        _, _, next_value = agent.select_action([next_image_state, next_velocity_state])
        
        returns, advantages = compute_advantages_and_returns(rewards, values, dones, next_value)
        
        actor_loss, critic_loss = agent.train(
            [np.array(states_img), np.array(states_vel)],
            np.array(actions), np.array(log_probs), returns, advantages
        )
        
        all_actor_losses.append(actor_loss)
        all_critic_losses.append(critic_loss)
        print(f"Update Complete. Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        
        with summary_writer.as_default():
            tf.summary.scalar('Average Actor Loss', actor_loss, step=episode_count)
            tf.summary.scalar('Average Critic Loss', critic_loss, step=episode_count)
            
        if (episode_count) % config.SAVE_FREQ == 0 and episode_count > start_episode:
            if not os.path.exists(config.MODEL_PATH): os.makedirs(config.MODEL_PATH)
            agent.model.save_weights(os.path.join(config.MODEL_PATH, f"ppo_model_ep{episode_count}.h5"))
            save_and_plot_metrics(episode_count-1, all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions)
            
    env.disconnect()

