import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import re

# Import your custom modules
from airsim_env import Env
from PER import PrioritizedReplayBuffer
import config

# --- Helper function for plotting and saving stats ---
def save_and_plot_metrics(episode, scores, actor_losses, critic_losses, steps, y_positions):
    """Saves metrics to a CSV and generates plots of the training progress."""
    
    # --- Save metrics to CSV ---
    # We need to make sure the length of each list is the same.
    # The episode number passed is the current one (e.g., 49 for the 50th episode)
    # The range should be up to episode + 1.
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
    df.to_csv(os.path.join(config.SAVE_PATH, 'td3_per_stat.csv'), index=False)
    
    # --- Generate and save plots ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('TD3+PER Training Progress', fontsize=16)
    
    # Plot scores
    axes[0, 0].plot(scores, label='Total Reward')
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Plot Actor Loss
    axes[0, 1].plot(actor_losses, label='Actor Loss', color='orange')
    axes[0, 1].set_title('Average Actor Loss per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Plot Critic Loss
    axes[1, 0].plot(critic_losses, label='Critic Loss', color='green')
    axes[1, 0].set_title('Average Critic Loss per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Plot Steps per Episode
    axes[1, 1].plot(steps, label='Steps', color='red')
    axes[1, 1].set_title('Steps per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Number of Steps')
    axes[1, 1].grid(True)

    # Plot Final Y Position
    axes[2, 0].plot(y_positions, label='Y Position', color='purple')
    axes[2, 0].set_title('Final Y Position per Episode')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Y Coordinate')
    axes[2, 0].grid(True)

    # Hide the empty subplot
    axes[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if not os.path.exists(config.GRAPH_PATH):
        os.makedirs(config.GRAPH_PATH)
    plt.savefig(os.path.join(config.GRAPH_PATH, 'td3_per_training_progress.png'))
    plt.close()
    print(f"Metrics and plots saved for episode {episode + 1}")

# --- Ornstein-Uhlenbeck Noise for Exploration ---
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
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
    def __init__(self, state_image_shape, state_velocity_shape, action_dim, action_bound):
        self.state_image_shape = state_image_shape
        self.state_velocity_shape = state_velocity_shape
        self.action_dim = action_dim
        self.action_bound = action_bound

        # --- Build Actor and Critic Models ---
        self.actor = self._build_actor()
        self.critic_1, self.critic_2 = self._build_critic(), self._build_critic()

        self.target_actor = self._build_actor()
        self.target_critic_1, self.target_critic_2 = self._build_critic(), self._build_critic()

        # Initialize target networks with the same weights as the main networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        # --- Optimizers ---
        self.actor_optimizer = Adam(learning_rate=config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=config.CRITIC_LEARNING_RATE)

        # --- Replay Buffer and Noise ---
        self.buffer = PrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.PER_ALPHA)
        self.noise = OUActionNoise(mean=np.zeros(action_dim), std_deviation=float(config.EXPLORATION_NOISE) * np.ones(action_dim))
        
    def _build_actor(self):
        # Image input
        image_input = Input(shape=self.state_image_shape)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_input)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        image_flatten = Flatten()(conv)

        # Velocity input
        velocity_input = Input(shape=self.state_velocity_shape)
        
        # Concatenate processed inputs
        concat = Concatenate()([image_flatten, velocity_input])
        
        # Dense layers
        dense = Dense(256, activation='relu')(concat)
        dense = Dense(256, activation='relu')(dense)
        # Output layer with tanh activation to bound actions between -1 and 1
        output = Dense(self.action_dim, activation='tanh')(dense)
        # Scale output to the action bound
        scaled_output = output * self.action_bound

        return Model(inputs=[image_input, velocity_input], outputs=scaled_output)

    def _build_critic(self):
        # State inputs (image and velocity)
        image_input = Input(shape=self.state_image_shape)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_input)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        image_flatten = Flatten()(conv)

        velocity_input = Input(shape=self.state_velocity_shape)
        state_concat = Concatenate()([image_flatten, velocity_input])
        state_dense = Dense(128, activation='relu')(state_concat)

        # Action input
        action_input = Input(shape=(self.action_dim,))
        action_dense = Dense(128, activation='relu')(action_input)

        # Concatenate processed state and action inputs
        concat = Concatenate()([state_dense, action_dense])
        
        # Dense layers for Q-value estimation
        dense = Dense(256, activation='relu')(concat)
        dense = Dense(256, activation='relu')(dense)
        output = Dense(1)(dense) # Output is a single Q-value

        return Model(inputs=[image_input, velocity_input, action_input], outputs=output)

    def select_action(self, state):
        image_state = tf.expand_dims(tf.convert_to_tensor(state[0]), 0)
        velocity_state = tf.expand_dims(tf.convert_to_tensor(state[1]), 0)
        action = self.actor([image_state, velocity_state])[0].numpy()
        action += self.noise()
        return np.clip(action, -self.action_bound, self.action_bound)

    def train(self, step):
        if len(self.buffer) < config.BATCH_SIZE:
            return None, None 

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(config.BATCH_SIZE)
        
        state_images = np.array([s[0] for s in states])
        state_velocities = np.array([s[1] for s in states])
        next_state_images = np.array([s[0] for s in next_states])
        next_state_velocities = np.array([s[1] for s in next_states])
        
        actor_loss_val = None
        
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor([next_state_images, next_state_velocities])
            noise = tf.clip_by_value(tf.random.normal(shape=tf.shape(target_actions), stddev=config.POLICY_NOISE),
                                     -config.NOISE_CLIP, config.NOISE_CLIP)
            target_actions = tf.clip_by_value(target_actions + noise, -self.action_bound, self.action_bound)

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

        critic_grad = tape.gradient(critic_loss_val, self.critic_1.trainable_variables + self.critic_2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_1.trainable_variables + self.critic_2.trainable_variables))
        
        self.buffer.update_priorities(indices, td_errors.numpy())

        if step % config.POLICY_UPDATE_FREQUENCY == 0:
            with tf.GradientTape() as tape:
                new_actions = self.actor([state_images, state_velocities])
                q1_new = self.critic_1([state_images, state_velocities, new_actions])
                actor_loss_val = -tf.reduce_mean(q1_new)
            
            actor_grad = tape.gradient(actor_loss_val, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            
            self._soft_update_target_networks()
        
        return actor_loss_val, critic_loss_val
            
    def _soft_update_target_networks(self):
        actor_weights = np.array(self.actor.get_weights(), dtype=object)
        target_actor_weights = np.array(self.target_actor.get_weights(), dtype=object)
        new_weights = config.TAU * actor_weights + (1 - config.TAU) * target_actor_weights
        self.target_actor.set_weights(new_weights)
        
        critic_1_weights = np.array(self.critic_1.get_weights(), dtype=object)
        target_critic_1_weights = np.array(self.target_critic_1.get_weights(), dtype=object)
        new_weights = config.TAU * critic_1_weights + (1 - config.TAU) * target_critic_1_weights
        self.target_critic_1.set_weights(new_weights)
        
        critic_2_weights = np.array(self.critic_2.get_weights(), dtype=object)
        target_critic_2_weights = np.array(self.target_critic_2.get_weights(), dtype=object)
        new_weights = config.TAU * critic_2_weights + (1 - config.TAU) * target_critic_2_weights
        self.target_critic_2.set_weights(new_weights)
        
# --- Main Training Loop ---
if __name__ == '__main__':
    env = Env()
    agent = TD3Agent(config.STATE_IMAGE_SHAPE, config.STATE_VELOCITY_SHAPE, config.ACTION_DIM, config.ACTION_BOUND)
    
    summary_writer = tf.summary.create_file_writer(config.TENSORBOARD_PATH)
    
    # --- Check for existing models and logs to resume training ---
    start_episode = 0
    all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions = [], [], [], [], []
    
    if os.path.exists(config.MODEL_PATH):
        # Find the latest episode number from saved models
        model_files = [f for f in os.listdir(config.MODEL_PATH) if f.startswith('actor_ep')]
        if model_files:
            latest_episode = max([int(re.search(r'(\d+)', f).group(1)) for f in model_files])
            
            actor_path = os.path.join(config.MODEL_PATH, f"actor_ep{latest_episode}.h5")
            critic1_path = os.path.join(config.MODEL_PATH, f"critic1_ep{latest_episode}.h5")
            critic2_path = os.path.join(config.MODEL_PATH, f"critic2_ep{latest_episode}.h5")
            
            if os.path.exists(actor_path) and os.path.exists(critic1_path) and os.path.exists(critic2_path):
                print(f"Resuming training from episode {latest_episode}...")
                agent.actor.load_weights(actor_path)
                agent.critic_1.load_weights(critic1_path)
                agent.critic_2.load_weights(critic2_path)
                
                # Also load target networks
                agent.target_actor.load_weights(actor_path)
                agent.target_critic_1.load_weights(critic1_path)
                agent.target_critic_2.load_weights(critic2_path)

                start_episode = latest_episode
                
                # Load previous stats
                stats_path = os.path.join(config.SAVE_PATH, 'td3_per_stat.csv')
                if os.path.exists(stats_path):
                    df = pd.read_csv(stats_path)
                    all_scores = df['score'].tolist()
                    all_actor_losses = df['actor_loss'].tolist()
                    all_critic_losses = df['critic_loss'].tolist()
                    all_steps = df['steps'].tolist()
                    all_y_positions = df['y_position'].tolist()
                    print("Loaded previous training statistics.")

    # --- Start the training loop ---
    for e in range(start_episode, config.MAX_EPISODES):
        state = env.reset()
        image_state = np.frombuffer(state[0][0].image_data_uint8, dtype=np.uint8)
        image_state = image_state.reshape(state[0][0].height, state[0][0].width, 3)
        image_state = tf.image.rgb_to_grayscale(image_state)
        image_state = tf.image.resize(image_state, [config.STATE_IMAGE_SHAPE[0], config.STATE_IMAGE_SHAPE[1]])
        image_state = image_state.numpy() / 255.0
        velocity_state = state[1]
        state = [image_state, velocity_state]
        
        total_reward = 0
        actor_losses, critic_losses = [], []
        
        for step in range(config.MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            next_image_state = np.frombuffer(next_state[0][0].image_data_uint8, dtype=np.uint8)
            next_image_state = next_image_state.reshape(next_state[0][0].height, next_state[0][0].width, 3)
            next_image_state = tf.image.rgb_to_grayscale(next_image_state)
            next_image_state = tf.image.resize(next_image_state, [config.STATE_IMAGE_SHAPE[0], config.STATE_IMAGE_SHAPE[1]])
            next_image_state = next_image_state.numpy() / 255.0
            next_velocity_state = next_state[1]
            next_state_processed = [next_image_state, next_velocity_state]

            agent.buffer.add(state, action, reward, next_state_processed, done)
            actor_loss, critic_loss = agent.train(step)
            
            if actor_loss is not None: actor_losses.append(actor_loss)
            if critic_loss is not None: critic_losses.append(critic_loss.numpy())

            state = next_state_processed
            total_reward += reward
            
            if done:
                break
        
        # --- Store and log metrics for the episode ---
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        
        # Ensure we are appending to the lists, not overwriting
        if e >= len(all_scores):
            all_scores.append(total_reward)
            all_actor_losses.append(avg_actor_loss)
            all_critic_losses.append(avg_critic_loss)
            all_steps.append(step + 1)
            all_y_positions.append(info.get('Y', 0))
        else: # This case is for re-running an episode that was interrupted.
            all_scores[e] = total_reward
            all_actor_losses[e] = avg_actor_loss
            all_critic_losses[e] = avg_critic_loss
            all_steps[e] = step + 1
            all_y_positions[e] = info.get('Y', 0)


        with summary_writer.as_default():
            tf.summary.scalar('Total Reward', total_reward, step=e)
            tf.summary.scalar('Average Actor Loss', avg_actor_loss, step=e)
            tf.summary.scalar('Average Critic Loss', avg_critic_loss, step=e)
            tf.summary.scalar('Steps per Episode', step + 1, step=e)
            tf.summary.scalar('Final Y Position', info.get('Y', 0), step=e)
        
        print(f"Episode: {e+1}, Steps: {step+1}, Total Reward: {total_reward:.2f}, Info: {info}")
        
        # --- Save models, metrics, and plots periodically ---
        if (e + 1) % 50 == 0:
            if not os.path.exists(config.MODEL_PATH): os.makedirs(config.MODEL_PATH)
            agent.actor.save(os.path.join(config.MODEL_PATH, f"actor_ep{e+1}.h5"))
            agent.critic_1.save(os.path.join(config.MODEL_PATH, f"critic1_ep{e+1}.h5"))
            agent.critic_2.save(os.path.join(config.MODEL_PATH, f"critic2_ep{e+1}.h5"))
            
            save_and_plot_metrics(e, all_scores, all_actor_losses, all_critic_losses, all_steps, all_y_positions)
            
    env.disconnect()