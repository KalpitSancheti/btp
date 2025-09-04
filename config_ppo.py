TOTAL_TIMESTEPS = 2e6       # Total timesteps for training
UPDATE_TIMESTEPS = 2048     # Number of steps to collect before updating the policy
BATCH_SIZE = 64             # Mini-batch size for training
PPO_EPOCHS = 10             # Number of epochs to train on the collected data
GAMMA = 0.99                # Discount factor for future rewards
GAE_LAMBDA = 0.95           # Lambda for Generalized Advantage Estimation
PPO_CLIP = 0.2              # Clipping parameter for PPO
LEARNING_RATE = 2.5e-4      # Learning rate for the optimizer
ENTROPY_COEF = 0.01         # Entropy coefficient for encouraging exploration
CRITIC_COEF = 0.5           # Value function loss coefficient
MAX_GRAD_NORM = 0.5         # Maximum norm for gradient clipping

# --- Environment and State/Action Shapes (same as before) ---
STATE_IMAGE_SHAPE = (84, 84, 1)
STATE_VELOCITY_SHAPE = (3,)
ACTION_DIM = 3
ACTION_BOUND = 1.0

# --- Saving and Logging Paths ---
MODEL_PATH = 'save_model/ppo/'
SAVE_PATH = 'save_stat/'
GRAPH_PATH = 'save_graph/ppo/'
TENSORBOARD_PATH = 'tensorboard_log/ppo/'

# --- Training Settings ---
MAX_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 1000
SAVE_FREQ = 100  # Save model 