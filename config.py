# -- Agent Hyperparameters --
# Actor and Critic learning rates
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001

# Discount factor for future rewards
GAMMA = 0.99
# Soft update factor for target networks (polyak averaging)
TAU = 0.001

# -- TD3 Specific Hyperparameters --
# Noise added to the target policy for smoothing
POLICY_NOISE = 0.2
# Clipping value for the policy noise
NOISE_CLIP = 0.5
# Frequency of delayed policy updates
POLICY_UPDATE_FREQUENCY = 2

# -- Prioritized Experience Replay (PER) Hyperparameters --
# Size of the replay buffer
BUFFER_SIZE = 100000
# Number of experiences to sample from the buffer during training
BATCH_SIZE = 64
# Exponent for calculating priority (0 = uniform sampling, 1 = full prioritization)
PER_ALPHA = 0.6
# Importance-sampling exponent, annealed from BETA_START to 1.0
PER_BETA_START = 0.4
# Epsilon value to ensure no transition has zero priority
PER_EPSILON = 1e-6

# -- Training Configuration --
# Maximum number of training episodes
MAX_EPISODES = 5000
# Maximum number of steps per episode
MAX_STEPS_PER_EPISODE = 1000
# Standard deviation for Ornstein-Uhlenbeck exploration noise
EXPLORATION_NOISE = 0.1

# -- Environment Settings --
# Dimensions of the state from the environment (image and velocity)
STATE_IMAGE_SHAPE = (56, 100, 1)
STATE_VELOCITY_SHAPE = (3,)
# Number of actions the drone can take
ACTION_DIM = 3
# Maximum absolute value for each action
ACTION_BOUND = 1.0

# -- File Paths --
# Directory to save training statistics and graphs
SAVE_PATH = "./save_stat/"
MODEL_PATH = "./save_model/"
GRAPH_PATH = "./save_graph/"
TENSORBOARD_PATH = "./tensorboard/"