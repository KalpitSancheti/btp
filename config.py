# config.py

# --- Your Original Settings ---
GOAL_Y = 57.0
OUT_Y = -0.5
MAX_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1000
SAVE_FREQ = 50
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.001
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_UPDATE_FREQUENCY = 1  # Update actor more frequently
BUFFER_SIZE = 100000
BUFFER_WARMUP = 2000
BATCH_SIZE = 128  # Larger batch for more stable learning
PER_ALPHA = 0.6
BUFFER_SIZE = 20000  # Smaller buffer to keep good experiences longer
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_FRAMES = 500000
EXPLORATION_NOISE = 0.3
MIN_EXPLORATION_NOISE = 0.1
NOISE_DECAY = 0.9999
STATE_IMAGE_SHAPE = (84, 84, 1)
STATE_VELOCITY_SHAPE = (3,)
ACTION_DIM = 3
ACTION_BOUND = 1.0
SAVE_PATH = "./save_stat/"
GRAPH_PATH = "./save_graph/"
MODEL_PATH = "./save_model/"
TENSORBOARD_PATH = "./tensorboard/"
TAKEOFF_VELOCITY = 3.0

# --- REWARD VALUES (Final, Synchronized Version) ---
REWARD_SUCCESS = 10.0
REWARD_COLLISION = -10.0
REWARD_WAYPOINT = 10.0
REWARD_OUT_OF_BOUNDS = -20.0
REWARD_STALL = -15.0
# Small center-keeping weights (used to softly prefer X≈0 and Z≈target_altitude)
CENTER_X_WEIGHT = 0.01  # Gentler quadratic penalty for X
CENTER_Z_WEIGHT = 0.015  # Gentler quadratic penalty for Z
CENTER_PENALTY_CAP = 0.5