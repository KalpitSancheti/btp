# config_ppo.py  (conservative adjustments — safe to apply now)
import os

# --- Timesteps / batching -----------------------------------
TOTAL_TIMESTEPS = int(2e6)   # keep your total budget
UPDATE_TIMESTEPS = 2048      # collect this many steps before each PPO update
BATCH_SIZE = 64              # mini-batch size for training

# --- PPO training stability tweaks (small, conservative) ---
PPO_EPOCHS = 6               # reduced from 10 → fewer, safer updates per rollout
PPO_CLIP = 0.15              # slightly tighter clipping to avoid big policy jumps
LEARNING_RATE = 1e-4         # lowered from 2.5e-4 to stabilize actor updates

# --- Exploration / losses ----------------------------------
ENTROPY_COEF = 0.03         # slightly higher entropy to encourage exploration
CRITIC_COEF = 0.5            # keep as-is
MAX_GRAD_NORM = 0.5         # gradient clipping

# --- GAE / discount ----------------------------------------
GAMMA = 0.99                 # discount factor
GAE_LAMBDA = 0.95            # GAE lambda

# --- Observation / action shapes ---------------------------
STATE_IMAGE_SHAPE = (84, 84, 1)
STATE_VELOCITY_SHAPE = (3,)
ACTION_DIM = 3
ACTION_BOUND = 1.0

# --- Optional helpers (used by training script if present) ---
# These do not change env physics if your train script doesn't read them,
# but they're handy if you later enable image normalizing / clipping.
REWARD_CLIP = (-20.0, 50.0)    # safe clip for extreme per-step rewards (optional)
DEPTH_MAX = 30.0               # for depth image normalization (tune to your scene)
IMAGE_EMA_ALPHA = 0.6          # image smoothing alpha used by training script

# --- Saving & logging (preserved) --------------------------
MODEL_PATH = 'save_model/ppo/'
SAVE_PATH = 'save_stat/'
GRAPH_PATH = 'save_graph/ppo/'
TENSORBOARD_PATH = 'tensorboard_log/ppo/'

# --- Episode and persistence --------------------------------
MAX_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 1000
SAVE_FREQ = 20  # save every N episodes

# --- Misc ---------------------------------------------------
SEED = 12345     # reproducibility
PRINT_FREQ = 1   # episodes: how often to print logs

# create dirs if missing (safe)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(TENSORBOARD_PATH, exist_ok=True)
