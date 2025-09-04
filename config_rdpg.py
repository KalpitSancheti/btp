# config_rdpg.py
# Complete config for R-DDPG script (contains all names used by R-DDPG.py)

# ----------------- Training / runtime -----------------
TOTAL_TIMESTEPS = int(2e6)       # total environment timesteps to run
MAX_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 1000

# ----------------- Replay buffer / sequence -----------------
REPLAY_BUFFER_SIZE = int(1e6)   # used by R-DDPG.py (buffer = SequenceReplayBuffer(...))
BUFFER_SIZE = REPLAY_BUFFER_SIZE  # alias (just in case script expects BUFFER_SIZE)
SEQ_LEN = 10                    # sequence length used by LSTM encoder
MIN_BUFFER_TO_LEARN = 1000      # minimum sequences before learning begins (used in script)
WARMUP_STEPS = 10000            # optional warmup (not mandatory) - you had this earlier

# ----------------- Batch / update -----------------
BATCH_SIZE = 64
UPDATES_PER_STEP = 1            # how many gradient updates per env step (script expects this name)
UPDATE_FREQ = 100               # alternate update frequency (kept for compatibility)

# ----------------- Learning / optimization -----------------
ACTOR_LR = 1e-4                 # script expects this name (actor optimizer)
CRITIC_LR = 1e-3                # script expects this name (critic optimizer)
MAX_GRAD_NORM = 0.5             # clipnorm used in Adam creation
TAU = 0.005                     # soft update factor for target networks
GAMMA = 0.99                    # discount factor

# ----------------- OU noise (script expects OU_SIGMA / OU_THETA / OU_DT) ----------
OU_SIGMA = 0.2                  # Ornstein-Uhlenbeck sigma (volatility)
OU_THETA = 0.15                 # OU theta (mean reversion speed)
OU_DT = 0.01                    # OU dt (time step)
OU_SCALE = 0.3                  # optional overall scale (not required by script but included)

# ----------------- Environment shapes -----------------
STATE_IMAGE_SHAPE = (84, 84, 1)
STATE_VELOCITY_SHAPE = (3,)
ACTION_DIM = 3
ACTION_BOUND = 1.0

# ----------------- Paths / logging / saving -----------------
MODEL_PATH = 'save_model/rdpg/'
SAVE_PATH = 'save_stat/'
GRAPH_PATH = 'save_graph/rdpg/'
TENSORBOARD_PATH = 'tensorboard_log/rdpg/'

# Save frequency used by your training loop
SAVE_FREQ = 100

# ----------------- Misc / compatibility -----------------
# Keep older-style names present in case other parts of script use them
LEARNING_RATE_ACTOR = ACTOR_LR
LEARNING_RATE_CRITIC = CRITIC_LR
GRAD_CLIP = MAX_GRAD_NORM

# sanity: ensure ints where needed
TOTAL_TIMESTEPS = int(TOTAL_TIMESTEPS)
REPLAY_BUFFER_SIZE = int(REPLAY_BUFFER_SIZE)
BUFFER_SIZE = int(BUFFER_SIZE)
BATCH_SIZE = int(BATCH_SIZE)
SEQ_LEN = int(SEQ_LEN)
MIN_BUFFER_TO_LEARN = int(MIN_BUFFER_TO_LEARN)
WARMUP_STEPS = int(WARMUP_STEPS)
UPDATES_PER_STEP = int(UPDATES_PER_STEP)
