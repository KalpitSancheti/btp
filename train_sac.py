# train_sac.py
# Fixed version: SafeInfoWrapper subclasses gymnasium.Wrapper so Monitor accepts it.
# Full training script with robust resume, monitor tailing, and PER support (no blocking warm-up).

import os
import glob
import time
import argparse
import csv
import shutil
import re
import json
import warnings

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement, CallbackList

from airsim_sac_env import AirSimEnv
from per_buffer import PrioritizedReplayBuffer

# --------------------
# Safe wrapper (subclass gym.Wrapper)
# --------------------
class SafeInfoWrapper(gym.Wrapper):
    """
    A gymnasium.Wrapper that ensures the info dict returned by step/reset
    contains required keys (with safe defaults). Compatible with Monitor.
    """
    def __init__(self, env: gym.Env, required_keys=()):
        super().__init__(env)
        self.required_keys = tuple(required_keys)
        # ensure wrapper exposes same spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})

    def _ensure_info(self, info):
        if info is None:
            info = {}
        # ensure keys exist with sensible defaults
        for k in self.required_keys:
            if k not in info:
                if k in ("checkpoints_reached",):
                    info[k] = []
                elif "count" in k or "override" in k:
                    info[k] = 0
                elif "max_y" in k or "min_clear" in k or "max" in k or "distance" in k:
                    info[k] = 0.0
                else:
                    info[k] = ""
        return info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # gymnasium reset returns (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            info = self._ensure_info(info)
            return obs, info
        # older envs: just obs
        return out, {k: (0.0 if "max" in k or "min" in k else ([] if "checkpoints" in k else 0 if "count" in k or "override" in k else "")) for k in self.required_keys}

    def step(self, action):
        out = self.env.step(action)
        # handle gym-style 4-tuple (obs, reward, done, info)
        if isinstance(out, tuple):
            if len(out) == 4:
                obs, reward, done, info = out
                info = self._ensure_info(info)
                # gymnasium expects (obs, reward, terminated, truncated, info)
                # convert done -> terminated, truncated=False
                return obs, reward, done, False, info
            elif len(out) == 5:
                obs, reward, terminated, truncated, info = out
                info = self._ensure_info(info)
                return obs, reward, terminated, truncated, info
        # otherwise just return as-is (best effort)
        return out

# --------------------
# Logging & callbacks
# --------------------
def get_or_create_log_dir(restart=False):
    import datetime
    if restart:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"sac_training_logs_v2_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created new training directory: {log_dir}")
        return log_dir
    else:
        existing_dirs = glob.glob("sac_training_logs_v2_*")
        if existing_dirs:
            existing_dirs.sort()
            log_dir = existing_dirs[-1]
            print(f"Continuing with existing directory: {log_dir}")
            return log_dir
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"sac_training_logs_v2_{timestamp}"
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created new training directory: {log_dir}")
            return log_dir

def find_latest_checkpoint(log_dir: str):
    checkpoints = glob.glob(os.path.join(log_dir, "sac_model_ep*.zip"))
    if not checkpoints:
        return None
    ep_numbers = []
    for p in checkpoints:
        m = re.search(r"ep(\d+)\.zip$", os.path.basename(p))
        if m:
            ep_numbers.append(int(m.group(1)))
    if not ep_numbers:
        return None
    latest = max(ep_numbers)
    return os.path.join(log_dir, f"sac_model_ep{latest}.zip")

def parse_ep_from_checkpoint_path(path: str):
    if not path:
        return 0
    m = re.search(r"ep(\d+)\.zip$", os.path.basename(path))
    if m:
        return int(m.group(1))
    return 0

def find_best_checkpoint_from_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    best_ep = None
    best_reward = -float('inf')
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = float(row.get("episode_reward", 0))
                    ep = int(row.get("episode", 0))
                    if r > best_reward:
                        best_reward = r
                        best_ep = ep
                except Exception:
                    continue
    except Exception:
        return None
    if best_ep is None:
        return None
    candidate = os.path.join(LOG_DIR, f"sac_model_ep{best_ep}.zip")
    if os.path.exists(candidate):
        return candidate
    return find_latest_checkpoint(LOG_DIR)

class CSVCheckpointCallback(BaseCallback):
    def __init__(self, log_dir: str, save_freq_episodes: int = 50, verbose=1, start_episode: int = 0):
        super(CSVCheckpointCallback, self).__init__(verbose=verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "training_log.csv")
        self.save_freq = save_freq_episodes
        self.episode_counter = 0
        self.start_episode = int(start_episode or 0)
        self.monitor_path = os.path.join(self.log_dir, "monitor.csv")
        self._monitor_header = None
        self._monitor_last_line = 0

        print(f"CSVCheckpointCallback initialized (csv: {self.csv_path}, monitor: {self.monitor_path}, start_episode={self.start_episode})")

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode","total_steps","episode_reward","episode_length","max_y_distance","termination_reason","collision_type","checkpoints_reached","override_count","min_clear_above","wall_time"])
        else:
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                data_lines = [ln for ln in lines if not ln.startswith("episode,")]
                if data_lines:
                    last_line = data_lines[-1]
                    last_parts = next(csv.reader([last_line]))
                    last_ep = int(last_parts[0])
                    self.episode_counter = last_ep
                    print(f"Resuming episode counter from training_log.csv: {self.episode_counter}")
                else:
                    if self.start_episode > 0:
                        self.episode_counter = self.start_episode
                        print(f"No data in CSV; using checkpoint-based start_episode: {self.episode_counter}")
            except Exception as e:
                print("Warning reading training_log.csv:", e)
                if self.start_episode > 0:
                    self.episode_counter = self.start_episode
                    print(f"Using checkpoint-based start_episode: {self.episode_counter}")

        if os.path.exists(self.monitor_path):
            try:
                with open(self.monitor_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                header_idx = None
                for i, ln in enumerate(lines):
                    if not ln.startswith("#"):
                        header_idx = i
                        break
                if header_idx is not None:
                    self._monitor_header = [c.strip() for c in lines[header_idx].strip().split(",")]
                self._monitor_last_line = len(lines)
                print(f"Found existing monitor.csv with {len(lines)} lines; header captured.")
            except Exception as e:
                print("Warning reading existing monitor file:", e)

    def _on_training_start(self) -> None:
        print("CSVCheckpointCallback: training started, episode_counter =", self.episode_counter)
        if os.path.exists(self.monitor_path):
            print("Monitor file exists at start:", self.monitor_path)
        else:
            print("Monitor file will be created by Monitor wrapper during env initialization.")

    def _process_monitor_new_lines(self, new_lines):
        appended = 0
        for ln in new_lines:
            if not ln.strip():
                continue
            if ln.startswith("#"):
                continue
            if self._monitor_header is None:
                self._monitor_header = [c.strip() for c in ln.strip().split(",")]
                continue
            try:
                vals = next(csv.reader([ln]))
            except Exception:
                continue
            if len(vals) == 0:
                continue
            row_map = dict(zip(self._monitor_header, vals))
            reward = row_map.get("r", "")
            length = row_map.get("l", "")
            max_y = row_map.get("max_y_distance", "")
            term = row_map.get("termination_reason", "")
            collision_type = row_map.get("collision_type", "")
            cps = row_map.get("checkpoints_reached", "")
            overrides = row_map.get("override_count", "")
            min_clear_above = row_map.get("min_clear_above", "")
            total_steps = self.num_timesteps

            self.episode_counter += 1

            try:
                r_float = float(reward)
                print(f"[Episode {self.episode_counter}] Reward: {r_float:.2f}  Len: {length}  MaxY: {float(max_y) if max_y!='' else 0:.2f}  Term: {term}  Coll: {collision_type}  CPs: {cps}  Overrides: {overrides}  MinClearAbove: {min_clear_above}")
            except Exception:
                print(f"[Episode {self.episode_counter}] Reward: {reward}  Len: {length}  MaxY: {max_y}  Term: {term}")

            try:
                with open(self.csv_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode_counter, total_steps, reward, length, max_y, term, collision_type, cps, overrides, min_clear_above, time.time()])
                appended += 1
            except Exception as e:
                print("Failed to append to training_log.csv:", e)
        return appended

    def _on_step(self) -> bool:
        try:
            if os.path.exists(self.monitor_path):
                with open(self.monitor_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > self._monitor_last_line:
                    new_lines = lines[self._monitor_last_line:]
                    added = self._process_monitor_new_lines(new_lines)
                    self._monitor_last_line = len(lines)
                    if added:
                        print(f"CSVCheckpointCallback: appended {added} new episode rows to {self.csv_path}")
                    if self.episode_counter > 0 and (self.episode_counter % self.save_freq == 0):
                        try:
                            model_path = os.path.join(self.log_dir, f"sac_model_ep{self.episode_counter}.zip")
                            self.model.save(model_path)
                            if self.verbose:
                                print(f"Saved checkpoint: {model_path}")
                        except Exception as e:
                            print("Failed to save periodic checkpoint:", e)
        except Exception as e:
            print("CSVCheckpointCallback error while tailing monitor:", e)
        return True

# --------------------
# Main training flow
# --------------------
def remove_old_checkpoints(log_dir):
    for p in glob.glob(os.path.join(log_dir, "sac_model_ep*.zip")):
        try:
            os.remove(p)
        except Exception:
            pass
    final = os.path.join(log_dir, "sac_model_final.zip")
    if os.path.exists(final):
        try:
            os.remove(final)
        except Exception:
            pass
    print("Removed old checkpoints and vecnormalize stats (if any).")

def make_env(LOG_DIR, info_keys):
    def _init():
        base = AirSimEnv()
        wrapped = SafeInfoWrapper(base, required_keys=info_keys)
        monitor_fp = os.path.join(LOG_DIR, "monitor.csv") if LOG_DIR is not None else None
        return Monitor(wrapped, filename=monitor_fp, info_keywords=info_keys)
    return _init

def main(total_timesteps=int(3e6), restart=False, resume_best=False, use_per=True):
    global LOG_DIR, CSV_FILE, VECSTAT_PATH

    LOG_DIR = get_or_create_log_dir(restart)
    CSV_FILE = os.path.join(LOG_DIR, "training_log.csv")
    VECSTAT_PATH = os.path.join(LOG_DIR, "vecnormalize_stats.pkl")

    config_info = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "log_dir": LOG_DIR,
        "environment": "AirSimEnv with altitude fixes",
        "algorithm": "SAC",
        "use_per": use_per,
    }
    with open(os.path.join(LOG_DIR, "training_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    if restart:
        remove_old_checkpoints(LOG_DIR)

    # Decide checkpoint to load
    checkpoint_to_load = None
    if resume_best:
        checkpoint_to_load = find_best_checkpoint_from_csv(CSV_FILE)
        if checkpoint_to_load:
            print(f"Found best checkpoint from CSV: {checkpoint_to_load}")
    else:
        checkpoint_to_load = find_latest_checkpoint(LOG_DIR)
        if checkpoint_to_load:
            print(f"Found latest checkpoint: {checkpoint_to_load}")

    checkpoint_ep = parse_ep_from_checkpoint_path(checkpoint_to_load) if checkpoint_to_load else 0
    if checkpoint_ep:
        print(f"Detected checkpoint episode number: {checkpoint_ep}")

    info_keys = ("max_y_distance","termination_reason","collision_type","checkpoints_reached","override_count","min_clear_above")

    # Create vectorized env
    venv = DummyVecEnv([make_env(LOG_DIR, info_keys)])

    # load or create VecNormalize
    if os.path.exists(VECSTAT_PATH) and not restart:
        try:
            env = VecNormalize.load(VECSTAT_PATH, venv)
            print(f"Loaded VecNormalize stats from {VECSTAT_PATH}")
        except Exception as e:
            warnings.warn(f"Failed to load VecNormalize stats: {e}. Creating new VecNormalize.")
            env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = None
    if checkpoint_to_load and not restart:
        try:
            print("Attempting to load model WITHOUT env, then attach env via set_env()...")
            model = SAC.load(checkpoint_to_load, env=None, tensorboard_log=LOG_DIR)
            model.set_env(env)
            print("Loaded model and attached env via set_env().")
        except Exception as e:
            print("Loading model without env failed:", e)
            print("Falling back to load with env passed...")
            try:
                model = SAC.load(checkpoint_to_load, env=env, tensorboard_log=LOG_DIR)
                print("Loaded model with env successfully.")
            except Exception as e2:
                print("Failed to load checkpoint:", e2)
                model = None

        try:
            if model is not None and use_per:
                print("Recreating PrioritizedReplayBuffer instance and attaching to model (will be empty).")
                per_buffer = PrioritizedReplayBuffer(
                    buffer_size=int(2e6) if use_per else int(1e6),
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=0.6,
                    beta=0.4,
                    beta_increment=0.001,
                )
                model.replay_buffer = per_buffer
                print("Attached new empty PER buffer to model.")
        except Exception as e:
            print("Warning: could not recreate PER buffer:", e)

        try:
            old_ls = getattr(model, "learning_starts", 0)
            new_ls = max(old_ls, 5000)
            model.learning_starts = new_ls
            print(f"Set model.learning_starts = {new_ls} to avoid immediate learning on an empty buffer.")
        except Exception as e:
            print("Warning: could not set model.learning_starts:", e)

    if model is None:
        sac_kwargs = {
            'buffer_size': int(2e6) if use_per else int(1e6),
            'learning_rate': 3e-4,
            'batch_size': 512,
            'gamma': 0.99,
            'tau': 0.01,
            'train_freq': (4, "step"),
            'gradient_steps': 4,
            'learning_starts': 5000,
            'ent_coef': 0.1,
            'tensorboard_log': LOG_DIR,
            'device': 'cpu'
        }
        if use_per:
            print("Creating new SAC model with Prioritized Experience Replay...")
            per_buffer = PrioritizedReplayBuffer(
                buffer_size=sac_kwargs['buffer_size'],
                observation_space=env.observation_space,
                action_space=env.action_space,
                alpha=0.6,
                beta=0.4,
                beta_increment=0.001,
            )
            model = SAC(policy='MlpPolicy', env=env, verbose=1, **sac_kwargs)
            model.replay_buffer = per_buffer
            print("PER buffer integrated successfully!")
        else:
            print("Creating new SAC model with standard replay buffer...")
            model = SAC(policy='MlpPolicy', env=env, verbose=1, **sac_kwargs)

    csv_cb = CSVCheckpointCallback(log_dir=LOG_DIR, save_freq_episodes=50, verbose=1, start_episode=checkpoint_ep)

    eval_venv = DummyVecEnv([make_env(LOG_DIR, info_keys)])
    eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    try:
        if hasattr(env, "obs_rms") and env.obs_rms is not None:
            eval_env.obs_rms = env.obs_rms
            if hasattr(env, "ret_rms"):
                eval_env.ret_rms = env.ret_rms
            print("Copied obs_rms/ret_rms from training env to eval env.")
    except Exception as e:
        print("Warning: could not copy obs_rms to eval env:", e)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=50, verbose=1)
    eval_cb = EvalCallback(eval_env, best_model_save_path=LOG_DIR, log_path=LOG_DIR, eval_freq=5000, n_eval_episodes=5, deterministic=True, render=False, callback_on_new_best=stop_callback)

    callback_list = CallbackList([csv_cb, eval_cb])

    try:
        env.training = True
        print("Set env.training = True for continued learning (VecNormalize will update stats).")
    except Exception:
        pass
    try:
        eval_env.training = False
    except Exception:
        pass

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list)
    except KeyboardInterrupt:
        print("Interrupted by user. Saving current model and stats.")
    finally:
        final_model_path = os.path.join(LOG_DIR, "sac_model_final.zip")
        try:
            model.save(final_model_path)
            print(f"Saved final model to {final_model_path}")
        except Exception as e:
            print("Failed to save final model:", e)
        try:
            if isinstance(env, VecNormalize):
                env.save(VECSTAT_PATH)
                print(f"Saved VecNormalize stats to {VECSTAT_PATH}")
        except Exception as e:
            print("Warning: failed to save VecNormalize stats:", e)
        env.close()
        eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=int(3e6))
    parser.add_argument("--restart", action="store_true", help="Delete old checkpoints and VecNormalize stats then start fresh")
    parser.add_argument("--resume-best", action="store_true", help="Resume from best checkpoint found in CSV (if any)")
    parser.add_argument("--use-per", action="store_true", default=True, help="Use Prioritized Experience Replay (default: True)")
    parser.add_argument("--no-per", action="store_true", help="Disable Prioritized Experience Replay")
    args = parser.parse_args()
    use_per = args.use_per and not args.no_per
    main(total_timesteps=args.timesteps, restart=args.restart, resume_best=args.resume_best, use_per=use_per)
