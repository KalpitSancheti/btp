#!/usr/bin/env python3
"""
resume_check.py

Run this script to check why a resumed SAC model behaves as if it "forgot".
It will attempt to:
  - find your latest sac_training_logs_v2_* directory (or use --log-dir)
  - locate vecnormalize stats and a checkpoint (or use --vecstats/--checkpoint)
  - create the same env wrapper, load VecNormalize, and load the saved model
  - print observation/action shapes and vecstats (mean/var) info
  - run a few deterministic evaluation episodes and print returns

Usage examples:
  python resume_check.py --log-dir sac_training_logs_v2_20250919_123558 --checkpoint sac_model_ep3950.zip
  python resume_check.py  # will try to auto-find latest log dir and latest checkpoint

Keep this script in the same folder as your project (train_sac.py / airsim_sac_env.py).
"""

import os
import glob
import argparse
import pprint
import time

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Try to import your environment
try:
    from airsim_sac_env import AirSimEnv
except Exception as e:
    AirSimEnv = None
    print("Warning: couldn't import AirSimEnv from airsim_sac_env.py:\n", e)


def find_latest_logdir():
    dirs = glob.glob("sac_training_logs_v2_*")
    if not dirs:
        return None
    dirs.sort()
    return dirs[-1]


def find_latest_checkpoint(log_dir):
    if not log_dir:
        return None
    pattern = os.path.join(log_dir, "sac_model_ep*.zip")
    files = glob.glob(pattern)
    if not files:
        return None
    # extract ep number
    try:
        ep_nums = [(int(os.path.splitext(os.path.basename(p))[0].split('ep')[-1]), p) for p in files]
        ep_nums.sort()
        return ep_nums[-1][1]
    except Exception:
        return sorted(files)[-1]


def make_venv():
    if AirSimEnv is None:
        raise RuntimeError("AirSimEnv could not be imported. Make sure airsim_sac_env.py is in PYTHONPATH and importable.")

    def _init():
        env = AirSimEnv()
        return Monitor(env)
    return DummyVecEnv([_init])


def print_space_info(label, space):
    print(f"--- {label} ---")
    try:
        print("shape:", getattr(space, 'shape', None))
        print("dtype:", getattr(space, 'dtype', None))
        print("low:", getattr(space, 'low', None))
        print("high:", getattr(space, 'high', None))
    except Exception as e:
        print("(could not fully print space):", e)


def main(args):
    log_dir = args.log_dir or find_latest_logdir()
    if not log_dir:
        print("No log directory found. Use --log-dir to specify the folder containing your checkpoints and vecnormalize stats.")
        return

    print("Using log dir:", log_dir)

    # paths
    vecstats_path = args.vecstats or os.path.join(log_dir, 'vecnormalize_stats.pkl')
    checkpoint = args.checkpoint or find_latest_checkpoint(log_dir)

    print("Vecnormalize path:", vecstats_path if os.path.exists(vecstats_path) else f"(not found at {vecstats_path})")
    print("Checkpoint:", checkpoint if checkpoint and os.path.exists(checkpoint) else "(not found)")

    # Create venv
    try:
        venv = make_venv()
    except Exception as e:
        print("Failed to create DummyVecEnv:", e)
        return

    # Load or create VecNormalize wrapper
    env = None
    if os.path.exists(vecstats_path):
        try:
            env = VecNormalize.load(vecstats_path, venv)
            print("Loaded VecNormalize from:", vecstats_path)
        except Exception as e:
            print("Failed to load VecNormalize stats:", e)
            print("Creating a new VecNormalize wrapper instead (this will not have the old normalization stats).")
            env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        print("Created new VecNormalize because vecstats file was not found.")

    # Print observation/action spaces
    print_space_info('env.observation_space', env.observation_space)
    print_space_info('env.action_space', env.action_space)

    # Print obs_rms info if available
    try:
        print("obs_rms exists?", hasattr(env, 'obs_rms'))
        if hasattr(env, 'obs_rms') and env.obs_rms is not None:
            print("obs_rms mean shape:", getattr(env.obs_rms, 'mean', None).shape)
            print("obs_rms var shape:", getattr(env.obs_rms, 'var', None).shape)
            # print first few elements
            print("obs_rms mean (first 10):", getattr(env.obs_rms, 'mean', None)[:10])
            print("obs_rms var  (first 10):", getattr(env.obs_rms, 'var', None)[:10])
    except Exception as e:
        print("Couldn't inspect obs_rms:", e)

    model = None
    if checkpoint and os.path.exists(checkpoint):
        print('\nTrying to load model with env...')
        try:
            model = SAC.load(checkpoint, env=env)
            print('Loaded model with env successfully.')
        except Exception as e:
            print('Loading with env failed:', e)
            print('Trying to load model without env then set_env...')
            try:
                model = SAC.load(checkpoint, env=None)
                model.set_env(env)
                print('Loaded model without env and attached env via set_env().')
            except Exception as e2:
                print('Failed to load model at all:', e2)

    if model is None:
        print('No model loaded. Exiting checks.')
        return

    # Print model expected spaces
    try:
        # stable-baselines3 models keep observation_space in model.policy or model.observation_space
        mo = getattr(model, 'observation_space', None) or getattr(model.policy, 'observation_space', None)
        ma = getattr(model, 'action_space', None) or getattr(model, 'policy', None) and getattr(model.policy, 'action_space', None)
        print_space_info('model.observation_space', mo)
        print_space_info('model.action_space', ma)
    except Exception as e:
        print("Couldn't print model spaces:", e)

    # Quick check: shapes match?
    try:
        env_obs_shape = env.observation_space.shape
        model_obs_shape = mo.shape if mo is not None else None
        if env_obs_shape != model_obs_shape:
            print('\n*** OBSERVATION SHAPE MISMATCH ***')
            print('env obs shape =', env_obs_shape)
            print('model obs shape =', model_obs_shape)
            print('If these differ, your AirSimEnv changed (features added/removed or order changed). You cannot safely resume this model.')
        else:
            print('\nObservation shapes match. Good.')
    except Exception as e:
        print('Could not compare observation shapes:', e)

    # Make sure env is set to evaluation mode (don't update running stats)
    try:
        env.training = False
        print('Set env.training = False for deterministic eval (VecNormalize will not update stats).')
    except Exception:
        pass

    # Deterministic eval runs
    n_eval = args.eval_episodes
    max_steps = args.max_steps
    print(f'\nRunning {n_eval} deterministic eval episodes (max {max_steps} steps each).')

    for ep in range(n_eval):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info = None
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward if isinstance(reward, (int, float)) else np.sum(reward)
            last_info = info
            steps += 1
        print(f'Eval episode {ep+1}: return={total_reward}, steps={steps}, last_info_keys={list(last_info.keys()) if isinstance(last_info, dict) else None}')

    # Quick sanity: show first few actions on a fresh reset
    print('\nSampling a few policy actions on a single reset:')
    obs = env.reset()
    for i in range(5):
        action, _ = model.predict(obs, deterministic=True)
        print(' action:', np.array(action).tolist())
        obs, _, _, _ = env.step(env.action_space.sample())

    # If we reach here, we printed everything helpful.
    print('\nCheck above messages for:')
    print('- obs/action shapes and whether they match the model')
    print('- vecnormalize mean/var shapes and values (first few entries)')
    print('- deterministic eval returns (are they reasonable?)')
    print('\nIf you see an observation shape mismatch or missing vecstats, revert AirSimEnv to the version used when the model was trained, or retrain from scratch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default=None, help='Path to sac_training_logs_v2_* directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a specific sac_model_epXXXX.zip file')
    parser.add_argument('--vecstats', type=str, default=None, help='Path to vecnormalize_stats.pkl (optional)')
    parser.add_argument('--eval-episodes', type=int, default=3, help='Number of deterministic eval episodes to run')
    parser.add_argument('--max-steps', type=int, default=2000, help='Max steps per eval episode')
    args = parser.parse_args()
    main(args)