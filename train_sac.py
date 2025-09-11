import os
import csv
import glob
import time
from airsim_sac_env import AirSimEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

class TrainingCallback(BaseCallback):
    """
    Custom callback for advanced logging and periodic checkpointing based on episodes.
    """
    def __init__(self, log_dir: str, save_freq_episodes: int, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq_episodes
        self.csv_file_path = os.path.join(self.log_dir, "training_log.csv")
        self.episode_count = 0

        # Find the last episode number from an existing log file
        if os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    self.episode_count = int(last_row[0])
            print(f"Resuming logging from episode {self.episode_count + 1}.")
            self.csv_file = open(self.csv_file_path, "a", newline="")
            self.csv_writer = csv.writer(self.csv_file)
        else:
            self.csv_file = open(self.csv_file_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["episode", "total_steps", "episode_reward", "episode_length", "max_y_distance", "termination_reason"])
        
        print(f"Logging training data to {self.csv_file_path}")

    def _on_step(self) -> bool:
        # Check if an episode has finished
        if self.locals["dones"][0]:
            self.episode_count += 1
            info = self.locals["infos"][0]
            
            reward = info['episode']['r']
            length = info['episode']['l']
            max_y = info.get("max_y_distance", 0)
            termination_reason = info.get("termination_reason", "unknown")
            
            # Log to terminal
            print(f"[ Episode {self.episode_count} ]  "
                  f"Reward: {reward:<8.2f}  "
                  f"Max Distance: {max_y:<7.2f}m  "
                  f"Length: {length:<4}  "
                  f"Reason: {termination_reason}")
            
            # Log to CSV
            self.csv_writer.writerow([self.episode_count, self.num_timesteps, reward, length, max_y, termination_reason])
            self.csv_file.flush()

            # Checkpoint every N episodes
            if self.episode_count % self.save_freq == 0:
                model_path = os.path.join(self.log_dir, f"sac_model_ep{self.episode_count}.zip")
                self.model.save(model_path)
                if self.verbose > 0:
                    print(f"💾 Checkpoint saved to {model_path}")
        return True
    
    def _on_training_end(self) -> None:
        self.csv_file.close()

def find_latest_checkpoint(log_dir: str):
    """Finds the latest model checkpoint based on episode number."""
    checkpoints = glob.glob(os.path.join(log_dir, "sac_model_ep*.zip"))
    if not checkpoints:
        return None
    # Extract episode numbers and find the max
    ep_numbers = [int(f.split('ep')[-1].split('.zip')[0]) for f in checkpoints]
    latest_ep = max(ep_numbers)
    return os.path.join(log_dir, f"sac_model_ep{latest_ep}.zip")

def main():
    log_dir = "sac_training_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = AirSimEnv()
    
    # --- Checkpointing and Resumption ---
    latest_checkpoint = find_latest_checkpoint(log_dir)
    if latest_checkpoint:
        print(f"✅ Resuming training from checkpoint: {latest_checkpoint}")
        # Important: You need to pass the custom_objects if you have them, but for basic SAC it's fine.
        model = SAC.load(latest_checkpoint, env=env)
    else:
        print("🚀 Starting a new training run.")
        model = SAC('MlpPolicy', env, verbose=0,
                    buffer_size=150000,
                    learning_rate=0.0003,
                    batch_size=256,
                    gamma=0.99,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    learning_starts=5000,
                    device='cuda') # Change to 'cpu' if you don't have a GPU

    # --- Setup Custom Callback ---
    # This single callback handles both logging and checkpointing
    training_callback = TrainingCallback(log_dir=log_dir, save_freq_episodes=50)

    # --- Start Training ---
    try:
        # If resuming, reset_num_timesteps should be False
        model.learn(total_timesteps=3000000, callback=training_callback, reset_num_timesteps=not bool(latest_checkpoint))
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        final_model_path = os.path.join(log_dir, "sac_model_final.zip")
        model.save(final_model_path)
        print(f"🏁 Final model saved to {final_model_path}")
        env.close()

if __name__ == '__main__':
    main()
