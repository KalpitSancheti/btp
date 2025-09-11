import airsim
import numpy as np
import math
import time
import gym
from gym import spaces

class AirSimEnv(gym.Env):
    """
    A custom Gym environment for training a drone in AirSim using LiDAR data.
    """
    def __init__(self, ip_address="127.0.0.1", vehicle_name="SimpleFlight"):
        super(AirSimEnv, self).__init__()

        # --- AirSim Connection ---
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name)
        self.client.armDisarm(True, vehicle_name)
        self.vehicle_name = vehicle_name

        # --- Lidar Parameters ---
        self.lidar_name = "LidarSensor1"
        self.num_lidar_bins = 15  # Focused forward bins for precision
        self.max_lidar_range = 50.0

        # --- State/Observation Space (Normalized) ---
        # 15 LiDAR bins + 3 velocity components (vx, vy, vz) = 18-dimensional state
        self.observation_space = spaces.Box(
            low=np.full(self.num_lidar_bins + 3, -1.0),
            high=np.full(self.num_lidar_bins + 3, 1.0),
            dtype=np.float32
        )

        # --- Action Space (Continuous & Normalized) ---
        # Action[0]: Forward velocity (-1 to 1 mapped to 0 to max_speed)
        # Action[1]: Yaw rate (-1 to 1 mapped to -max_turn_rate to +max_turn_rate)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.max_speed = 6.0
        self.max_turn_rate = 90.0

        # --- Episode State Variables ---
        self.goal_y = 60.0
        self.last_y_pos = 0.0
        self.max_y_pos_episode = 0.0
        self.episode_step_count = 0

    def _get_observation(self):
        """Processes sensor data into a normalized observation vector."""
        # Get drone kinematics
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        vel = state.kinematics_estimated.linear_velocity

        # Normalize velocity and clip to ensure it's within [-1, 1]
        norm_vx = np.clip(vel.x_val / self.max_speed, -1.0, 1.0)
        norm_vy = np.clip(vel.y_val / self.max_speed, -1.0, 1.0)
        norm_vz = np.clip(vel.z_val / self.max_speed, -1.0, 1.0)
        velocity_state = np.array([norm_vx, norm_vy, norm_vz])

        # Get and process LiDAR data
        lidar_data = self.client.getLidarData(lidar_name=self.lidar_name, vehicle_name=self.vehicle_name)
        # Initialize bins to max range (normalized to 1.0)
        lidar_bins = np.ones(self.num_lidar_bins, dtype=np.float32)

        if len(lidar_data.point_cloud) >= 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            for x, y, z in points:
                # We only care about the forward 120-degree cone (Y > 0, tighter focus)
                if y <= 0:
                    continue

                distance = math.sqrt(x**2 + y**2)
                # Angle relative to forward Y-axis
                angle = math.atan2(x, y) * (180 / math.pi)

                # Map angle from [-60, 60] to a bin index [0, num_bins-1]
                if -60 < angle < 60:
                    angle_normalized = angle + 60  # Shift range to [0, 120]
                    bin_index = int(angle_normalized / (120 / self.num_lidar_bins))
                    # Update bin with the closest point found (normalized distance)
                    norm_dist = distance / self.max_lidar_range
                    if norm_dist < lidar_bins[bin_index]:
                        lidar_bins[bin_index] = norm_dist

        # Combine lidar and velocity into the final observation vector
        return np.concatenate((lidar_bins, velocity_state))

    def _compute_reward_and_done(self):
        """Calculates the reward and determines if the episode is finished."""
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        info = {}

        # Update max y position reached in this episode
        self.max_y_pos_episode = max(self.max_y_pos_episode, pos.y_val)
        info['max_y_distance'] = self.max_y_pos_episode

        # Reward for forward progress
        progress = pos.y_val - self.last_y_pos
        self.last_y_pos = pos.y_val
        reward = (progress * 100)/(pos.y_val + self.last_y_pos) # High incentive for moving forward
        

        # Penalty for being too close to obstacles (encourages staying in the center)
        lidar_obs = self._get_observation()[:self.num_lidar_bins]
        min_lidar_dist = np.min(lidar_obs) * self.max_lidar_range
        # if min_lidar_dist < 2.0:
        #     reward -= (2.0 - min_lidar_dist) # Penalize proximity

        # Large penalty for collision
        if self.client.simGetCollisionInfo().has_collided:
            info['termination_reason'] = 'collision'
            return -100, True, info

        # Large reward for reaching the goal
        if pos.y_val >= self.goal_y:
            info['termination_reason'] = 'goal_reached'
            return 200, True, info

        # End episode if it takes too long
        if self.episode_step_count >= 1500:
            info['termination_reason'] = 'max_steps_reached'
            return 0, True, info

        return reward, False, info

    def step(self, action):
        self.episode_step_count += 1

        # De-normalize actions from [-1, 1] to physical values
        # Map velocity action [-1, 1] to [0, max_speed] for forward-only motion
        forward_vel = (action[0] + 1) / 2 * self.max_speed
        yaw_rate = action[1] * self.max_turn_rate


        # Move forward with specified velocity
        self.client.moveByVelocityAsync(
            vx=0, vy=float(forward_vel), vz=0,
            duration=0.1,
            vehicle_name=self.vehicle_name
        ).join()

        # Apply yaw rate separately
        if abs(yaw_rate) > 1e-3:
            self.client.rotateByYawRateAsync(
                yaw_rate=float(yaw_rate),
                duration=0.1,
                vehicle_name=self.vehicle_name
            ).join()

        obs = self._get_observation()
        reward, done, info = self._compute_reward_and_done()

        return obs, reward, done, info

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        initial_pos = self.client.getMultirotorState().kinematics_estimated.position
        self.last_y_pos = initial_pos.y_val
        self.max_y_pos_episode = self.last_y_pos
        self.episode_step_count = 0

        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.client.moveToZAsync(-3, 2, vehicle_name=self.vehicle_name).join() # Start at -3m height

        return self._get_observation()

    def close(self):
        self.client.reset()
        self.client.armDisarm(False, self.vehicle_name)
        self.client.enableApiControl(False, self.vehicle_name)
