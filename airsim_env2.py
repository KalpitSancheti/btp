# airsim_env2.py

import time
import numpy as np
import airsim
import config

class Env:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.vehicle_name = ""

        # Your original environment parameters
        self.timeslice = 0.1
        self.goalY = config.GOAL_Y
        self.outY = -0.5
        self.goals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, self.goalY]
        self.target_altitude = -1.7
        self.max_episode_steps = config.MAX_STEPS_PER_EPISODE

        # Your original stall counter
        self.stall_counter = 0
        self.max_stall_steps = 50
        self.prev_y = 0.0

        # Variables for the reward logic
        self.episode_steps = 0
        self.level = 0
        
    def reset(self):
        self.level = 0
        self.episode_steps = 0
        self.stall_counter = 0
        self.prev_y = 0.0

        # reset + stable startup
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        time.sleep(0.2)  # allow simulator a little time to restart

        # try to ensure collision flag is cleared before starting
        for _ in range(20):  # ~1s max (20 * 0.05)
            if not self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).has_collided:
                break
            time.sleep(0.05)
        else:
            # If still stuck, teleport the vehicle slightly above the reset pose (forces clear)
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            # Add a small upward offset to Z to avoid ground collision
            safe_z = float(self.target_altitude) + 1.0
            pose = airsim.Pose(airsim.Vector3r(float(pos.x_val), float(pos.y_val), safe_z))
            # ignore_collision param (teleport) to ensure we free it from stuck colliding state
            self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.vehicle_name)
            time.sleep(0.1)

        # safe takeoff/positioning
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        fast_velocity = getattr(config, "TAKEOFF_VELOCITY", 3.0)
        self.client.moveToPositionAsync(
            float(pos.x_val),
            float(pos.y_val),
            float(self.target_altitude),
            fast_velocity,
            vehicle_name=self.vehicle_name
        ).join()
        time.sleep(0.05)

        # initialize prev_y AFTER settling/teleport/takeoff
        quad_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        self.prev_y = quad_state.kinematics_estimated.position.y_val

        # return state (use keyword for vehicle_name to be explicit)
        image_request = airsim.ImageRequest("0", airsim.ImageType.DepthVis, pixels_as_float=True)
        responses = self.client.simGetImages([image_request], vehicle_name=self.vehicle_name)
        quad_vel = quad_state.kinematics_estimated.linear_velocity
        quad_vel_np = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32)
        return [responses, quad_vel_np]



    def step(self, action):
        self.episode_steps += 1
        
        # Clear collision state at the start of first step
        if self.episode_steps == 1:
            ci = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            if ci.has_collided:
                # teleport to current XY with safe altitude
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                p = state.kinematics_estimated.position
                pose = airsim.Pose(
                    airsim.Vector3r(float(p.x_val), float(p.y_val), float(self.target_altitude))
                )
                self.client.simSetVehiclePose(
                    pose, ignore_collision=True, vehicle_name=self.vehicle_name
                )
                time.sleep(0.05)
        
        action = np.clip(action, -config.ACTION_BOUND, config.ACTION_BOUND)
        
        # Execute action
        self.client.moveByVelocityAsync(
            float(action[0]), float(action[1]), float(action[2]), self.timeslice, vehicle_name=self.vehicle_name
        ).join()
        
        # Check collision after action execution - less sensitive approach
        time.sleep(0.02)  # Small delay to let physics settle
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        has_collided = collision_info.has_collided
        
        # If collision detected, give one chance to recover
        if has_collided:
            time.sleep(0.03)
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            has_collided = collision_info.has_collided

        quad_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        quad_pos = quad_state.kinematics_estimated.position
        quad_vel = quad_state.kinematics_estimated.linear_velocity
        
        progress = quad_pos.y_val - self.prev_y
        if progress < 0.01:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        # Your original termination logic
        dead = has_collided or quad_pos.y_val <= self.outY or self.stall_counter > self.max_stall_steps
        done = dead or (quad_pos.y_val >= self.goalY) or self.episode_steps >= self.max_episode_steps
        
        status = 'going'
        if has_collided: status = 'collision'
        elif quad_pos.y_val <= self.outY: status = 'out'
        elif self.stall_counter > self.max_stall_steps: status = 'stalled'
        elif quad_pos.y_val >= self.goalY: status = 'goal'
        
        reward = self.compute_reward(quad_pos, quad_vel, dead, has_collided)
        
        self.prev_y = quad_pos.y_val

        image_request = airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
        responses = self.client.simGetImages([image_request], self.vehicle_name)
        quad_vel_np = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32)
        next_state = [responses, quad_vel_np]

        info = {'Y': quad_pos.y_val, 'status': status, 'steps': self.episode_steps}

        return next_state, reward, done, info


    def compute_reward(self, quad_pos, quad_vel, dead, has_collided):
        """
        Improved reward function with better stability.
        """
        # --- TERMINAL REWARDS FIRST ---
        if has_collided:
            return config.REWARD_COLLISION
            
        if quad_pos.y_val <= self.outY:
            return config.REWARD_OUT_OF_BOUNDS

        if quad_pos.y_val >= self.goalY:
            return config.REWARD_SUCCESS

        # New: Strong terminal penalty for stalling
        if self.stall_counter > self.max_stall_steps:
            return config.REWARD_STALL

        # --- CONTINUOUS REWARD ---
        reward = 0.0

        # 1. FORWARD PROGRESS REWARD - More stable scaling
        progress = quad_pos.y_val - self.prev_y
        if progress > 0:
            reward += progress * 30.0  # Increased for stronger forward incentive
        elif progress < -0.05:  # Only penalize significant backward movement
            reward += progress * 10.0

        # 2. Base survival reward to encourage staying alive
        reward += 0.3  # Increased per-step reward for exploration and persistence

        # 3. Forward velocity bonus (less aggressive)
        if quad_vel.y_val > 0.5:
            reward += 0.3
        elif quad_vel.y_val < 0.1:
            reward -= 0.2  # Reduced penalty for slow movement

        # 4. Waypoint reward
        if self.level < len(self.goals) and quad_pos.y_val >= self.goals[self.level]:
            reward += config.REWARD_WAYPOINT
            self.level += 1

        # 5. Position-based reward to encourage forward movement
        reward += quad_pos.y_val * 0.01  # Small bonus for being further forward

        # 6. Quadratic centering penalty on X and Z (stronger, still capped)
        try:
            dx = float(quad_pos.x_val)
            dz = float(quad_pos.z_val - self.target_altitude)
            pen = (
                config.CENTER_X_WEIGHT * min(dx**2, 25.0) +
                config.CENTER_Z_WEIGHT * min(dz**2, 25.0)
            )
            reward -= min(pen, config.CENTER_PENALTY_CAP)
        except Exception:
            pass

        return reward


    def disconnect(self):
        self.client.reset()
        self.client.armDisarm(False, self.vehicle_name)
        self.client.enableApiControl(False, self.vehicle_name)
        print('Disconnected.')