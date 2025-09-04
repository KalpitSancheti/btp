import time
import numpy as np
import airsim

# --- Simulation Configuration ---
CLOCKS_PEED = 1
TIME_SLICE = 0.5 / CLOCKS_PEED
GOAL_Y = 57
OUT_Y = -0.5
FLOOR_Z = 1.18
WAYPOINTS = [7, 17, 27.5, 45, GOAL_Y]
SPEED_LIMIT = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

class Env:
    """
    AirSim environment for deep reinforcement learning.
    """
    def __init__(self):
        """
        Connects to the AirSim simulator and initializes the environment.
        """
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_size = 3
        self.level = 0
        self.prev_distance = 0

    def reset(self):
        """
        Resets the simulation to the initial state.
        """
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * TIME_SLICE).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * TIME_SLICE).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)

        # Get initial observation
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]

        # Initialize previous distance to the first waypoint
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        self.prev_distance = np.linalg.norm([quad_pos.y_val - WAYPOINTS[self.level], quad_pos.x_val, quad_pos.z_val])

        return observation

    def step(self, quad_offset):
        """
        Executes a single step in the simulation.
        """
        quad_offset = [float(i) for i in quad_offset]
        self.client.simPause(False)

        has_collided = False
        landed = False
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], TIME_SLICE)

        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < TIME_SLICE:
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

            collided = self.client.simGetCollisionInfo().has_collided
            landed = (quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0) or quad_pos.z_val > FLOOR_Z

            if collided or landed:
                collision_count += 1
            if collision_count > 10:
                has_collided = True
                break
        self.client.simPause(True)

        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        dead = has_collided or quad_pos.y_val <= OUT_Y
        done = dead or quad_pos.y_val >= GOAL_Y

        reward = self.compute_reward(quad_pos, quad_vel, dead)

        info = {
            'Y': quad_pos.y_val,
            'level': self.level,
            'status': 'going'
        }
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.y_val <= OUT_Y:
            info['status'] = 'out'
        elif quad_pos.y_val >= GOAL_Y:
            info['status'] = 'goal'

        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation, reward, done, info

    def compute_reward(self, quad_pos, quad_vel, dead):
        """
        Computes the reward based on the current state.
        """
        # --- Reward for Velocity in Y Direction ---
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32)
        speed = np.linalg.norm(vel)
        reward = float(vel[1]) * 0.1

        # --- Reward for Distance from Waypoints ---
        current_distance = np.linalg.norm([quad_pos.y_val - WAYPOINTS[self.level], quad_pos.x_val, quad_pos.z_val])
        reward -= current_distance * 0.01

        # --- Reward for Improvement in Distance ---
        distance_improvement = self.prev_distance - current_distance
        reward += distance_improvement * 0.1
        self.prev_distance = current_distance

        # --- Time Penalty ---
        reward -= 0.01

        # --- Other Conditions ---
        if dead:
            reward = -10.0
        elif quad_pos.y_val >= WAYPOINTS[self.level]:
            self.level += 1
            reward = 10.0 * (1 + self.level / len(WAYPOINTS))
        elif speed < SPEED_LIMIT:
            reward = -1.0

        return reward

    def disconnect(self):
        """
        Disconnects from the AirSim simulator.
        """
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')