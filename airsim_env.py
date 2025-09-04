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
        # **FIX**: Define the vehicle name to be used in API calls
        self.vehicle_name = "" 

    def reset(self):
        """
        Resets the simulation to the initial state.
        """
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        # Takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * TIME_SLICE, vehicle_name=self.vehicle_name).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * TIME_SLICE, vehicle_name=self.vehicle_name).join()
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        self.client.simPause(True)

        # Get initial observation
        quad_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        quad_vel = quad_state.kinematics_estimated.linear_velocity
        
        # **FIX**: Use string for camera name and specify vehicle_name
        image_request = airsim.ImageRequest("1", airsim.ImageType.DepthVis, pixels_as_float=True)
        responses = self.client.simGetImages([image_request], vehicle_name=self.vehicle_name)
        
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]

        # Initialize previous distance to the first waypoint
        quad_pos = quad_state.kinematics_estimated.position
        self.prev_distance = np.linalg.norm([quad_pos.y_val - WAYPOINTS[self.level], quad_pos.x_val, quad_pos.z_val])

        return observation

    def step(self, quad_offset):
        """
        Executes a single step in the simulation.
        """
        quad_offset = [float(i) for i in quad_offset]
        self.client.simPause(False)

        has_collided = False
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], TIME_SLICE, vehicle_name=self.vehicle_name)

        start_time = time.time()
        while time.time() - start_time < TIME_SLICE:
            # Short sleep to avoid spamming the API
            time.sleep(0.01)

        self.client.simPause(True)

        # Get new state information
        quad_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        quad_pos = quad_state.kinematics_estimated.position
        quad_vel = quad_state.kinematics_estimated.linear_velocity
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        has_collided = collision_info.has_collided
        
        # **FIX**: Use string for camera name and specify vehicle_name
        image_request = airsim.ImageRequest("1", airsim.ImageType.DepthVis, pixels_as_float=True)
        responses = self.client.simGetImages([image_request], vehicle_name=self.vehicle_name)

        dead = has_collided or quad_pos.y_val <= OUT_Y
        done = dead or quad_pos.y_val >= GOAL_Y

        reward = self.compute_reward(quad_pos, quad_vel, dead)

        info = {
            'Y': quad_pos.y_val,
            'level': self.level,
            'status': 'going'
        }
        if has_collided:
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
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32)
        speed = np.linalg.norm(vel)
        reward = float(vel[1]) * 0.1

        current_distance = np.linalg.norm([quad_pos.y_val - WAYPOINTS[self.level], quad_pos.x_val, quad_pos.z_val])
        reward -= current_distance * 0.01

        distance_improvement = self.prev_distance - current_distance
        reward += distance_improvement * 0.1
        self.prev_distance = current_distance

        reward -= 0.01

        if dead:
            reward = -10.0
        elif quad_pos.y_val >= WAYPOINTS[self.level]:
            self.level = min(self.level + 1, len(WAYPOINTS) - 1)
            reward = 10.0 * (1 + self.level / len(WAYPOINTS))
        elif speed < SPEED_LIMIT:
            reward = -1.0

        return reward

    def disconnect(self):
        """
        Disconnects from the AirSim simulator.
        """
        self.client.armDisarm(False, self.vehicle_name)
        self.client.enableApiControl(False, self.vehicle_name)
        print('Disconnected.')