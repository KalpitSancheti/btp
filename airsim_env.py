# airsim_env.py
import time
import numpy as np
import airsim

# --- Simulation / env configuration ---
FRAME_SKIP = 2              # how many internal physics steps per env step (lower -> stronger actions)
INNER_DURATION = 0.08       # seconds per internal velocity command (so env step ~ FRAME_SKIP * INNER_DURATION)
GOAL_Y = 60.0
WAYPOINTS = [7.0, 17.0, 27.5, 45.0, GOAL_Y]
MAX_STEPS_PER_EPISODE = 400

# Tunnel geometry / comfort bands (tune to your scene)
TUNNEL_CENTER_Z = 1.18      # "center" height (positive distance above ground); AirSim z is negative altitude
CENTER_BAND_Z = (0.8, 1.6)  # comfortable z band (positive numbers representing distance above ground)
FREE_CENTER_RADIUS = 0.2    # free radius around centerline without penalty (meters)

# max speeds to scale actions to (m/s)
MAX_SPEED_X = 1.0
MAX_SPEED_Y = 2.4           # forward should be strongest to break stalling
MAX_SPEED_Z = 1.0

# reward clipping
REWARD_CLIP = (-50.0, 50.0)

class Env:
    def __init__(self, vehicle_name=""):
        self.vehicle_name = vehicle_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # physics clock speed doesn't change physics — keep for visuals if you want
        try:
            self.client.simSetTimeOfDay(is_enabled=True, start_datetime="", is_start_datetime_dst=False,
                                       celestial_clock_speed=1.0, update_interval_secs=0.01)
        except Exception:
            pass

        # action/obs sizes (agent should output 3 continuous values in [-1,1])
        self.action_size = 3

        # episode bookkeeping
        self.level = 0
        self.episode_steps = 0
        self.prev_phi = 0.0
        self.last_y = 0.0
        self.stall_counter = 0

        # small safety: enable API control now if not already
        try:
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
        except Exception:
            pass

    def reset(self):
        """Reset environment and return initial observation (depth image, velocity vector)."""
        self.level = 0
        self.episode_steps = 0
        self.stall_counter = 0
        self.prev_phi = 0.0
        self.last_y = 0.0

        # reset and prepare vehicle
        self.client.reset()
        time.sleep(0.05)
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.simPause(False)

        # safer smooth takeoff / position to center band height
        target_z = TUNNEL_CENTER_Z
        try:
            # AirSim uses negative Z for altitude above ground
            self.client.moveToZAsync(-target_z, 1.0, vehicle_name=self.vehicle_name).join()
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass

        quad_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        quad_pos = quad_state.kinematics_estimated.position
        self.last_y = quad_pos.y_val

        # request a depth float image (DepthPlanar gives floats suitable for learning)
        image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True)
        responses = self.client.simGetImages([image_request], self.vehicle_name)

        # initialize potential baseline
        self.prev_phi = self._potential(quad_pos)

        quad_vel = quad_state.kinematics_estimated.linear_velocity
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32)
        observation = [responses, quad_vel]
        return observation

    def step(self, action):
        """
        action: iterable of 3 floats (ax, ay, az) each in [-1,1]
        returns: observation, reward, done, info
        """
        self.episode_steps += 1

        # clamp actions and scale to velocities
        a = [float(action[i]) for i in range(3)]
        ax = float(np.clip(a[0], -1.0, 1.0))
        ay = float(np.clip(a[1], -1.0, 1.0))
        az = float(np.clip(a[2], -1.0, 1.0))

        vx = ax * MAX_SPEED_X
        vy = ay * MAX_SPEED_Y
        vz = az * MAX_SPEED_Z

        total_reward = 0.0
        done = False
        has_collided = False
        quad_pos = None
        quad_vel = None

        for _ in range(FRAME_SKIP):
            # send velocity command
            try:
                self.client.moveByVelocityAsync(vx, vy, vz, INNER_DURATION, vehicle_name=self.vehicle_name).join()
            except Exception:
                # if move fails, sleep small and continue — avoid crashing env
                time.sleep(INNER_DURATION)

            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            if collision_info.has_collided:
                has_collided = True

            quad_state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            quad_pos = quad_state.kinematics_estimated.position
            quad_vel = quad_state.kinematics_estimated.linear_velocity

            dead = has_collided or (quad_pos.y_val < -0.5)
            timeout = (self.episode_steps >= MAX_STEPS_PER_EPISODE)
            done = dead or (quad_pos.y_val >= GOAL_Y) or timeout

            # compute per-inner-step reward (we'll sum them)
            r = self.compute_reward(quad_pos, quad_vel, dead, timeout)
            total_reward += r

            if done:
                # pause sim so agent sees terminal state clearly
                try:
                    self.client.simPause(True)
                except Exception:
                    pass
                break

        # latest depth frame and velocity observation
        image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True)
        responses = self.client.simGetImages([image_request], self.vehicle_name)

        info = {
            'Y': quad_pos.y_val if quad_pos is not None else None,
            'level': self.level,
            'status': 'going',
            'steps': self.episode_steps
        }
        if has_collided:
            info['status'] = 'collision'
        elif quad_pos is not None and quad_pos.y_val < -0.5:
            info['status'] = 'out'
        elif quad_pos is not None and quad_pos.y_val >= GOAL_Y:
            info['status'] = 'goal'
        elif timeout:
            info['status'] = 'timeout'

        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float32)
        observation = [responses, quad_vel]

        # clip reward to avoid exploding gradients
        total_reward = float(np.clip(total_reward, REWARD_CLIP[0], REWARD_CLIP[1]))

        return observation, total_reward, done, info

    # ---------------- reward shaping helpers ----------------
    def _potential(self, pos):
        """
        Potential function Phi(pos) that is higher for:
         - larger forward y (closer to GOAL)
         - smaller center distance (x,z)
         - staying in comfortable z band (not hugging floor or roof)
        Using ΔPhi as part of step reward encourages improvement.
        """
        # forward progress term (linear with y)
        y_term = float(pos.y_val)

        # centerline penalty (x and z distance from centerline)
        radial_x = abs(float(pos.x_val))
        radial_z = abs(float(pos.z_val) - TUNNEL_CENTER_Z)
        center_dist = np.sqrt(radial_x ** 2 + radial_z ** 2)

        # small free tube of radius FREE_CENTER_RADIUS (no penalty within)
        center_pen = max(0.0, center_dist - FREE_CENTER_RADIUS)

        # z band penalty (if outside comfortable height band)
        z_low, z_high = CENTER_BAND_Z
        z_pen = 0.0
        if pos.z_val < z_low:
            z_pen = (z_low - pos.z_val)
        elif pos.z_val > z_high:
            z_pen = (pos.z_val - z_high)

        # weights - tune these if needed
        w_y = 1.0
        w_center = 2.0  # Stronger centering weight to prevent wall crashes
        w_z = 1.0       # Better altitude control

        phi = (w_y * y_term) - (w_center * center_pen) - (w_z * z_pen)
        return float(phi)

    def compute_reward(self, quad_pos, quad_vel, dead, timeout):
        """
        Returns float reward for the current state. Internally uses a potential based shaping:
            r = small_alive + ΔPhi - small_time_cost + soft_velocity_terms - penalties
        Terminal states override to clear large negative or positive rewards.
        """
        # current potential and delta
        phi = self._potential(quad_pos)
        delta_phi = phi - self.prev_phi
        self.prev_phi = phi

        # small alive bonus and tiny time cost (encourages shorter solutions but not too strongly)
        r = 0.05 + delta_phi - 0.001

        # small encouraging direct forward velocity signal (bounded)
        fwd_v = float(quad_vel.y_val)
        r += 0.1 * np.tanh(fwd_v)
        if fwd_v > 0:
            r += 0.02 * fwd_v  

        # mild penalty for lateral movement (discourage drifting / wobbling)
        r -= 0.02 * abs(float(quad_vel.x_val))  # Moderate penalty for sideways movement
        # penalize large vertical velocity (to avoid bouncing to roof/floor)
        r -= 0.01 * max(0.0, abs(float(quad_vel.z_val)) - 0.3)
        
        # **DIRECT**: Additional penalty for being off-center (clearer signal)
        x_distance = abs(float(quad_pos.x_val))
        if x_distance > 0.8:  # If more than 0.8m from center (close to wall)
            r -= (x_distance - 0.8) * 1.0  # Strong penalty for being near walls

        # anti-stall logic: if y hasn't changed for many steps penalize a bit
        dy = float(quad_pos.y_val) - float(self.last_y)
        self.last_y = float(quad_pos.y_val)
        if abs(dy) < 0.002:  # essentially no forward progress
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        if self.stall_counter > int(2.0 / (FRAME_SKIP * INNER_DURATION)):  # ~2 seconds of no progress
            r -= 0.03   

        # small bonus when crossing a waypoint (keeps distributed rewards)
        current_wp = WAYPOINTS[min(self.level, len(WAYPOINTS) - 1)]
        if quad_pos.y_val >= current_wp and self.level < len(WAYPOINTS) - 1:
            self.level += 1
            r += 5.0

        # Terminals — stronger, clearer signals
        if dead:
            # strong negative for collision/out-of-bounds
            return -10.0
        if timeout:
            # small negative but scale by progress so long-lived agents with progress are not harshly punished
            return -1.0 + 0.5 * (quad_pos.y_val / GOAL_Y)
        if quad_pos.y_val >= GOAL_Y:
            # strong success reward
            return r + 20.0

        return float(r)

    def disconnect(self):
        try:
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
            self.client.reset()
        except Exception as e:
            print(f"Error during disconnect: {e}")
        finally:
            print("Disconnected.")
