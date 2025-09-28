# airsim_sac_env.py
"""
Safer AirSim RL environment for tunnel flight.

This file includes:
 - Dynamic detection of LiDAR z frame (world vs sensor local)
 - Corrected altitude sign handling (descend = less-negative z)
 - Stronger window forcing & longer hold during window approach
 - Enabled curriculum (15m -> 30m) by default
 - All original overrides (lateral, brute force, window exploration, emergency altitude)
 - Clamping and safety checks to avoid hitting ground
 - Debug hooks via `debug_verbose=True`
"""

import airsim
import numpy as np
import math
import time
import collections
import gymnasium as gym
from gymnasium import spaces


class AirSimEnv(gym.Env):
    def __init__(self,
                 ip_address="127.0.0.1",
                 vehicle_name="SimpleFlight",
                 num_lidar_bins=15,
                 max_lidar_range=50.0,
                 max_speed=4.0,
                 max_lat_speed=1.5,
                 goal_y=30.0,
                 tunnel_width_x=5.0,
                 tunnel_height=5.0,
                 dt=0.2,
                 max_steps=1500,
                 desired_z=-1.5,
                 operational_z_offset=-0.3,
                 checkpoints=None,
                 checkpoint_bonus=80.0,
                 reactive_safe_dist=2.5,
                 reactive_ttc_thresh=0.6,
                 window_safe_dist=1.5,
                 # vertical override params (safer defaults)
                 ceiling_warn_thresh=2.0,
                 ceiling_stop_thresh=1.0,
                 ceiling_override_thresh=1.5,
                 ceiling_override_down=1.2,   # larger to force meaningful descent
                 vertical_override_cooldown=1, # allow faster subsequent nudges
                 pos_override_down=0.9,
                 # reward & behavior tuning
                 center_reward_scale=8.0,
                 center_penalty_scale=-3.0,
                 upward_penalty_coeff=8.0,
                 reward_forward_scale=12.0,
                 alive_bonus=0.06,
                 time_penalty=-0.01,
                 collision_penalty=-300.0,
                 goal_reward=800.0,
                 startup_grace_seconds=1.6,
                 debug_verbose=False):
        super().__init__()

        # AirSim client
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name)
        self.client.armDisarm(True, vehicle_name)
        self.vehicle_name = vehicle_name

        # sensors
        self.lidar_name = "LidarSensor1"
        self.num_lidar_bins = int(num_lidar_bins)
        self.max_lidar_range = float(max_lidar_range)

        # action / obs spaces (enhanced with more features)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)
        # obs: lidar(15) + velocity(3) + [dist_to_goal, height_norm] + [ceiling_norm, floor_norm] +
        # prev_action(2) + progress_rate(1) + tunnel_center_offset(1) + time_since_collision(1) = 27
        obs_dim = self.num_lidar_bins + 3 + 2 + 2 + 2 + 1 + 1 + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # dynamics / geometry
        self.max_speed = float(max_speed)
        self.max_lat_speed = float(max_lat_speed)
        self.dt = float(dt)
        self.original_goal_y = float(goal_y)
        self.goal_y = 30.0  # default; curriculum may change
        self.curriculum_stage = -1
        self.curriculum_success_count = 0
        self.curriculum_episodes_in_stage = 0
        self.tunnel_half_width_x = float(tunnel_width_x) / 2.0
        self.tunnel_half_height = float(tunnel_height) / 2.0

        # --- DESIRED Z defaults ---
        # Use a slightly higher cruise (more negative) so window descent (less negative) is meaningful
        if desired_z == -1.5:
            self.desired_z = -2.0
        else:
            self.desired_z = float(desired_z)

        self.operational_z_offset = float(operational_z_offset)
        self.operational_z = self.desired_z + self.operational_z_offset

        # Window-specific altitude control (LESS negative => closer to ground)
        self.window_desired_z = -0.6
        self.window_center_z = -0.6

        # Window exploration parameters
        self.window_exploration_epsilon = 0.9
        self.window_exploration_decay = 0.9995
        self.exploration_episode_count = 0
        self.operational_z_tolerance = 0.20

        # vertical override & safety
        self.ceiling_warn_thresh = float(ceiling_warn_thresh)
        self.ceiling_stop_thresh = float(ceiling_stop_thresh)
        self.ceiling_override_thresh = float(ceiling_override_thresh)
        self.ceiling_override_down = float(ceiling_override_down)
        self.vertical_override_cooldown = int(vertical_override_cooldown)
        self.pos_override_down = float(pos_override_down)
        self._last_vertical_override_step = -999

        # forced temporary operational altitude
        self._forced_operational_z = None
        self._forced_operational_z_until_step = -1
        self._forced_operational_z_duration_steps = max(2, int(self.vertical_override_cooldown))

        # checkpoints for progress tracking
        if checkpoints is None:
            checkpoints = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
            self.checkpoints = [float(cp) for cp in checkpoints]
        else:
            self.checkpoints = sorted([float(c) for c in checkpoints])
        self.checkpoint_bonus = float(checkpoint_bonus)

        # Window navigation parameters
        self.window_y_position = 25.0
        self.window_approach_zone = 3.0
        self.window_center_x = 0.0
        self.window_tolerance_x = 1.2
        self.window_tolerance_z = 0.8

        # episode bookkeeping
        self.max_steps = int(max_steps)
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.last_y_pos = 0.0
        self.max_y_pos_episode = 0.0
        self._reached_checkpoints = set()

        # reward tuning
        self.reward_forward_scale = float(reward_forward_scale)
        self.reward_shaping_scale = 0.0
        self.center_reward_scale = float(center_reward_scale)
        self.center_penalty_scale = float(center_penalty_scale)
        self.obstacle_penalty_scale = -1.0
        self.alive_bonus = float(alive_bonus)
        self.time_penalty = float(time_penalty)
        self.collision_penalty = float(collision_penalty)
        self.goal_reward = float(goal_reward)
        self.safe_dist = float(reactive_safe_dist)
        self.window_safe_dist = float(window_safe_dist)

        # reactive horizontal params
        self.reactive_ttc_thresh = float(reactive_ttc_thresh)
        self.override_count = 0
        self.vertical_override_count = 0
        self.lateral_override_count = 0

        # upward penalty
        self.upward_penalty_coeff = float(upward_penalty_coeff)

        # startup grace
        self.startup_grace_seconds = float(startup_grace_seconds)
        self.startup_grace_steps = max(1, int(round(self.startup_grace_seconds / self.dt)))

        # debug / trackers
        self._debug_verbose = bool(debug_verbose)
        self.min_clear_above = float('inf')
        self._last_z = None

        # collision recovery window
        self._collision_window = collections.deque([0] * 12, maxlen=12)
        self._recent_collision_count = 0

        # enhanced observation tracking
        self._prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self._prev_prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self._last_collision_step = -1000
        self._progress_history = collections.deque([0.0] * 5, maxlen=5)

        # random seed
        self._seed = None

        # Initialize curriculum to stage 0 if unset
        if self.curriculum_stage < 0:
            self.curriculum_stage = 0
            self.curriculum_success_count = 0
            self.curriculum_episodes_in_stage = 0

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)

    # ---------- dynamic operational z (honors forced override) ----------
    def _get_operational_z(self):
        # if forced override active, honor it
        if self.episode_step_count <= self._forced_operational_z_until_step and (self._forced_operational_z is not None):
            return self._forced_operational_z
        # otherwise use base desired_z + offset
        return self.desired_z + self.operational_z_offset

    def _detect_collision_type(self, collision_info, drone_pos):
        """
        Improved collision type detection using multiple criteria
        """
        try:
            cp_x = float(collision_info.position.x_val)
            cp_y = float(collision_info.position.y_val)
            cp_z = float(collision_info.position.z_val)

            drone_x = float(drone_pos.x_val)
            drone_y = float(drone_pos.y_val)
            drone_z = float(drone_pos.z_val)

            # Calculate relative positions
            dx = abs(cp_x - drone_x)
            dy = abs(cp_y - drone_y)
            dz = cp_z - drone_z

            # Collision type logic based on tunnel geometry and collision position

            # 1. Roof collision: collision point significantly above drone
            if dz > 0.5:  # Collision point is 0.5m+ above drone
                return 'roof'

            # 2. Floor collision: collision point significantly below drone
            elif dz < -0.8:  # Collision point is 0.8m+ below drone
                return 'floor'

            # 3. Wall collision: large lateral displacement
            elif dx > 1.5:  # Collision point is 1.5m+ away laterally
                return 'wall'

            # 4. Forward obstacle: collision ahead in flight direction
            elif dy > 2.0:  # Collision point is ahead of drone
                return 'obstacle'

            # 5. Use lidar data to get more context
            try:
                clear_above, clear_below, _, _ = self._compute_vertical_clearance()
                if np.isfinite(clear_above) and clear_above < 1.0:
                    return 'roof'  # Very close to ceiling
                elif np.isfinite(clear_below) and clear_below < 1.0:
                    return 'floor'  # Very close to floor
            except Exception:
                pass

            # 6. Fallback: use original simple logic but with better thresholds
            if dz > 0.2:  # More conservative roof detection
                return 'roof'
            elif dz < -0.3:  # Floor detection
                return 'floor'
            else:
                return 'wall'  # Default to wall for side collisions

        except Exception as e:
            if self._debug_verbose:
                print(f"[COLLISION_DETECT_ERROR] {e}")
            return 'unknown'

    def _update_curriculum(self, termination_reason, max_y_distance):
        """
        Simple curriculum: Stage 0 -> goal_y=15m (learn basic flight)
                         Stage 1 -> goal_y=30m (final with window)
        This increments the per-stage episode counter and moves forward when success ratio
        exceeds thresholds.
        """
        # Define curriculum stages
        stages = [  # Stage 0
            {"goal_y": 45.0, "success_threshold": 0.20, "min_episodes": 80},  # Stage 1
        ]

        # Ensure stage index valid
        if self.curriculum_stage < 0:
            self.curriculum_stage = 0

        # Track episodes in stage
        self.curriculum_episodes_in_stage += 1

        # Count success (reaching goal or getting close)
        if termination_reason == 'goal_reached' or max_y_distance >= stages[self.curriculum_stage]["goal_y"] * 0.9:
            self.curriculum_success_count += 1

        # Only attempt stage advance after some minimum episodes
        cur_stage = stages[self.curriculum_stage]
        if (self.curriculum_episodes_in_stage >= cur_stage["min_episodes"]
                and (self.curriculum_success_count / max(1, self.curriculum_episodes_in_stage)) >= cur_stage["success_threshold"]):
            # advance stage
            self.curriculum_stage = min(self.curriculum_stage + 1, len(stages) - 1)
            self.goal_y = stages[self.curriculum_stage]["goal_y"]
            print(f"[CURRICULUM] Advanced to stage {self.curriculum_stage}: goal_y = {self.goal_y}m")
            # reset counters
            self.curriculum_success_count = 0
            self.curriculum_episodes_in_stage = 0
        else:
            # keep current target goal
            self.goal_y = stages[self.curriculum_stage]["goal_y"]

    def _lateral_override(self, action, drone_pos):
        """
        Virtual guardrail system: Override lateral movement to prevent wall collisions
        """
        try:
            x_pos = float(drone_pos.x_val)
            lateral_action = float(action[1])  # action[1] is lateral movement

            # Define virtual barriers (slightly inside actual tunnel walls)
            left_barrier = -2.0   # Left wall at -2.5, barrier at -2.0
            right_barrier = 2.0   # Right wall at +2.5, barrier at +2.0

            # Strong override zone (close to walls)
            strong_left = -1.8
            strong_right = 1.8

            # Check if override is needed
            override_needed = False
            new_lateral_action = lateral_action

            # Strong override: Very close to walls
            if x_pos <= strong_left and lateral_action < 0:
                # Too far left and trying to go more left
                new_lateral_action = 0.8  # Force right movement
                override_needed = True
            elif x_pos >= strong_right and lateral_action > 0:
                # Too far right and trying to go more right  
                new_lateral_action = -0.8  # Force left movement
                override_needed = True

            # Soft override: Approaching barriers
            elif x_pos <= left_barrier and lateral_action < 0:
                # Approaching left wall and trying to go left
                new_lateral_action = max(0.0, lateral_action * 0.3)  # Reduce/reverse left movement
                override_needed = True
            elif x_pos >= right_barrier and lateral_action > 0:
                # Approaching right wall and trying to go right
                new_lateral_action = min(0.0, lateral_action * 0.3)  # Reduce/reverse right movement
                override_needed = True

            if override_needed:
                self.lateral_override_count += 1
                if self._debug_verbose:
                    print(f"[LATERAL_OVERRIDE] x_pos: {x_pos:.2f}, action: {lateral_action:.2f} → {new_lateral_action:.2f}")

                # Return modified action
                new_action = action.copy()
                new_action[1] = new_lateral_action
                return new_action

            return None  # No override needed

        except Exception as e:
            if self._debug_verbose:
                print(f"[LATERAL_OVERRIDE_ERROR] {e}")
            return None

    def _detect_window_opening(self, bins_m, pos):
        """
        Detect if there's a window opening ahead using LIDAR data
        Returns: (has_opening, opening_direction, confidence)
        """
        try:
            y_pos = float(pos.y_val)

            # Only check for window opening when approaching the critical window
            if not (20.0 <= y_pos <= 28.0):
                return False, 0.0, 0.0

            center_idx = self.num_lidar_bins // 2

            # Look for a "gap" in the center bins (window opening)
            center_bins = bins_m[center_idx-2:center_idx+3]  # 5 center bins
            side_bins = np.concatenate([bins_m[:3], bins_m[-3:]])  # Side bins

            center_clearance = np.mean(center_bins)
            side_clearance = np.mean(side_bins)

            # Window opening detected if center is much clearer than sides
            opening_threshold = 3.0  # meters
            if center_clearance > opening_threshold and center_clearance > side_clearance * 1.5:
                # Check vertical clearance too
                clear_above, clear_below, _, _ = self._compute_vertical_clearance()
                if np.isfinite(clear_above) and clear_above > 0.8:  # Enough vertical space
                    confidence = min(1.0, (center_clearance - opening_threshold) / 5.0)
                    return True, 0.0, confidence  # Opening straight ahead

            return False, 0.0, 0.0

        except Exception as e:
            if self._debug_verbose:
                print(f"[WINDOW_DETECT_ERROR] {e}")
            return None

    def _window_exploration_override(self, intended_action, pos):
        """
        AGGRESSIVE exploration near window to break the 25m barrier
        """
        try:
            y_pos = float(pos.y_val)
            current_z = float(pos.z_val)

            # Only apply near window where agent gets stuck (expanded range)
            if not (20.0 <= y_pos <= 27.0):
                return None

            # ALWAYS apply exploration in this critical zone
            exploration_strategies = [
                # Strategy 1: Go lower and forward
                [0.6, 0.0],  # Forward, centered
                # Strategy 2: Go lower and left
                [0.4, -0.7],  # Forward-left
                # Strategy 3: Go lower and right  
                [0.4, 0.7],   # Forward-right
                # Strategy 4: Slow and low
                [0.2, 0.0],   # Very slow forward
                # Strategy 5: Aggressive forward
                [0.9, 0.0],   # Fast forward
            ]

            # Choose random strategy based on epsilon-exploration
            if np.random.rand() < self.window_exploration_epsilon:
                strategy = exploration_strategies[np.random.randint(len(exploration_strategies))]
                new_action = np.array(strategy, dtype=np.float32)
            else:
                new_action = None

            # CORRECT altitude fix - FORCE DESCENT (toward ground = less negative)
            try:
                # If it's above the window center (more negative than target), or with some prob, attempt direct moveToZ
                if current_z < (self.window_desired_z - 0.15) or np.random.rand() < 0.25:
                    target_z = self.window_desired_z  # use configured lower window altitude (less negative)

                    # Direct position override - bypass all other systems (best-effort)
                    try:
                        self.client.moveToZAsync(target_z, velocity=2.5, timeout_sec=0.8, vehicle_name=self.vehicle_name).join()
                        if self._debug_verbose:
                            print(f"[WINDOW_EXPLORE][MOVEZ] y:{y_pos:.1f} z:{current_z:.2f} -> {target_z:.2f}")
                    except Exception:
                        pass

                    # Force operational altitude to window_desired_z for a longer span
                    self._forced_operational_z = target_z
                    self._forced_operational_z_until_step = self.episode_step_count + max(20, int(1.0 / max(1e-6, self.dt)))
                    if self._debug_verbose:
                        print(f"[WINDOW_EXPLORE][FORCE_OPZ] forced_op_z={self._forced_operational_z} until step={self._forced_operational_z_until_step}")

            except Exception as e:
                if self._debug_verbose:
                    print(f"[WINDOW_EXPLORE_ERROR_ALT] {e}")

            # Decay exploration slowly so it remains aggressive longer
            if self.episode_step_count % 8 == 0:
                self.window_exploration_epsilon *= self.window_exploration_decay
                self.window_exploration_epsilon = max(0.35, self.window_exploration_epsilon)

            if self._debug_verbose:
                print(f"[WINDOW_EXPLORE] y:{y_pos:.1f} z:{current_z:.1f} eps:{self.window_exploration_epsilon:.3f} strategy:{new_action}")

            return new_action

        except Exception as e:
            if self._debug_verbose:
                print(f"[WINDOW_EXPLORE_ERROR] {e}")
            return None

    def _emergency_altitude_fix(self, pos):
        """
        Emergency altitude correction when drone hits window frame repeatedly
        """
        try:
            y_pos = float(pos.y_val)
            current_z = float(pos.z_val)

            # Emergency intervention at window approach
            # If flying too HIGH (more negative) above the window approach, force descent
            if 22.0 <= y_pos <= 25.5 and current_z < (self.window_center_z - 0.3):
                # IMMEDIATE descent to safe window altitude (LESS negative = lower)
                safe_z = self.window_desired_z

                if self._debug_verbose:
                    print(f"[EMERGENCY_ALT] y:{y_pos:.1f} z:{current_z:.1f} -> EMERGENCY DESCENT TO {safe_z}m (LOWER)")

                # Direct altitude command
                try:
                    self.client.moveToZAsync(safe_z, velocity=3.0, timeout_sec=0.5, vehicle_name=self.vehicle_name).join()
                    return True
                except Exception:
                    # Fallback
                    self._forced_operational_z = safe_z
                    self._forced_operational_z_until_step = self.episode_step_count + 10
                    return True

            return False

        except Exception as e:
            if self._debug_verbose:
                print(f"[EMERGENCY_ALT_ERROR] {e}")
            return False

    def _brute_force_navigate(self, intended_action, pos):
        """
        Brute force LIDAR-based navigation - analyze obstacles and force movement to clear path
        """
        try:
            bins_m = self._compute_polar_histogram()
            y_pos = float(pos.y_val)

            # Only apply brute force near critical areas (window approach)
            if not (20.0 <= y_pos <= 28.0):
                return None

            # Get vertical clearance
            clear_above, clear_below, _, _ = self._compute_vertical_clearance()

            # Analyze horizontal LIDAR pattern
            center_idx = self.num_lidar_bins // 2
            left_bins = bins_m[:center_idx-1]
            center_bins = bins_m[center_idx-1:center_idx+2]  # 3 center bins
            right_bins = bins_m[center_idx+1:]

            min_forward = np.min(center_bins) if len(center_bins) > 0 else float('inf')
            min_left = np.min(left_bins) if len(left_bins) > 0 else float('inf')
            min_right = np.min(right_bins) if len(right_bins) > 0 else float('inf')

            # Decision logic based on LIDAR analysis
            new_action = intended_action.copy()
            override_applied = False

            # If forward path blocked but sides are clear
            if min_forward < 2.0:
                if min_left > min_right and min_left > 3.0:
                    new_action[1] = -0.6  # Move left
                    new_action[0] = max(0.2, intended_action[0] * 0.7)
                    override_applied = True
                elif min_right > 3.0:
                    new_action[1] = 0.6  # Move right
                    new_action[0] = max(0.2, intended_action[0] * 0.7)
                    override_applied = True

            # Vertical navigation if horizontal is blocked
            if min_forward < 1.5 and not override_applied:
                if np.isfinite(clear_above) and clear_above > 1.0:
                    # Try to go up slightly (not ideal — prefer to stay low)
                    new_action[0] = 0.1
                elif np.isfinite(clear_below) and clear_below > 1.0:
                    new_action[0] = 0.1
                    override_applied = True

            # Window opening: go for it
            center_clearance = np.mean(center_bins) if len(center_bins) > 0 else 0.0
            if center_clearance > 4.0:
                new_action[0] = 0.8  # Full speed ahead
                new_action[1] = new_action[1] * 0.3  # Minimal lateral movement
                override_applied = True

                if self._debug_verbose:
                    print(f"[BRUTE_FORCE] Clear path detected: {center_clearance:.1f}m - FULL SPEED AHEAD!")

            if override_applied:
                if self._debug_verbose:
                    print(f"[BRUTE_FORCE] y:{y_pos:.1f} fwd:{min_forward:.1f} left:{min_left:.1f} right:{min_right:.1f} action:{intended_action} -> {new_action}")
                return new_action

            return None

        except Exception as e:
            if self._debug_verbose:
                print(f"[BRUTE_FORCE_ERROR] {e}")
            return None

    # ---------- LIDAR helpers ----------
    def _compute_polar_histogram(self):
        bins = np.full(self.num_lidar_bins, self.max_lidar_range, dtype=np.float32)
        lidar = self.client.getLidarData(lidar_name=self.lidar_name, vehicle_name=self.vehicle_name)
        if len(lidar.point_cloud) < 3:
            return bins
        pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        for x, y, z in pts:
            # consider only forward hemisphere (y > 0)
            if y <= 0:
                continue
            dist = math.hypot(x, y)
            if dist < 0.05:
                continue
            angle = math.degrees(math.atan2(x, y))
            if -60.0 <= angle <= 60.0:
                a_norm = angle + 60.0
                idx = int(a_norm / (120.0 / self.num_lidar_bins))
                idx = min(max(idx, 0), self.num_lidar_bins - 1)
                bins[idx] = min(bins[idx], dist)
        return bins

    def _compute_vertical_clearance(self, forward_dist=6.0, forward_angle_deg=30.0):
        """
        Returns: clearance_above (m), clearance_below (m), highest_z (world) or None, lowest_z or None

        This function dynamically decides whether LiDAR returns points in sensor-local z offsets
        (in which case we add drone_z to convert to world z) or already returns world z. The
        decision is made based on the distribution of z values in the point cloud.
        """
        lidar = self.client.getLidarData(lidar_name=self.lidar_name, vehicle_name=self.vehicle_name)
        if len(lidar.point_cloud) < 3:
            return float('inf'), float('inf'), None, None
        pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        drone_z = float(state.kinematics_estimated.position.z_val)

        # decide frame for lidar z: if median absolute z is small (<0.5), treat z as local offsets
        z_vals = pts[:, 2]
        median_abs_z = np.median(np.abs(z_vals))
        median_z = np.median(z_vals)

        # Heuristics:
        # - if lidar z offsets are small (median_abs_z < 0.5), likely local-frame offsets -> add drone_z
        # - if median_z is close to drone_z (within 0.5m), lidar likely returns world coords (use directly)
        # - otherwise default to local offsets (safer)
        use_local_offsets = False
        if median_abs_z < 0.5:
            use_local_offsets = True
        elif abs(median_z - drone_z) < 0.5:
            use_local_offsets = False
        else:
            # ambiguous => prefer local offsets but record debug trace
            use_local_offsets = True

        clearance_above = float('inf')
        clearance_below = float('inf')
        highest_z = -1e9
        lowest_z = 1e9

        for x, y, z in pts:
            if y <= 0:
                continue
            horiz = math.hypot(x, y)
            if horiz > forward_dist:
                continue
            ang = abs(math.degrees(math.atan2(x, y)))
            if ang > forward_angle_deg:
                continue
            if use_local_offsets:
                pt_world_z = drone_z + z
            else:
                pt_world_z = z
            if pt_world_z > highest_z:
                highest_z = pt_world_z
            if pt_world_z < lowest_z:
                lowest_z = pt_world_z
            if pt_world_z > drone_z:
                d_above = pt_world_z - drone_z
                if d_above < clearance_above:
                    clearance_above = d_above
            elif pt_world_z < drone_z:
                d_below = drone_z - pt_world_z
                if d_below < clearance_below:
                    clearance_below = d_below

        if highest_z == -1e9:
            highest_z = None
        if lowest_z == 1e9:
            lowest_z = None

        # update tracker
        if np.isfinite(clearance_above):
            self.min_clear_above = min(self.min_clear_above, float(clearance_above))

        if self._debug_verbose:
            frame_type = "local_offsets" if use_local_offsets else "world_coords"
            print(f"[VERT_CLEAR] drone_z={drone_z:.3f} median_z={median_z:.3f} median_abs_z={median_abs_z:.3f} frame={frame_type} "
                  f"clear_above={clearance_above:.3f} clear_below={clearance_below:.3f}")

        return clearance_above, clearance_below, highest_z, lowest_z

    # ---------- observations ----------
    def _get_observation(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        vel = state.kinematics_estimated.linear_velocity
        pos = state.kinematics_estimated.position

        vx = np.clip(vel.x_val / max(1e-6, self.max_lat_speed), -1.0, 1.0)
        vy = np.clip(vel.y_val / max(1e-6, self.max_speed), -1.0, 1.0)
        vz = np.clip(vel.z_val / max(1e-6, abs(self._get_operational_z()) + 1.0), -1.0, 1.0)
        velocity_state = np.array([vx, vy, vz], dtype=np.float32)

        bins = np.clip(self._compute_polar_histogram() / max(1e-6, self.max_lidar_range), 0.0, 1.0).astype(np.float32)

        dist_to_goal = max(0.0, self.goal_y - pos.y_val)
        dist_to_goal_norm = np.clip(dist_to_goal / max(1e-6, self.goal_y), 0.0, 1.0)
        height_norm = np.clip((pos.z_val - self._get_operational_z()) / 10.0, -1.0, 1.0)

        clear_above, clear_below, _, _ = self._compute_vertical_clearance(forward_dist=6.0, forward_angle_deg=30.0)
        max_check = 10.0
        ceiling_norm = np.clip(clear_above / max_check, 0.0, 1.0) if np.isfinite(clear_above) else 1.0
        floor_norm = np.clip(clear_below / max_check, 0.0, 1.0) if np.isfinite(clear_below) else 1.0

        # Previous action (helps with action consistency)
        prev_action_norm = self._prev_action.copy()

        # Progress rate (recent forward movement trend)
        current_progress = pos.y_val - self.last_y_pos if hasattr(self, 'last_y_pos') else 0.0
        self._progress_history.append(current_progress)
        progress_rate = np.mean(self._progress_history) / max(1e-6, self.dt * self.max_speed)
        progress_rate_norm = np.clip(progress_rate, -1.0, 1.0)

        # Tunnel center offset (lateral position relative to tunnel center)
        tunnel_center_offset = pos.x_val / max(1e-6, self.tunnel_half_width_x)
        tunnel_center_offset_norm = np.clip(tunnel_center_offset, -1.0, 1.0)

        # Time since last collision (helps with collision avoidance learning)
        steps_since_collision = self.episode_step_count - self._last_collision_step
        time_since_collision = min(steps_since_collision * self.dt, 10.0) / 10.0  # Normalize to [0,1]

        obs = np.concatenate((
            bins,
            velocity_state,
            np.array([dist_to_goal_norm, height_norm, ceiling_norm, floor_norm], dtype=np.float32),
            prev_action_norm,
            np.array([progress_rate_norm, tunnel_center_offset_norm, time_since_collision], dtype=np.float32)
        ))
        return obs

    # ---------- horizontal reactive override ----------
    def _horizontal_override(self, intended_action):
        try:
            bins_m = self._compute_polar_histogram()
        except Exception:
            return None

        # Get current position for window-aware logic
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position

        # Check for window opening
        has_opening, opening_dir, confidence = self._detect_window_opening(bins_m, pos)

        # Use different safe distance near window
        y_pos = float(pos.y_val)
        if 20.0 <= y_pos <= 28.0:  # Near window
            current_safe_dist = self.window_safe_dist
        else:
            current_safe_dist = self.safe_dist

        min_dist = float(np.min(bins_m))
        fwd_norm = float(np.clip(intended_action[0], -1.0, 1.0))
        fwd_speed = (fwd_norm + 1.0) / 2.0 * self.max_speed
        if fwd_speed < 1e-6:
            ttc = float('inf')
        else:
            ttc = (min_dist - current_safe_dist) / max(1e-6, fwd_speed)

        # If window opening detected, be less aggressive about avoiding
        if has_opening and confidence > 0.3:
            # Encourage forward movement through opening
            if self._debug_verbose:
                print(f"[WINDOW_OPENING] Detected opening, confidence: {confidence:.2f}")
            # Only override if very close to obstacles
            override_threshold = current_safe_dist * 0.7

            # Also when opening detected, force op_z to window_desired_z to encourage descent
            try:
                self._forced_operational_z = self.window_desired_z
                self._forced_operational_z_until_step = self.episode_step_count + max(15, int(0.7 / max(1e-6, self.dt)))
                if self._debug_verbose:
                    print(f"[WINDOW_OPENING][FORCE_OPZ] forced_op_z={self._forced_operational_z} until={self._forced_operational_z_until_step}")
            except Exception:
                pass
        else:
            override_threshold = current_safe_dist

        if min_dist < override_threshold or ttc < self.reactive_ttc_thresh:
            safe_bins = np.where(bins_m > current_safe_dist)[0]
            if safe_bins.size > 0:
                center_idx = self.num_lidar_bins // 2
                best = safe_bins[np.argmin(np.abs(safe_bins - center_idx))]
                angle_per_bin = 120.0 / self.num_lidar_bins
                target_angle_deg = (best + 0.5) * angle_per_bin - 60.0
                lat_cmd = float(np.clip(math.sin(math.radians(target_angle_deg)), -1.0, 1.0))
                reduced_forward = max(0.0, fwd_speed * 0.45)
                fwd_cmd = (reduced_forward / max(1e-6, self.max_speed)) * 2.0 - 1.0
                fwd_cmd = float(np.clip(fwd_cmd, -1.0, 1.0))
                self.override_count += 1
                return np.array([fwd_cmd, lat_cmd], dtype=np.float32)
            else:
                back_cmd = -1.0
                dir_sign = 1.0 if (self.override_count % 2 == 0) else -1.0
                lat_cmd = 0.6 * dir_sign
                self.override_count += 1
                return np.array([back_cmd, lat_cmd], dtype=np.float32)
        return None

    # ---------- checkpoint helper ----------
    def _check_and_award_checkpoints(self, prev_y, cur_y):
        newly = []
        bonus = 0.0
        if cur_y <= prev_y:
            return bonus, newly
        for cp in self.checkpoints:
            if prev_y < cp <= cur_y and cp not in self._reached_checkpoints:
                self._reached_checkpoints.add(cp)
                newly.append(cp)
                bonus += self.checkpoint_bonus
        return bonus, newly

    def _compute_window_navigation_reward(self, pos):
        """
        Special reward for navigating through the critical window at y=25m
        """
        try:
            y_pos = float(pos.y_val)
            x_pos = float(pos.x_val)
            z_pos = float(pos.z_val)

            # Only apply window guidance when approaching the critical window
            if y_pos < (self.window_y_position - self.window_approach_zone) or y_pos > (self.window_y_position + 1.0):
                return 0.0

            # Distance from window center
            x_distance_from_center = abs(x_pos - self.window_center_x)
            z_distance_from_center = abs(z_pos - self.window_center_z)

            # Reward for being aligned with window center
            window_alignment_reward = 0.0

            # X-axis alignment (lateral positioning)
            if x_distance_from_center <= self.window_tolerance_x:
                x_alignment = 1.0 - (x_distance_from_center / self.window_tolerance_x)
                window_alignment_reward += x_alignment * 15.0  # Strong reward for lateral alignment
            else:
                # Penalty for being too far from window center laterally
                window_alignment_reward -= min(10.0, x_distance_from_center * 5.0)

            # Z-axis alignment (altitude positioning)
            if z_distance_from_center <= self.window_tolerance_z:
                z_alignment = 1.0 - (z_distance_from_center / self.window_tolerance_z)
                window_alignment_reward += z_alignment * 15.0
            else:
                # Penalty for being too far from window center vertically
                window_alignment_reward -= min(8.0, z_distance_from_center * 4.0)

            # Extra bonus for being very close to perfect alignment
            total_distance = (x_distance_from_center**2 + z_distance_from_center**2)**0.5
            if total_distance < 0.5:  # Very close to window center
                window_alignment_reward += 20.0

            # Additional precision centering bonus for window passage
            if x_distance_from_center <= 0.3:  # Very precisely centered
                precision_bonus = 8.0 * (0.3 - x_distance_from_center) / 0.3
                window_alignment_reward += precision_bonus

            # Progressive reward as approaching window
            approach_progress = max(0.0, self.window_approach_zone - (self.window_y_position - y_pos)) / self.window_approach_zone
            window_alignment_reward *= approach_progress

            if self._debug_verbose and abs(window_alignment_reward) > 1.0:
                print(f"[WINDOW_NAV] y:{y_pos:.1f} x_dist:{x_distance_from_center:.2f} z_dist:{z_distance_from_center:.2f} reward:{window_alignment_reward:.1f}")

            return window_alignment_reward

        except Exception as e:
            if self._debug_verbose:
                print(f"[WINDOW_NAV_ERROR] {e}")
            return 0.0

    # ---------- reward and termination ----------
    def _compute_reward_and_done(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        info = {}

        self.max_y_pos_episode = max(self.max_y_pos_episode, pos.y_val)
        info['max_y_distance'] = self.max_y_pos_episode

        progress = pos.y_val - self.last_y_pos
        prev_last_y = self.last_y_pos
        self.last_y_pos = pos.y_val

        prev_dist = getattr(self, 'prev_dist_to_goal', max(0.0, self.goal_y - prev_last_y))
        cur_dist = max(0.0, self.goal_y - pos.y_val)
        self.prev_dist_to_goal = cur_dist
        shaping = self.reward_shaping_scale * (prev_dist - cur_dist)

        forward_reward = self.reward_forward_scale * progress

        # Enhanced center reward with stability bonus
        x_norm = abs(pos.x_val) / max(1e-6, self.tunnel_half_width_x)
        center_reward = self.center_reward_scale * (1.0 - min(1.0, x_norm))
        center_penalty = self.center_penalty_scale * (x_norm ** 2)

        # Velocity alignment reward (encourage forward velocity, discourage lateral drift)
        vel = state.kinematics_estimated.linear_velocity
        forward_vel_norm = vel.y_val / max(1e-6, self.max_speed)
        lateral_vel_norm = abs(vel.x_val) / max(1e-6, self.max_lat_speed)
        velocity_alignment_reward = 2.0 * max(0, forward_vel_norm) - 1.0 * lateral_vel_norm

        # Stability reward (penalize excessive action changes)
        action_change_penalty = 0.0
        if self.episode_step_count > 1:  # Only after first step
            action_diff = np.linalg.norm(self._prev_action - self._prev_prev_action)
            action_change_penalty = -1.0 * action_diff

        # Smooth progress reward (reward consistent forward movement)
        smooth_progress_reward = 0.0
        if len(self._progress_history) >= 3:
            recent_progress = list(self._progress_history)[-3:]
            if all(p > 0 for p in recent_progress):  # Consistent forward progress
                smooth_progress_reward = 3.0 * np.mean(recent_progress)

        # asymmetric altitude penalty (much stronger when above op_z)
        op_z = self._get_operational_z()
        # If pos.z_val < op_z => agent is higher than operational (more negative), punish strongly
        if pos.z_val < op_z:
            altitude_penalty = -15.0 * (op_z - pos.z_val)
        else:
            altitude_penalty = -1.0 * (pos.z_val - op_z)

        # obstacle penalty
        lidar_obs = self._get_observation()[:self.num_lidar_bins]
        min_lidar = np.min(lidar_obs) * self.max_lidar_range
        obstacle_penalty = 0.0
        if min_lidar < self.safe_dist:
            obstacle_penalty = self.obstacle_penalty_scale * (self.safe_dist - min_lidar)

        # ceiling penalty from forward clearance
        clear_above, clear_below, _, _ = self._compute_vertical_clearance(forward_dist=6.0, forward_angle_deg=30.0)
        ceiling_penalty = 0.0
        if np.isfinite(clear_above) and clear_above < self.ceiling_warn_thresh:
            ceiling_penalty = -10.0 * (self.ceiling_warn_thresh - clear_above)

        # alive + step penalty
        alive = self.alive_bonus
        time_pen = self.time_penalty

        base_reward = (forward_reward + shaping + center_reward + center_penalty + altitude_penalty +
                      obstacle_penalty + alive + time_pen + ceiling_penalty + velocity_alignment_reward +
                      action_change_penalty + smooth_progress_reward)

        # checkpoints
        cp_bonus = 0.0
        newly_reached = []
        bonus, newly_reached = self._check_and_award_checkpoints(prev_last_y, pos.y_val)
        cp_bonus += bonus
        if newly_reached:
            info['new_checkpoints'] = newly_reached

        # Window navigation reward (critical for passing y=25m window)
        window_reward = self._compute_window_navigation_reward(pos)

        # upward movement penalty (strongly discourage climbing)
        z_penalty = 0.0
        if self._last_z is not None:
            z_delta = pos.z_val - self._last_z
            if z_delta > 0.0:
                z_penalty = - self.upward_penalty_coeff * z_delta * 3.0  # Triple the penalty for upward movement
        self._last_z = pos.z_val

        reward = base_reward + cp_bonus + z_penalty + window_reward

        # collision detection & bookkeeping
        col = self.client.simGetCollisionInfo()
        collision = col.has_collided

        # update collision window
        self._collision_window.append(1 if collision else 0)
        self._recent_collision_count = sum(self._collision_window)

        if collision:
            # Update collision tracking for enhanced observations
            self._last_collision_step = self.episode_step_count

            # Improved collision type detection
            collision_type = self._detect_collision_type(col, pos)

            if self._debug_verbose:
                print(f"[COLLISION] Type: {collision_type}, Drone pos: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f}), "
                      f"Collision pos: ({col.position.x_val:.2f}, {col.position.y_val:.2f}, {col.position.z_val:.2f})")
            info['termination_reason'] = 'collision'
            info['collision_type'] = collision_type
            reward += self.collision_penalty
            done = True
            info['episode'] = {'r': self.episode_reward + reward, 'l': self.episode_step_count}
            info['max_y_distance'] = self.max_y_pos_episode
            info['checkpoints_reached'] = sorted(list(self._reached_checkpoints))
            info['override_count'] = self.override_count
            info['vertical_override_count'] = self.vertical_override_count
            info['lateral_override_count'] = self.lateral_override_count
            info['min_clear_above'] = None if not np.isfinite(self.min_clear_above) else float(self.min_clear_above)
            return reward, True, info

        # goal reached
        if pos.y_val >= self.goal_y:
            info['termination_reason'] = 'goal_reached'
            reward += self.goal_reward
            done = True
            info['episode'] = {'r': self.episode_reward + reward, 'l': self.episode_step_count}
            info['max_y_distance'] = self.max_y_pos_episode
            info['checkpoints_reached'] = sorted(list(self._reached_checkpoints))
            info['override_count'] = self.override_count
            info['vertical_override_count'] = self.vertical_override_count
            info['lateral_override_count'] = self.lateral_override_count
            info['min_clear_above'] = None if not np.isfinite(self.min_clear_above) else float(self.min_clear_above)
            return reward, True, info

        # out of bounds (tunnel boundaries - allow reasonable flight envelope)
        if abs(pos.x_val) > self.tunnel_half_width_x or pos.z_val > 1.0 or pos.z_val < -4.0:
            if self._debug_verbose:
                print(f"[OUT_OF_BOUNDS] pos.x={pos.x_val:.3f} (limit=±{self.tunnel_half_width_x:.3f}), pos.z={pos.z_val:.3f} (limits: 1.0 to -4.0)")
            info['termination_reason'] = 'out_of_bounds'
            reward += -80.0
            done = True
            info['episode'] = {'r': self.episode_reward + reward, 'l': self.episode_step_count}
            info['max_y_distance'] = self.max_y_pos_episode
            info['checkpoints_reached'] = sorted(list(self._reached_checkpoints))
            info['override_count'] = self.override_count
            info['vertical_override_count'] = self.vertical_override_count
            info['lateral_override_count'] = self.lateral_override_count
            info['min_clear_above'] = None if not np.isfinite(self.min_clear_above) else float(self.min_clear_above)
            return reward, True, info

        # max steps
        if self.episode_step_count >= self.max_steps:
            info['termination_reason'] = 'max_steps_reached'
            done = True
            info['episode'] = {'r': self.episode_reward + reward, 'l': self.episode_step_count}
            info['max_y_distance'] = self.max_y_pos_episode
            info['checkpoints_reached'] = sorted(list(self._reached_checkpoints))
            info['override_count'] = self.override_count
            info['vertical_override_count'] = self.vertical_override_count
            info['lateral_override_count'] = self.lateral_override_count
            info['min_clear_above'] = None if not np.isfinite(self.min_clear_above) else float(self.min_clear_above)
            return reward, True, info

        info['max_y_distance'] = self.max_y_pos_episode
        info['checkpoints_reached'] = sorted(list(self._reached_checkpoints))
        info['override_count'] = self.override_count
        info['vertical_override_count'] = self.vertical_override_count
        info['min_clear_above'] = None if not np.isfinite(self.min_clear_above) else float(self.min_clear_above)
        return reward, False, info

    # ---------- low-level downward nudge ----------
    def _do_positional_down_nudge(self, current_pos, down_m):
        """
        Move the drone 'down' toward the ground. For AirSim z-axis:
          - more negative z = higher altitude
          - less negative z (closer to zero) = lower altitude (closer to ground)
        So to descend (toward ground) we must INCREASE z (add).
        """
        try:
            # target_z: move toward ground by 'down_m' meters (less negative)
            target_z = float(current_pos.z_val) + abs(down_m)

            # clamp so we don't move too close to the ground:
            # allow at most desired_z + 0.5 (i.e., not more than 0.5m below desired altitude toward ground)
            max_allowed_z = self.desired_z + 0.5
            if target_z > max_allowed_z:
                target_z = max_allowed_z

            # send the command (blocking join to apply immediately)
            self.client.moveToZAsync(target_z, 1.2, vehicle_name=self.vehicle_name).join()
            time.sleep(0.02)
            return True
        except Exception:
            return False

    # ---------- step / reset ----------
    def step(self, action):
        truncated = False
        try:
            self.episode_step_count += 1

            # read state
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            clear_above, clear_below, _, _ = self._compute_vertical_clearance(forward_dist=6.0, forward_angle_deg=30.0)

            # 1) If drone higher than operational_z (more negative), immediate stronger descent
            op_z = self._get_operational_z()
            up_margin = 0.10  # Reduced margin to be more aggressive
            if pos.z_val < (op_z - up_margin) and (self.episode_step_count - self._last_vertical_override_step) > 1:
                if self._debug_verbose:
                    print(f"[FORCE_DESCEND] step={self.episode_step_count} pos.z={pos.z_val:.3f} op_z={op_z:.3f}")
                did = self._do_positional_down_nudge(pos, max(self.ceiling_override_down, 0.9))
                if did:
                    self.vertical_override_count += 1
                    self._last_vertical_override_step = self.episode_step_count
                    forced = float(op_z) + abs(self.pos_override_down)
                    max_allowed = self.desired_z + 0.5
                    if forced > max_allowed:
                        forced = max_allowed
                    self._forced_operational_z = forced
                    self._forced_operational_z_until_step = self.episode_step_count + self._forced_operational_z_duration_steps
                    obs = self._get_observation()
                    reward, done, info = self._compute_reward_and_done()
                    self.episode_reward += reward
                    terminated = bool(done)
                    if terminated and 'episode' not in info:
                        info['episode'] = {'r': self.episode_reward, 'l': self.episode_step_count}
                    return obs, reward, terminated, truncated, info

            # 2) If forward lidar sees roof close -> nudge earlier
            if np.isfinite(clear_above) and clear_above < self.ceiling_override_thresh and (self.episode_step_count - self._last_vertical_override_step) > self.vertical_override_cooldown:
                if self._debug_verbose:
                    print(f"[LIDAR_NUDGE] step={self.episode_step_count} clear_above={clear_above:.3f}")
                did = self._do_positional_down_nudge(pos, self.ceiling_override_down)
                if did:
                    self.vertical_override_count += 1
                    self._last_vertical_override_step = self.episode_step_count
                    forced = float(self._get_operational_z()) + abs(self.pos_override_down)
                    max_allowed = self.desired_z + 0.5
                    if forced > max_allowed:
                        forced = max_allowed
                    self._forced_operational_z = forced
                    self._forced_operational_z_until_step = self.episode_step_count + self._forced_operational_z_duration_steps
                    obs = self._get_observation()
                    reward, done, info = self._compute_reward_and_done()
                    self.episode_reward += reward
                    terminated = bool(done)
                    if terminated and 'episode' not in info:
                        info['episode'] = {'r': self.episode_reward, 'l': self.episode_step_count}
                    return obs, reward, terminated, truncated, info

            # 3) Collision recovery: if many collisions recently, force stronger descent + short freeze
            col = self.client.simGetCollisionInfo()
            self._collision_window.append(1 if col.has_collided else 0)
            self._recent_collision_count = sum(self._collision_window)
            if self._recent_collision_count >= 4 and (self.episode_step_count - self._last_vertical_override_step) > 1:
                if self._debug_verbose:
                    print(f"[COLLISION_RECOVER] step={self.episode_step_count} recent_collisions={self._recent_collision_count}")
                self._do_positional_down_nudge(pos, max(1.0, self.ceiling_override_down))
                self._last_vertical_override_step = self.episode_step_count
                self._forced_operational_z = float(op_z) + max(0.8, self.pos_override_down)
                self._forced_operational_z_until_step = self.episode_step_count + max(4, self._forced_operational_z_duration_steps)
                obs = self._get_observation()
                reward, done, info = self._compute_reward_and_done()
                self.episode_reward += reward
                terminated = bool(done)
                if terminated and 'episode' not in info:
                    info['episode'] = {'r': self.episode_reward, 'l': self.episode_step_count}
                return obs, reward, terminated, truncated, info

            # 4) Otherwise apply horizontal action / override with priority system
            a = np.array(action, dtype=np.float32)

            # PRIORITY 0: Emergency altitude fix (highest priority)
            emergency_handled = False
            try:
                emergency_handled = self._emergency_altitude_fix(pos)
            except Exception:
                emergency_handled = False

            # Initialize override variables
            window_exploration_override = None

            if emergency_handled:
                # Skip other overrides, let emergency altitude fix handle it
                used_action = a  # Keep original action but altitude is being fixed
            else:
                # PRIORITY 1: Window exploration (force exploration near window)
                try:
                    window_exploration_override = self._window_exploration_override(a, pos)
                except Exception:
                    window_exploration_override = None

                if window_exploration_override is not None:
                    used_action = window_exploration_override
                else:
                    # PRIORITY 2: Brute force navigation
                    brute_force_override = None
                    try:
                        brute_force_override = self._brute_force_navigate(a, pos)
                    except Exception:
                        brute_force_override = None

                    if brute_force_override is not None:
                        used_action = brute_force_override
                    else:
                        # PRIORITY 3: Lateral override (virtual guardrails)
                        lateral_override = None
                        try:
                            lateral_override = self._lateral_override(a, pos)
                        except Exception:
                            lateral_override = None

                        # Apply lateral override if needed
                        if lateral_override is not None:
                            a = lateral_override

                        # PRIORITY 4: Horizontal obstacle override
                        horizontal_override = None
                        try:
                            horizontal_override = self._horizontal_override(a)
                        except Exception:
                            horizontal_override = None

                        used_action = horizontal_override if (horizontal_override is not None) else a
            a_clipped = np.clip(used_action, self.action_space.low, self.action_space.high)

            # Store previous action for enhanced observations
            self._prev_prev_action = self._prev_action.copy()
            self._prev_action = a_clipped.copy()

            forward_norm = (float(a_clipped[0]) + 1.0) / 2.0
            forward_vel = float(forward_norm * self.max_speed)
            lateral_vel = float(a_clipped[1] * self.max_lat_speed)

            # aggressive forward clamp if ceiling close
            if np.isfinite(clear_above) and clear_above < self.ceiling_warn_thresh:
                forward_vel = min(forward_vel, 0.35 * self.max_speed)

            # startup grace
            if self.episode_step_count <= self.startup_grace_steps:
                forward_vel, lateral_vel = 0.0, 0.0

            # small soft snap back to operational z when drifted
            try:
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                pos = state.kinematics_estimated.position
                if abs(pos.z_val - self._get_operational_z()) > self.operational_z_tolerance:
                    try:
                        # Move altitude back toward operational z (less negative means lower)
                        self.client.moveToZAsync(self._get_operational_z(), 0.8, vehicle_name=self.vehicle_name).join()
                        time.sleep(0.01)
                        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                        pos = state.kinematics_estimated.position
                    except Exception:
                        pass
            except Exception:
                pass

            # send horizontal velocity (vz=0) - vertical corrections handled by nudges above
            try:
                self.client.moveByVelocityAsync(vx=lateral_vel, vy=forward_vel, vz=0.0, duration=self.dt,
                                                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                                                vehicle_name=self.vehicle_name).join()
            except Exception:
                pass

            obs = self._get_observation()

            reward, done, info = self._compute_reward_and_done()
            self.episode_reward += reward

            terminated = bool(done)
            if terminated and 'episode' not in info:
                info['episode'] = {'r': self.episode_reward, 'l': self.episode_step_count}
                info['max_y_distance'] = self.max_y_pos_episode

            # Update curriculum learning if episode ended
            if terminated:
                self._update_curriculum(info.get('termination_reason', 'unknown'), self.max_y_pos_episode)

            return obs, reward, terminated, truncated, info

        except Exception as e:
            if self._debug_verbose:
                print(f"[STEP_ERROR] {e}")
            # Return a safe observation if something went wrong
            try:
                obs = self._get_observation()
            except Exception:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -10.0, False, truncated, {}

    def reset(self, seed=None, options=None):
        """
        Reset environment state and reposition drone to start state.
        Compatible with new Gym API that expects seed and options parameters.
        """
        # Handle seed parameter for reproducibility
        if seed is not None:
            self.seed(seed)
        # Reset episode bookkeeping
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.last_y_pos = 0.0
        self.max_y_pos_episode = 0.0
        self._reached_checkpoints = set()
        self._prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self._prev_prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self._collision_window = collections.deque([0] * 12, maxlen=12)
        self._last_collision_step = -1000
        self._progress_history = collections.deque([0.0] * 5, maxlen=5)
        self._forced_operational_z = None
        self._forced_operational_z_until_step = -1
        self._last_vertical_override_step = -999
        self._recent_collision_count = 0
        self._last_z = None
        
        # Initialize override counters
        self.override_count = 0
        self.vertical_override_count = 0
        self.lateral_override_count = 0

        # Reset the simulator (try best-effort)
        try:
            # try API reset first
            self.client.reset()
            time.sleep(0.2)
            self.client.enableApiControl(True, self.vehicle_name)
            self.client.armDisarm(True, self.vehicle_name)
            # takeoff to a starting altitude near desired_z
            try:
                self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
                # move to starting pose: x=0,y=0, z=desired_z (less negative = lower)
                start_z = self.desired_z
                # If desired_z is more negative than -3.5, clamp to -3.5 for safety
                if start_z < -3.5:
                    start_z = -3.5
                self.client.moveToZAsync(start_z, velocity=1.5, timeout_sec=3.0, vehicle_name=self.vehicle_name).join()
            except Exception:
                pass
        except Exception:
            # If reset isn't available, try a gentle reposition
            try:
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                pos = state.kinematics_estimated.position
                start_z = self.desired_z
                self.client.moveToZAsync(start_z, velocity=1.5, timeout_sec=3.0, vehicle_name=self.vehicle_name).join()
            except Exception:
                pass

        # Return initial observation and info dict (new Gym API)
        obs = self._get_observation()
        info = {
            "max_y_distance": self.max_y_pos_episode,
            "checkpoints_reached": sorted(list(self._reached_checkpoints)),
            "override_count": self.override_count,
            "vertical_override_count": self.vertical_override_count,
            "lateral_override_count": self.lateral_override_count
        }
        return obs, info

    def close(self):
        try:
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
        except Exception:
            pass
