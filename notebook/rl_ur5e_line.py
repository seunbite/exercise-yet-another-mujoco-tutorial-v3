import sys, os, mujoco, time, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import imageio
from time import strftime
import glob
from tqdm import tqdm
import time

# Add paths to import custom modules
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/gym/')
sys.path.append('../package/rl/')

from mujoco_parser import MuJoCoParserClass
from transformation import r2rpy, rpy2r, pr2t, t2p, t2r
from utility import get_colors, d2r, trim_scale, np2torch, torch2np
from sac import ReplayBufferClass, ActorClass, CriticClass, get_target
import cv2
import matplotlib.pyplot as plt
import numpy as np


class TimeTracker:
    def __init__(self):
        self.times = {}
        self.temp_start = None
        
    def start_timer(self, name):
        self.temp_start = time.time()
        
    def end_timer(self, name):
        if self.temp_start is not None:
            duration = time.time() - self.temp_start
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(duration)
            self.temp_start = None
            return duration
        return 0
    
    def get_stats(self):
        stats = {}
        for name, times_list in self.times.items():
            if len(times_list) > 0:
                stats[name] = {
                    'mean': np.mean(times_list),
                    'total': np.sum(times_list),
                    'count': len(times_list),
                    'max': np.max(times_list),
                    'min': np.min(times_list)
                }
        return stats
    
    def print_stats(self):
        stats = self.get_stats()
        total_time = sum(data['total'] for data in stats.values())
        
        print("\n" + "="*90)
        print("TIMING ANALYSIS")
        print("="*90)
        print(f"{'Component':<25} | {'Total (s)':<10} | {'%':<6} | {'Mean (ms)':<10} | {'Count':<8}")
        print("-"*90)
        
        for name, data in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            percentage = (data['total'] / total_time) * 100 if total_time > 0 else 0
            print(f"{name:<25} | {data['total']:>8.2f}s | {percentage:>5.1f}% | {data['mean']*1000:>8.1f}ms | {data['count']:>6}")
        
        print("="*90)
        print(f"{'TOTAL TIME':<25} | {total_time:>8.2f}s | {'100.0%':>6} |")
        print("="*90)
        
        # Special analysis for potential rendering issues
        if 'render' in stats and 'env_step' in stats:
            render_ratio = stats['render']['total'] / stats['env_step']['total'] if stats['env_step']['total'] > 0 else 0
            print(f"\nRENDER ANALYSIS:")
            print(f"Render time as % of env_step time: {render_ratio*100:.1f}%")
            
        if 'env_step_physics' in stats and 'env_step' in stats:
            physics_ratio = stats['env_step_physics']['total'] / stats['env_step']['total'] if stats['env_step']['total'] > 0 else 0
            print(f"Physics time as % of env_step time: {physics_ratio*100:.1f}%")
            
        print("="*90)
    
    def reset(self):
        self.times = {}
        
    def reset_counters(self):
        """Reset counters but keep recent timing data for accurate averages"""
        for name in self.times:
            if len(self.times[name]) > 100:  # Keep last 100 measurements
                self.times[name] = self.times[name][-100:]


class UR5eHeadTouchEnv:
    def __init__(
            self,
            env,
            HZ=50,
            history_total_sec=1.0,
            history_intv_sec=0.1,
            object_table_position=[1.0, 0, 0], 
            base_table_position=[0, 0, 0],
            head_position=[1, 0.0, 0.53], 
            waiting_time=0.0, 
            normalize_obs=True, 
            normalize_reward=False, 
            reward_mode='1',
            start_reached=False,
            dynamic_target=10.0,
            ):
        self.env = env
        self.HZ = HZ
        self.dt = 1/self.HZ
        self.history_total_sec = history_total_sec
        self.n_history = int(self.HZ * self.history_total_sec)
        self.history_intv_sec = history_intv_sec
        self.history_intv_tick = int(self.HZ * self.history_intv_sec)
        self.history_ticks = np.arange(0, self.n_history, self.history_intv_tick)
        
        self.mujoco_nstep = self.env.HZ // self.HZ
        self.tick = 0
        
        self.name = env.name
        self.head_position = np.array(head_position)
        self.waiting_time = waiting_time
        self.object_table_position = np.array(object_table_position)
        self.base_table_position = np.array(base_table_position)
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.reward_mode = reward_mode
        self.dynamic_target_threshold = dynamic_target  # Success rate threshold for dynamic targets
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        self.heatmap_dots = []
        
        # Add tracking for directional movement reward
        self.prev_tip_pos = None
        self.prev_distance_to_end = None
        
        # Add tracking for latest reward components for rendering
        self.latest_reward = 0.0
        self.latest_reward_components = {}
        
        # Add tracking for joint movement (for reward_2)
        self.prev_qpos = None
        
        # Add cumulative tracking for episodes
        self.total_episodes = 0
        self.total_start_reached = 0
        self.total_end_reached = 0
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        available_joints = self.env.rev_joint_names
        valid_joint_names = []
        for joint_name in self.joint_names:
            if joint_name in available_joints:
                valid_joint_names.append(joint_name)
            else:
                print(f"Warning: Joint '{joint_name}' not found in model. Available joints: {available_joints}")
        
        if len(valid_joint_names) == 0:
            # Fallback: use first 6 revolute joints
            valid_joint_names = available_joints[:6] if len(available_joints) >= 6 else available_joints
            print(f"Using fallback joint names: {valid_joint_names}")
        
        self.joint_names = valid_joint_names
        
        self.initialte_start_reached = start_reached
        self.start_reached = False
        self.end_reached = False
        self.touch_threshold = 0.01
        
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0
        
        self.state_dim = len(self._get_observation_raw())
        self.state_history = np.zeros((self.n_history, self.state_dim))
        self.tick_history = np.zeros((self.n_history, 1))
        self.o_dim = len(self.get_observation())
        self.a_dim = len(self.joint_names)

        # -----------------------------------------------------------------------------------
        # Heat-map buffers
        #   Each element is (u, v, reward) where
        #     u = 0 at start, 1 at end, >1 beyond; v = perpendicular distance (m)
        # -----------------------------------------------------------------------------------
        self._heat_before_start = []  # data collected while not start_reached
        self._heat_after_start  = []  # data collected once start_reached is True

    def compute_gradient_norm(self, model):
        total_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def _dynamic_target(self, p_target, x=[0.0, 0.0], y=[0.0, 0.0], z=[0.0, 0.0]):
        offset = np.array([
            np.random.uniform(x[0], x[1]), 
            np.random.uniform(y[0], y[1]), 
            np.random.uniform(z[0], z[1])
        ])
        return p_target + offset

    def get_applicator_tip_pos(self, keyword='applicator_tip'):
        return self.env.get_p_body(keyword)

    def get_object_position(self, object_name):
        return self.env.get_p_body(object_name)
    
    def get_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def _get_distance_from_line(self):
        """Calculate perpendicular distance from applicator tip to the line"""
        if not hasattr(self, 'target_line'):
            return 0.0
        
        tip_pos = self.get_applicator_tip_pos()
        start_pos = self.target_line[0]
        end_pos = self.target_line[1]
        
        line_vec = end_pos - start_pos
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-6:
            return 0.0
        
        pos_vec = tip_pos - start_pos
        
        projection_length = np.dot(pos_vec, line_vec) / line_length
        projection_point = start_pos + (projection_length / line_length) * line_vec
        deviation = np.linalg.norm(tip_pos - projection_point)
        
        return deviation

    def _get_observation_raw(self):
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            qpos = self.env.data.qpos[joint_idxs]
            qvel = self.env.data.qvel[joint_idxs]
        except:
            qpos = self.env.data.qpos[:self.a_dim]
            qvel = self.env.data.qvel[:self.a_dim]
        
        tip_pos = self.get_applicator_tip_pos()
        base_table_pos = self.get_object_position('base_table')
        
        if hasattr(self, 'target_line'):
            start_point = self.target_line[0]
            end_point = self.target_line[1]
        else:
            start_point = np.zeros(3)
            end_point = np.zeros(3)
            print("No target line found")
        
        state = np.concatenate([
            qpos,
            qvel,
            tip_pos,
            np.array([int(self.start_reached)]),
            np.array([int(self.end_reached)]),
            base_table_pos,
            start_point,
            end_point,
        ])  
        
        return state

    def _update_obs_stats(self, obs):
        self.obs_count += 1
        if self.obs_mean is None:
            self.obs_mean = obs.copy()
            self.obs_std = np.ones_like(obs)
        else:
            alpha = 1.0 / self.obs_count
            self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs
            self.obs_std = (1 - alpha) * self.obs_std + alpha * np.abs(obs - self.obs_mean)

    def _normalize_obs(self, obs):
        if self.obs_mean is not None:
            std = np.maximum(self.obs_std, 1e-6)
            return (obs - self.obs_mean) / std
        return obs

    def _normalize_reward(self, reward):
        if self.reward_count > 10:
            std = max(self.reward_std, 1e-6)
            normalized_reward = (reward - self.reward_mean) / (3 * std)
            return np.clip(normalized_reward, -1.0, 1.0)
        return reward

    def _normalize_action(self, action):
        return np.tanh(action)

    def get_observation(self):
        """
        Get normalized observation - this might be expensive
        """
        # Get raw observation
        obs = self._get_observation_raw()
        
        # Update statistics
        if self.normalize_obs:
            self._update_obs_stats(obs)
            obs = self._normalize_obs(obs)
        
        # Add history if needed
        state_history_sparse = self.state_history[self.history_ticks, :]
        obs_with_history = np.concatenate([obs, state_history_sparse.flatten()])
        
        return obs_with_history

    def sample_action(self):
        """
        Sample random action
        """
        # Get control ranges
        try:
            ctrl_idxs = self.env.get_idxs_step(joint_names=self.joint_names)
            a_min = self.env.ctrl_ranges[ctrl_idxs, 0]
            a_max = self.env.ctrl_ranges[ctrl_idxs, 1]
        except:
            # Fallback to default ranges
            a_min = np.full(self.a_dim, -2.0)
            a_max = np.full(self.a_dim, 2.0)
        
        action = a_min + (a_max - a_min) * np.random.rand(len(a_min))
        return action

    def reward_1(self, applicator_tip_pos, current_distance_start, current_distance_end):
        r_START_TOUCH = 0
        r_END_TOUCH = 10
        r_TIME = -0.00001
        w_START_TOUCH = -0.2
        w_END_TOUCH = 0.01
        w_DISTANCE_IMPROVEMENT = 1.0

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        r_line_dev = 0.0
        r_directional_movement = 0.0
        stop_rewarding = False
        
        if not self.start_reached:
            delta_d = current_distance_start
            r_start_distance = delta_d * w_START_TOUCH
            
            if current_distance_start < self.touch_threshold:
                print("Start point reached!")
                self.start_reached = True
                reward += r_START_TOUCH
                stop_rewarding = True
            
        if self.start_reached and not self.end_reached and not stop_rewarding:
            # Initialize tracking variables if first time after start_reached
            if self.prev_tip_pos is None:
                self.prev_tip_pos = applicator_tip_pos.copy()
                self.prev_distance_to_end = current_distance_end
            
            # Calculate movement vector and direction
            movement_vector = applicator_tip_pos - self.prev_tip_pos
            movement_magnitude = np.linalg.norm(movement_vector)
            
            # Line direction (from start to end)
            line_direction = self.target_line[1] - self.target_line[0]
            
            # r_directional_movement: reward for moving in line direction
            r_directional_movement = 0.0
            if movement_magnitude > 1e-6 and self.target_line_length > 1e-6:
                movement_dir_normalized = movement_vector / movement_magnitude
                line_dir_normalized = line_direction / self.target_line_length
                direction_alignment = np.dot(movement_dir_normalized, line_dir_normalized)
                # Negative reward that becomes less negative when aligned with line direction
                r_directional_movement = (direction_alignment - 1.0) * w_END_TOUCH  # Range: [-0.02, 0]
            else:
                r_directional_movement = -0.005  # Penalty for not moving
            
            # r_in_progress: reward for getting closer to end point
            distance_improvement = self.prev_distance_to_end - current_distance_end
            r_in_progress = distance_improvement * w_DISTANCE_IMPROVEMENT  # Positive when getting closer
            
            # Update tracking variables
            self.prev_tip_pos = applicator_tip_pos.copy()
            self.prev_distance_to_end = current_distance_end
            
            r_line_dev = 0.0
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        reward += r_start_distance + r_in_progress + r_line_dev + r_TIME + r_directional_movement

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
            'r_directional_movement': r_directional_movement,
        }
        
        return reward, reward_components

    def reward_1_5(self, applicator_tip_pos, current_distance_start, current_distance_end):
        r_START_TOUCH = 0
        r_END_TOUCH = 10
        r_TIME = -0.00001
        w_START_TOUCH = -0.2
        w_END_TOUCH = 0.01
        w_DISTANCE_IMPROVEMENT = 1
        w_ABSOLUTE_DISTANCE = 1.0  # scaling for absolute distance reward (kept within ±0.5)

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        r_absolute_distance = 0.0
        r_directional_movement = 0.0
        stop_rewarding = False
        
        if not self.start_reached:
            delta_d = current_distance_start
            r_start_distance = delta_d * w_START_TOUCH
            
            if current_distance_start < self.touch_threshold:
                print("Start point reached!")
                self.start_reached = True
                reward += r_START_TOUCH
                stop_rewarding = True
            
        if self.start_reached and not self.end_reached and not stop_rewarding:
            if self.prev_tip_pos is None:
                self.prev_tip_pos = applicator_tip_pos.copy()
                self.prev_distance_to_end = current_distance_end
            
            movement_vector = applicator_tip_pos - self.prev_tip_pos
            movement_magnitude = np.linalg.norm(movement_vector)
            line_direction = self.target_line[1] - self.target_line[0]
            
            r_directional_movement = 0.0
            if movement_magnitude > 1e-6 and self.target_line_length > 1e-6:
                movement_dir_normalized = movement_vector / movement_magnitude
                line_dir_normalized = line_direction / self.target_line_length
                direction_alignment = np.dot(movement_dir_normalized, line_dir_normalized)
                r_directional_movement = (direction_alignment - 1.0) * w_END_TOUCH  # Range: [-0.02, 0]
            else:
                r_directional_movement = -0.005
            
            distance_improvement = self.prev_distance_to_end - current_distance_end
            r_in_progress = distance_improvement * w_DISTANCE_IMPROVEMENT
            
            # ---------------- Absolute distance term ----------------
            # Map current_distance_end ∈ [0, line_length] → reward ∈ [+0.5, -0.5]
            progress = 1 - (current_distance_end / max(self.target_line_length, 1e-6))
            # progress: 0 at start, 1 at end
            r_absolute_distance = (progress - 0.5) * w_ABSOLUTE_DISTANCE
            r_absolute_distance = float(np.clip(r_absolute_distance, -0.5, 0.5))
            
            self.prev_tip_pos = applicator_tip_pos.copy()
            self.prev_distance_to_end = current_distance_end
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        reward += r_start_distance + r_in_progress + r_absolute_distance + r_TIME + r_directional_movement

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
            'r_directional_movement': r_directional_movement,
            'r_absolute_distance': r_absolute_distance,
        }
        
        return reward, reward_components
    
    def reward_1_6(self, applicator_tip_pos, current_distance_start, current_distance_end):
        r_START_TOUCH = -0.2
        w_START_TOUCH = -0.1  # constant penalty before start reached
        r_END_TOUCH = 10
        w_END_TOUCH = 0.1     # scale for directional movement (will be clipped to [-0.1,0])
        r_TIME = -0.00001
        w_DISTANCE_IMPROVEMENT = 0.1
        w_ABSOLUTE_DISTANCE = 0.1  # now r_absolute_distance ∈ [-0.1,0]
        r_ON_THE_LINE = 0.2

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        r_absolute_distance = 0.0
        r_directional_movement = 0.0
        stop_rewarding = False
        
        if not self.start_reached:
            r_start_distance = current_distance_start * w_START_TOUCH + r_START_TOUCH
            
            if current_distance_start < self.touch_threshold:
                print("Start point reached!")
                self.start_reached = True
                stop_rewarding = True
            
        if self.start_reached and not self.end_reached and not stop_rewarding:
            if self.prev_tip_pos is None:
                self.prev_tip_pos = applicator_tip_pos.copy()
                self.prev_distance_to_end = current_distance_end
            
            movement_vector = applicator_tip_pos - self.prev_tip_pos
            movement_magnitude = np.linalg.norm(movement_vector)
            line_direction = self.target_line[1] - self.target_line[0]
            
            r_directional_movement = 0.0
            if movement_magnitude > 1e-6 and self.target_line_length > 1e-6:
                movement_dir_normalized = movement_vector / movement_magnitude
                line_dir_normalized = line_direction / self.target_line_length
                direction_alignment = np.dot(movement_dir_normalized, line_dir_normalized)
                r_directional_movement = np.clip((direction_alignment - 1.0) * w_END_TOUCH, -0.1, 0.0)
            else:
                r_directional_movement = -0.1  # maximum penalty when not moving
            
            distance_improvement = self.prev_distance_to_end - current_distance_end
            r_in_progress = distance_improvement * w_DISTANCE_IMPROVEMENT
            
            progress = 1 - (current_distance_end / max(self.target_line_length, 1e-6))  # 0 at start,1 at end
            r_absolute_distance = np.clip((progress - 1.0) * (-w_ABSOLUTE_DISTANCE), -0.1, 0.0)  # [-0.1,0]

            line_deviation = self._get_distance_from_line()
            self.over_perpendicular_plane = np.dot(line_direction, applicator_tip_pos - self.target_line[1]) > 0
            
            if line_deviation <= self.touch_threshold and r_in_progress > 0.0 and not self.over_perpendicular_plane:
                reward += r_ON_THE_LINE
            
            self.prev_tip_pos = applicator_tip_pos.copy()
            self.prev_distance_to_end = current_distance_end
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
            
        reward += r_start_distance + r_in_progress + r_absolute_distance + r_TIME + r_directional_movement

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
            'r_directional_movement': r_directional_movement,
            'r_absolute_distance': r_absolute_distance,
        }
        
        return reward, reward_components
    
    def reward_2(self, applicator_tip_pos, current_distance_start, current_distance_end):
        w_START_TOUCH = -2  # penalty proportional to distance before start reached

        r_END_TOUCH = 10    # bonus when end point touched
        r_TIME = -0.001     # tiny time penalty
        w_DISTANCE_PROGRESS = 0.1  # positive reward as tip approaches end

        reward = 0.0
        r_start_distance = 0.0
        r_distance_progress = 0.0
        stop_rewarding = False

        if not self.start_reached:
            delta_d = current_distance_start
            r_start_distance = delta_d * w_START_TOUCH
            
            if current_distance_start < self.touch_threshold:
                print("Start point reached!")
                self.start_reached = True
                stop_rewarding = True
            
        if self.start_reached and not self.end_reached and not stop_rewarding:
            r_distance_progress = (1 - (current_distance_end / max(self.target_line_length, 1e-6))) * w_DISTANCE_PROGRESS
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        reward += r_start_distance + r_distance_progress + r_TIME

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_progress': r_distance_progress,
        }
        
        return reward, reward_components

    def compute_reward(self, action, max_time=np.inf):
        applicator_tip_pos = self.get_applicator_tip_pos()
        current_distance_start = self.get_distance(applicator_tip_pos, self.target_line[0])
        current_distance_end = self.get_distance(applicator_tip_pos, self.target_line[1])
        
        SUCCESS = self.end_reached
        TIMEOUT = self.get_sim_time() >= max_time
        
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            qpos = self.env.data.qpos[joint_idxs]
            joint_ranges = self.env.joint_ranges[:len(qpos)]
            JOINT_LIMIT_VIOLATION = np.any(qpos < joint_ranges[:, 0]) or np.any(qpos > joint_ranges[:, 1])
        except:
            JOINT_LIMIT_VIOLATION = False
        
        done = SUCCESS or TIMEOUT or JOINT_LIMIT_VIOLATION

        if str(self.reward_mode) == '1':
            reward, reward_components = self.reward_1(applicator_tip_pos, current_distance_start, current_distance_end)
        elif str(self.reward_mode) == '1.5':
            reward, reward_components = self.reward_1_5(applicator_tip_pos, current_distance_start, current_distance_end)
        elif str(self.reward_mode) == '1.6':
            reward, reward_components = self.reward_1_6(applicator_tip_pos, current_distance_start, current_distance_end)
        elif str(self.reward_mode) == '2':
            reward, reward_components = self.reward_2(applicator_tip_pos, current_distance_start, current_distance_end)
        elif str(self.reward_mode) == '3':
            reward, reward_components = self.reward_3(applicator_tip_pos, current_distance_start, current_distance_end)
        elif str(self.reward_mode) == '4':
            reward, reward_components = self.reward_4(applicator_tip_pos, current_distance_start, current_distance_end)
        elif str(self.reward_mode) == '5':
            reward, reward_components = self.reward_5(applicator_tip_pos, current_distance_start, current_distance_end)
        else:
            raise ValueError(f"Invalid reward mode: {self.reward_mode}")
        
        # Store latest reward information for rendering
        self.latest_reward = reward
        self.latest_reward_components = reward_components.copy()
        
        if self.normalize_reward:
            reward = self._normalize_reward(reward)
            self.reward_count += 1
            self.reward_mean = (self.reward_mean * (self.reward_count - 1) + reward) / self.reward_count
            self.reward_std = np.sqrt((self.reward_std ** 2 * (self.reward_count - 1) + (reward - self.reward_mean) ** 2) / self.reward_count)

        state_info = {
            'distance_start': current_distance_start,
            'distance_end': current_distance_end,
            'start_reached': self.start_reached,
            'end_reached': self.end_reached,
            'success': SUCCESS,
            'timeout': TIMEOUT,
            'joint_limit_violation': JOINT_LIMIT_VIOLATION,
            'line_deviation': self._get_distance_from_line(),
            'done': done
        }
        
        # ----------------- Heat-map logging -----------------
        u, v = self._get_relative_coords(applicator_tip_pos)
        if not self.start_reached:
            self._heat_before_start.append((u, v, reward))
        else:
            self._heat_after_start.append((u, v, reward))

        return reward, reward_components, state_info

    def step(self, action, max_time=np.inf):
        self.tick += 1
        
        # Action normalization
        scaled_action = self._normalize_action(action)
        
        # Physics simulation step - this is where most time should be spent
        try:
            ctrl_idxs = self.env.get_idxs_step(joint_names=self.joint_names)
            self.env.step(ctrl=scaled_action, ctrl_idxs=ctrl_idxs, nstep=self.mujoco_nstep)
        except:
            self.env.step(ctrl=scaled_action, nstep=self.mujoco_nstep)
        
        # Reward computation
        reward, reward_components, state_info = self.compute_reward(action, max_time)
        
        # Observation processing - might be expensive
        self.accumulate_state_history()
        obs_next = self.get_observation()
        
        info = {**state_info, **reward_components}
        
        return obs_next, reward, state_info['done'], info

    def _make_line(self, start_point, direction, length):
        """Create a line segment from start_point in the given direction"""
        end_point = start_point + direction * length
        return np.array([start_point, end_point])

    def reset(self):
        # Update episode statistics before resetting
        if hasattr(self, 'total_episodes'):  # Check if this is not the first reset
            self.total_episodes += 1
            if self.start_reached:
                self.total_start_reached += 1
            if self.end_reached:
                self.total_end_reached += 1
        else:
            # Initialize counters on first reset
            self.total_episodes = 0
            self.total_start_reached = 0
            self.total_end_reached = 0
        
        self.tick = 0
        
        self.env.reset(step=True)
        
        self.env.set_p_body(body_name='ur_base', p=[0, 0, 0.5])
        self.env.set_p_body(body_name='base_table', p=self.base_table_position)
        self.env.set_p_body(body_name='object_table', p=self.object_table_position)

        self.end_reached = False
        
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            init_qpos = np.array([-30, -60, 90, -30, -30, 0]) * np.pi / 180
            if len(init_qpos) != len(self.joint_names):
                init_qpos = np.zeros(len(self.joint_names))
            self.env.forward(q=init_qpos, joint_idxs=joint_idxs)
        except Exception as e:
            print(f"Warning: Could not set initial configuration: {e}")
            pass
        if self.waiting_time > 0:
            for _ in range(int(self.waiting_time * self.HZ)):
                self.env.step(ctrl=np.zeros(len(self.joint_names)), nstep=1)
        
        self.start_reached = self.initialte_start_reached
        
        # Set target position based on success rate threshold
        # Calculate current end success rate percentage
        current_end_success_rate = (self.total_end_reached / max(self.total_episodes, 1)) * 100
        use_dynamic_target = current_end_success_rate >= self.dynamic_target_threshold
        
        # Debug output for target type switching
        if self.total_episodes > 0 and self.total_episodes % 50 == 0:  # Print every 50 episodes
            print(f"Episode {self.total_episodes}: End success rate: {current_end_success_rate:.1f}%, "
                  f"Target type: {'Dynamic' if use_dynamic_target else 'Static'} "
                  f"(threshold: {self.dynamic_target_threshold}%)")
        
        if use_dynamic_target:
            # Dynamic target positioning - harder, more varied
            self.target_pos_start = self._dynamic_target(self.head_position, x=[0.03, 0.03], y=[0.0, 0.03], z=[0.15, 0.18])
            self.target_line_length = random.uniform(0.1, 0.13)
            z_offset = np.random.uniform(0.05, 0.10) * np.random.choice([-1, 1])
            line_direction = np.array([np.random.choice([-1, 1]), np.random.uniform(-0.5, 0.5), z_offset / self.target_line_length])
            line_direction = line_direction / np.linalg.norm(line_direction)
        else:
            # Static target positioning - easier, consistent
            self.target_pos_start = self.head_position + np.array([0.03, 0.015, 0.165])  # Fixed offset
            self.target_line_length = 0.125  # Fixed length
            line_direction = np.array([1, 0, 0])  # Fixed direction (along x-axis)
        
        self.target_line = self._make_line(self.target_pos_start, direction=line_direction, length=self.target_line_length)
        self.prev_tip_pos = self.target_pos_start.copy()
        self.prev_distance_to_end = self.get_distance(self.target_pos_start, self.target_line[1])
        
        self.prev_qpos = None
        self.still_counter = 0  # Reset still counter for each episode
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0
        
        self.state_history = np.zeros((self.n_history, self.state_dim))
        self.tick_history = np.zeros((self.n_history, 1))
    
        return self.get_observation()

    def accumulate_state_history(self):
        state = self._get_observation_raw()
        
        self.state_history[1:, :] = self.state_history[:-1, :]
        self.state_history[0, :] = state
        
        self.tick_history[1:, :] = self.tick_history[:-1, :]
        self.tick_history[0, :] = self.tick
    
    def render(self, PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True, PLOT_REWARD=True, PLOT_STATS=True):
        if not hasattr(self.env, 'viewer') or not self.env.use_mujoco_viewer:
            return
        
        if PLOT_TARGET and hasattr(self, 'target_line'):
            start_pos = self.target_line[0]
            end_pos = self.target_line[1]
            
            self.env.plot_sphere(p=start_pos, r=0.008, rgba=[0, 1, 0, 0.8], label='Start')
            self.env.plot_sphere(p=end_pos, r=0.008, rgba=[0, 0, 1, 0.8], label='End')
            self.env.plot_line_fr2to(p_fr=start_pos, p_to=end_pos, rgba=[0.5, 0.5, 0.5, 0.6])
        
        if PLOT_EE:
            tip_pos = self.get_applicator_tip_pos()
            self.env.plot_sphere(p=tip_pos, r=0.005, rgba=[1, 0, 0, 0.8], label='Tip')
            
            if hasattr(self, 'target_line'):
                if not self.start_reached:
                    self.env.plot_line_fr2to(p_fr=tip_pos, p_to=self.target_line[0], rgba=[1, 0.5, 0, 1])
                elif not self.end_reached:
                    self.env.plot_line_fr2to(p_fr=tip_pos, p_to=self.target_line[1], rgba=[0, 0.5, 1, 1])
        
        if PLOT_DISTANCE and hasattr(self, 'target_line'):
            start_dist = self.get_distance(self.get_applicator_tip_pos(), self.target_line[0])
            end_dist = self.get_distance(self.get_applicator_tip_pos(), self.target_line[1])
            
            if not self.start_reached:
                distance_text = f'D: {start_dist:.3f}m'
                info_pos = self.target_line[0] + np.array([0, 0, 0.05])
            else:
                distance_text = f'D: {end_dist:.3f}m'
                info_pos = self.target_line[1] + np.array([0, 0, 0.05])

            if PLOT_REWARD and hasattr(self, 'latest_reward'):
                distance_text += f' | R: {self.latest_reward:.3f}'
            
            self.env.plot_text(p=info_pos, label=distance_text)
        
        if PLOT_REWARD and hasattr(self, 'latest_reward_components') and self.latest_reward_components:
            reward_display_pos = np.array([1.2, -0.5, 1.3])
            
            reward_text = f'R: {self.latest_reward:.3f}'
            for key, value in self.latest_reward_components.items():
                if abs(value) > 1e-6:
                    reward_text += f' | {key[:5]}: {value:.3f}'
            
            # Display cumulative episode statistics
            if PLOT_STATS and hasattr(self, 'total_episodes'):
                if self.total_episodes > 0:
                    start_rate = (self.total_start_reached / self.total_episodes) * 100
                    end_rate = (self.total_end_reached / self.total_episodes) * 100
                    stats_text = f'Ep:{self.total_episodes} S:{self.total_start_reached}({start_rate:.0f}%) E:{self.total_end_reached}({end_rate:.0f}%)'
                else:
                    stats_text = f'Ep:{self.total_episodes} S:{self.total_start_reached} E:{self.total_end_reached}'
                
                # Combine and truncate to stay under 99 characters
                combined_text = reward_text + ' | ' + stats_text
                if len(combined_text) > 95:  # Leave some margin
                    # If too long, show stats separately
                    self.env.plot_text(p=reward_display_pos, label=reward_text[:95])
                    stats_pos = reward_display_pos + np.array([0, 0, -0.1])
                    self.env.plot_text(p=stats_pos, label=stats_text[:95])
                else:
                    self.env.plot_text(p=reward_display_pos, label=combined_text)
            else:
                self.env.plot_text(p=reward_display_pos, label=reward_text[:95])
        
        self.env.render()
        
        if hasattr(self.env, 'viewer') and self.env.viewer:
            try:
                self.env.viewer.sync()
            except:
                pass

    def init_viewer(self):
        try:
            self.env.init_viewer(distance=2.0, elevation=-20, azimuth=45)
        except Exception as e:
            print(f"Failed to initialize MuJoCo viewer: {e}")
            raise e
    
    def close_viewer(self):
        self.env.close_viewer()
    
    def is_viewer_alive(self):
        if hasattr(self.env, 'viewer') and self.env.use_mujoco_viewer:
            return self.env.is_viewer_alive()
        return False
    
    def get_sim_time(self):
        return self.env.get_sim_time()

    # ================================================================
    # Heat-map helpers
    # ================================================================
    def _get_relative_coords(self, tip_pos):
        """Return (u, v) coordinates relative to the current target line.

        u : normalized coordinate along the line (0 = start, 1 = end)
        v : perpendicular distance (always >= 0)
        """
        if not hasattr(self, 'target_line'):
            return 0.0, 0.0
        start_pt, end_pt = self.target_line
        line_vec = end_pt - start_pt
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            return 0.0, 0.0
        line_dir = line_vec / line_len
        rel_vec  = tip_pos - start_pt
        proj_len = np.dot(rel_vec, line_dir)
        u = proj_len / line_len  # can be <0 or >1 when outside segment
        perp_vec = rel_vec - proj_len * line_dir
        v = np.linalg.norm(perp_vec)
        return float(u), float(v)
    
    def fixed_status(self, dots):

        if len(dots) == 0:
            return np.array([]), np.array([]), np.array([])

        xs, ys, cs = [], [], []
        for tip, start, end, _, reward in dots:
            tip   = np.asarray(tip)
            start = np.asarray(start)
            end   = np.asarray(end)
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-9:
                continue
            line_dir = line_vec / line_len
            rel_vec  = tip - start
            proj_len = np.dot(rel_vec, line_dir)
            u = proj_len / line_len  # 0 at start, 1 at end
            perp_vec = rel_vec - proj_len * line_dir
            # signed v: sign by dot with arbitrary perpendicular vector (here z-axis cross)
            v = np.linalg.norm(perp_vec)
            xs.append(u)
            ys.append(v)
            cs.append(reward)
        return np.array(xs), np.array(ys), np.array(cs)

    def save_heatmaps(self, filepath, cmap='turbo', u_lim=(-0.3, 1.3), v_lim=(-0.2, 0.2)):
        # Split dots
        before = [d for d in self.heatmap_dots if d[3] is False]
        after  = [d for d in self.heatmap_dots if d[3] is True]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, title, dots in zip(axes, ['Before start_reached', 'After start_reached'], [before, after]):
            xs, ys, cs = self.fixed_status(dots)
            if xs.size > 0:
                vmin_global, vmax_global = -2.0, 2.0
                sc = ax.scatter(xs, ys, c=cs, cmap=cmap, s=8, vmin=vmin_global, vmax=vmax_global)
                fig.colorbar(sc, ax=ax, label='reward', extend='both')

            ax.plot([0, 1], [0, 0], 'k-', lw=2)
            ax.scatter([0], [0], c='green', s=50, label='start')
            ax.scatter([1], [0], c='blue',  s=50, label='end')
            ax.set_xlim(u_lim)
            ax.set_ylim(v_lim)
            ax.set_xlabel('u (along line)')
            ax.set_title(f"{title} (dot N={len(dots)})")
        axes[0].set_ylabel('v (perpendicular distance)')

        fig.suptitle(f"episode {filepath.split('/')[-1].split('.')[0]}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"[Heat-map] saved {filepath}")
        self.heatmap_dots = []

        return


def train_ur5e_sac(
    xml_path='../asset/makeup_frida/scene_table.xml',
    what_changed='',
    start_reached=True,
    dynamic_target=0, # 0 means do dynamic target, 100 means do static target
    result_path=None,
    do_render=False,
    do_log=True,
    reward_mode='2',
    n_episode=1500,
    max_epi_sec=5.0,
    n_warmup_epi=10,
    buffer_limit=50000,
    init_alpha=0.1,  # <-- Increase init_alpha to encourage more exploration (default was 0.1)
    max_torque=1.0,
    lr_actor=0.0005,
    lr_alpha=0.0001,
    lr_critic=0.0001,
    n_update_per_tick=1,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    print_every=50,
    save_every_episode=100,
    plot_heatmap_every_episode=1000,
    seed=0,
    train_from_checkpoint=None,
    ):

    now_date = strftime("%Y-%m-%d")
    now_time = strftime("%H-%M-%S")
    if result_path is None:
        result_path = f'./result/weights/{now_date}_sac_ur5e/'
    # Directory for heatmaps: ./result/heatmap/{now_time}/
    heatmap_dir = f'./result/heatmap/{now_time}/'
    os.makedirs(heatmap_dir, exist_ok=True)

    buffer_warmup=buffer_limit // 5
    init_timer = TimeTracker()
    
    init_timer.start_timer('env_initialization')
    env = MuJoCoParserClass(name=f'UR5e_{what_changed}_{now_time}', rel_xml_path=xml_path, verbose=True)
    gym = UR5eHeadTouchEnv(env=env, HZ=50, reward_mode=reward_mode, start_reached=start_reached, dynamic_target=dynamic_target)
    max_epi_tick=int(max_epi_sec * gym.HZ)
    init_timer.end_timer('env_initialization')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    init_timer.start_timer('buffer_initialization')
    replay_buffer = ReplayBufferClass(buffer_limit, device=device)
    init_timer.end_timer('buffer_initialization')
    
    actor_arg = {
        'obs_dim': gym.o_dim, 'h_dims': [256, 256], 'out_dim': gym.a_dim,
        'max_out': max_torque, 'init_alpha': init_alpha, 'lr_actor': lr_actor,
        'lr_alpha': lr_alpha, 'device': device
    }
    critic_arg = {
        'obs_dim': gym.o_dim, 'a_dim': gym.a_dim, 'h_dims': [256, 256], 'out_dim': 1,
        'lr_critic': lr_critic, 'device': device
    }
    
    init_timer.start_timer('network_initialization')
    actor = ActorClass(**actor_arg).to(device)
    critic_one = CriticClass(**critic_arg).to(device)
    critic_two = CriticClass(**critic_arg).to(device)
    critic_one_trgt = CriticClass(**critic_arg).to(device)
    critic_two_trgt = CriticClass(**critic_arg).to(device)
    init_timer.end_timer('network_initialization')

    # Load checkpoint if specified
    start_episode = 0  # Always start counting episodes from 0 even when loading a checkpoint
    if train_from_checkpoint is not None:
        if os.path.exists(train_from_checkpoint):
            print(f"Loading checkpoint from: {train_from_checkpoint}")
            try:
                init_timer.start_timer('checkpoint_loading')
                checkpoint = torch.load(train_from_checkpoint, map_location=device)
                
                # Load actor state
                if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
                    # Full checkpoint with multiple components
                    actor.load_state_dict(checkpoint['actor_state_dict'])
                    if 'critic_one_state_dict' in checkpoint:
                        critic_one.load_state_dict(checkpoint['critic_one_state_dict'])
                    if 'critic_two_state_dict' in checkpoint:
                        critic_two.load_state_dict(checkpoint['critic_two_state_dict'])
                    if 'critic_one_trgt_state_dict' in checkpoint:
                        critic_one_trgt.load_state_dict(checkpoint['critic_one_trgt_state_dict'])
                    if 'critic_two_trgt_state_dict' in checkpoint:
                        critic_two_trgt.load_state_dict(checkpoint['critic_two_trgt_state_dict'])
                    if 'episode' in checkpoint:
                        resume_ep = checkpoint['episode']
                        print(f"Loaded checkpoint (was saved at episode {resume_ep}), but will train a fresh horizon of {n_episode} episodes.")
                    
                    loaded_optimizers = {}
                    if 'actor_optimizer_state_dict' in checkpoint and checkpoint['actor_optimizer_state_dict'] is not None:
                        loaded_optimizers['actor'] = checkpoint['actor_optimizer_state_dict']
                    if 'alpha_optimizer_state_dict' in checkpoint and checkpoint['alpha_optimizer_state_dict'] is not None:
                        loaded_optimizers['alpha'] = checkpoint['alpha_optimizer_state_dict']
                    if 'critic_one_optimizer_state_dict' in checkpoint and checkpoint['critic_one_optimizer_state_dict'] is not None:
                        loaded_optimizers['critic_one'] = checkpoint['critic_one_optimizer_state_dict']
                    if 'critic_two_optimizer_state_dict' in checkpoint and checkpoint['critic_two_optimizer_state_dict'] is not None:
                        loaded_optimizers['critic_two'] = checkpoint['critic_two_optimizer_state_dict']
                    
                    actor._loaded_optimizer_states = loaded_optimizers
                    
                    print("Loaded full checkpoint with model states and optimizers")
                else:
                    actor.load_state_dict(checkpoint)
                    
                filename = os.path.basename(train_from_checkpoint)
                if 'episode_' in filename:
                    try:
                        start_episode = int(filename.split('episode_')[-1].split('.')[0])
                        print(f"Extracted episode number from filename: {start_episode}")
                    except:
                        print("Could not extract episode number from filename")
                
                print("Loaded simple checkpoint with actor state only")
            
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch...")
                start_episode = 0
            finally:
                init_timer.end_timer('checkpoint_loading')
        else:
            print(f"Checkpoint file not found: {train_from_checkpoint}")
            print("Starting training from scratch...")
            start_episode = 0
    
    if do_log:
        import wandb
        wandb.init(project="ur5e-head-touch", name=f"{strftime('%Y-%m-%d_%H-%M-%S')}")
        wandb.config.update({
            "max_epi_sec": max_epi_sec,
            "n_warmup_epi": n_warmup_epi,
            "buffer_limit": buffer_limit,
            "init_alpha": init_alpha,
            "max_torque": max_torque,
            "lr_actor": lr_actor,
            "lr_alpha": lr_alpha,
            "lr_critic": lr_critic,
            "n_update_per_tick": n_update_per_tick,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
        })
        wandb.watch(actor, log="all")
        
    # Initialize target networks (only if not loaded from full checkpoint)
    if not (hasattr(actor, '_loaded_optimizer_states') and 
            'critic_one_trgt_state_dict' in str(train_from_checkpoint) if train_from_checkpoint else False):
        critic_one_trgt.load_state_dict(critic_one.state_dict())
        critic_two_trgt.load_state_dict(critic_two.state_dict())
    
    # Load optimizer states if available from checkpoint
    if hasattr(actor, '_loaded_optimizer_states'):
        loaded_optimizers = actor._loaded_optimizer_states
        try:
            if 'actor' in loaded_optimizers and hasattr(actor, 'optimizer'):
                actor.optimizer.load_state_dict(loaded_optimizers['actor'])
                print("Loaded actor optimizer state")
            if 'alpha' in loaded_optimizers and hasattr(actor, 'alpha_optimizer'):
                actor.alpha_optimizer.load_state_dict(loaded_optimizers['alpha'])
                print("Loaded alpha optimizer state")
            if 'critic_one' in loaded_optimizers and hasattr(critic_one, 'optimizer'):
                critic_one.optimizer.load_state_dict(loaded_optimizers['critic_one'])
                print("Loaded critic_one optimizer state")
            if 'critic_two' in loaded_optimizers and hasattr(critic_two, 'optimizer'):
                critic_two.optimizer.load_state_dict(loaded_optimizers['critic_two'])
                print("Loaded critic_two optimizer state")
        except Exception as e:
            print(f"Warning: Could not load optimizer states: {e}")
        
        # Clean up temporary attribute
        delattr(actor, '_loaded_optimizer_states')
    
    print("Starting training...")
    end_episode = start_episode + n_episode
    if start_episode > 0:
        print(f"Resuming from episode {start_episode}, will train for {n_episode} more episodes (until episode {end_episode})")
    else:
        print(f"Training from scratch for {n_episode} episodes")
    print(f"Max episode time: {max_epi_sec}s")
    print(f"Buffer size: {buffer_limit}, Warmup: {buffer_warmup}")
    
    if do_render:
        init_timer.start_timer('viewer_initialization')
        gym.init_viewer()
        init_timer.end_timer('viewer_initialization')
    else:
        gym.env.use_mujoco_viewer = False
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    timer = TimeTracker()
    timer.times = init_timer.times
    
    print("\nInitialization timing:")
    init_timer.print_stats()
    
    gym.reset()
    
    
    for epi_idx in tqdm(range(start_episode, end_episode + 1)):
        # Calculate progression from 0 to 1 based on current training progress
        progress = (epi_idx - start_episode) / n_episode
        one_to_zero = 1 - progress
        
        timer.start_timer('env_reset')
        s = gym.reset()
        timer.end_timer('env_reset')
        
        USE_RANDOM_POLICY = (np.random.rand() < (0.1 * one_to_zero)) or (epi_idx < n_warmup_epi)
        
        reward_total = 0.0
        success_count = 0
        
        for tick in range(max_epi_tick):
            # Select action
            if USE_RANDOM_POLICY:
                timer.start_timer('action_random')
                a_np = gym.sample_action()
                timer.end_timer('action_random')
            else:
                timer.start_timer('action_neural')
                a, log_prob = actor(np2torch(s, device=device))
                a_np = torch2np(a)
                timer.end_timer('action_neural')
            
            # Step environment
            timer.start_timer('env_step')
            
            # Break down env_step into components
            timer.start_timer('env_step_physics')
            s_prime, reward, done, info = gym.step(a_np, max_time=max_epi_sec)
            timer.end_timer('env_step_physics')
            
            timer.start_timer('env_step_heatmap')
            gym.heatmap_dots.append((gym.get_applicator_tip_pos(), gym.target_line[0], gym.target_line[1], gym.start_reached, reward))
            timer.end_timer('env_step_heatmap')
            
            timer.end_timer('env_step')
            
            timer.start_timer('replay_buffer_put')
            replay_buffer.put((s, a_np, reward, s_prime, done))
            timer.end_timer('replay_buffer_put')
            
            reward_total += reward
            if info['success']:
                success_count += 1
            
            s = s_prime
            
            if done:
                break

            if do_render and ((tick % 1) == 0):
                timer.start_timer('render')
                if gym.is_viewer_alive() and gym.env.use_mujoco_viewer:
                    gym.render()
                else:
                    # Don't break training! Just disable rendering if viewer fails
                    print("Viewer not alive, disabling rendering for this episode...")
                    do_render = False
                timer.end_timer('render')
            
            if replay_buffer.size() > buffer_warmup:
                timer.start_timer('neural_training')
                for _ in range(n_update_per_tick):
                    timer.start_timer('replay_buffer_sample')
                    mini_batch = replay_buffer.sample(batch_size)
                    timer.end_timer('replay_buffer_sample')
        
                    timer.start_timer('td_target_computation')
                    td_target = get_target(
                        actor, critic_one_trgt, critic_two_trgt,
                        gamma=gamma, mini_batch=mini_batch, device=device
                    )
                    timer.end_timer('td_target_computation')
                    
                    timer.start_timer('critic_training')
                    critic_one.train(td_target, mini_batch)
                    critic_two.train(td_target, mini_batch)
                    timer.end_timer('critic_training')
                    
                    timer.start_timer('actor_training')
                    actor.train(
                        critic_one, critic_two,
                        target_entropy=-gym.a_dim,
                        mini_batch=mini_batch
                    )
                    timer.end_timer('actor_training')
                    
                    timer.start_timer('soft_update')
                    critic_one.soft_update(tau=tau, net_target=critic_one_trgt)
                    critic_two.soft_update(tau=tau, net_target=critic_two_trgt)
                    timer.end_timer('soft_update')
                timer.end_timer('neural_training')
        
        
        if do_log:
            timer.start_timer('wandb_logging')
            wandb.log({
                'episode_reward': reward_total,
                'success_rate': success_count / max_epi_tick,
            })
            timer.end_timer('wandb_logging')

        if (epi_idx % print_every) == 0:
            start_distance = info['distance_start']
            end_distance = info['distance_end']
            final_reward = reward
            print(f"[{epi_idx}/{n_episode}] final_reward: [{final_reward:.3f}] start: [{info['start_reached']}] end: [{info['end_reached']}] sum_reward:[{reward_total:.2f}] start_distance:[{start_distance:.3f}] end_distance:[{end_distance:.3f}] epi_len:[{tick}/{max_epi_tick}] buffer_size:[{replay_buffer.size()}] alpha:[{actor.log_alpha.exp():.2f}]")
            
            # Print timing statistics every 1000 episodes
            if epi_idx > 0:
                timer.print_stats()
                # Reset counters to avoid memory buildup
                timer.reset_counters()
        
        if (epi_idx % plot_heatmap_every_episode) == 0 and epi_idx > 0:
            timer.start_timer('heatmap_save')
            heatmap_path = os.path.join(heatmap_dir, f'{epi_idx}.png')
            gym.save_heatmaps(filepath=heatmap_path)
            timer.end_timer('heatmap_save')
            
        if (epi_idx % save_every_episode) == 0 and epi_idx > 0:
            timer.start_timer('model_save')
            pth_path = f'./result/weights/{now_date}_sac_{gym.name.lower()}/episode_{epi_idx}.pth'
            dir_path = os.path.dirname(pth_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(actor.state_dict(), pth_path)
            timer.end_timer('model_save')
            print(f"  [Save] [{pth_path}] saved.")

            # save
            test_result = test_ur5e_sac(gym, actor, device, max_epi_sec=max_epi_sec, do_render=do_render)
            print(f"Test result: {test_result}")
            wandb.log({**test_result})
            
    
    if do_render and gym.is_viewer_alive():
        gym.close_viewer()
    
    # Print final timing analysis
    print("\n" + "="*80)
    print("FINAL TIMING ANALYSIS")
    print("="*80)
    timer.print_stats()



    
def test_ur5e_sac(gym, actor, device, max_epi_sec=10.0, do_render=True, n_test_episodes=3):
    """
    Test the trained UR5e SAC agent and save the best episode as GIF
    """
    max_epi_tick = int(max_epi_sec * 50)
    
    if do_render:
        try:
            gym.init_viewer()
        except Exception as e:
            print(f"Failed to initialize viewer: {e}")
            do_render = False
    
    test_rewards = []
    test_distances = []  
    test_success_rates = []
    
    for test_episode in range(n_test_episodes):
        s = gym.reset()
        
        episode_reward = 0.0
        episode_success = False
        final_distance = 0.0
        best_distance = float('inf')
        
        for tick in range(max_epi_tick):
            with torch.no_grad():
                a, _ = actor(np2torch(s, device=device), SAMPLE_ACTION=False)
                a_np = torch2np(a)
            
            s_prime, reward, done, info = gym.step(a_np, max_time=max_epi_sec)
            episode_reward += reward
            
            current_distance = info.get('distance_end', info.get('distance', 0.0))
            if current_distance < best_distance:
                best_distance = current_distance
            
            if info.get('success', False) or info.get('end_reached', False):
                episode_success = True
            
            if do_render and ((tick % 2) == 0):
                if gym.is_viewer_alive():
                    gym.render(PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True)
                    time.sleep(0.02)
                else:
                    print("Viewer closed, stopping rendering...")
                    do_render = False
                    break

            s = s_prime
            
            if done:
                break
        
        final_distance = current_distance
        test_rewards.append(episode_reward)
        test_distances.append(final_distance) 
        test_success_rates.append(1.0 if episode_success else 0.0)
        
        if do_render:
            time.sleep(2.0)
    
    if do_render:
        gym.close_viewer()
    
    result = {
        'rewards': np.mean(test_rewards),
        'distances': np.mean(test_distances),
        'success_rates': np.mean(test_success_rates),
    }
    
    return result


if __name__ == "__main__":
    import fire
    fire.Fire(train_ur5e_sac)