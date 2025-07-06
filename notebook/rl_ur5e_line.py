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
        Get normalized observation
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
        r_END_TOUCH = 100
        r_TIME = -0.00001
        c_START_TOUCH = 0.0
        w_START_TOUCH = -0.2
        w_DISTANCE_IMPROVEMENT = 2.0  # Positive weight for distance improvement
        w_DIRECTIONAL_MOVEMENT = 1.0  # Positive weight for directional movement
        w_STAY_STILL_PENALTY = -0.01  # Penalty for staying still after start reached
        stay_still_threshold = 0.001  # Threshold for considering "staying still" (1mm)

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        r_line_dev = 0.0
        r_directional_movement = 0.0
        r_stay_still_penalty = 0.0
        stop_rewarding = False
        
        if not self.start_reached:
            delta_d = current_distance_start
            r_start_distance = delta_d * w_START_TOUCH + c_START_TOUCH
            
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
                self.still_counter = 0  # Counter for consecutive still steps
            
            # Calculate movement vector and direction
            movement_vector = applicator_tip_pos - self.prev_tip_pos
            movement_magnitude = np.linalg.norm(movement_vector)
            
            # Line direction (from start to end)
            line_direction = self.target_line[1] - self.target_line[0]
            
            # r_directional_movement: positive reward for moving in line direction
            r_directional_movement = 0.0
            if movement_magnitude > 1e-6 and self.target_line_length > 1e-6:
                movement_dir_normalized = movement_vector / movement_magnitude
                line_dir_normalized = line_direction / self.target_line_length
                direction_alignment = np.dot(movement_dir_normalized, line_dir_normalized)
                # Positive reward when aligned with line direction (0 to w_DIRECTIONAL_MOVEMENT)
                r_directional_movement = max(0, direction_alignment) * w_DIRECTIONAL_MOVEMENT
            
            # r_in_progress: positive reward for getting closer to end point
            distance_improvement = self.prev_distance_to_end - current_distance_end
            r_in_progress = max(0, distance_improvement) * w_DISTANCE_IMPROVEMENT  # Only positive when getting closer
            
            # Check if staying still and apply exponential penalty
            if movement_magnitude < stay_still_threshold:
                self.still_counter += 1
                # Exponential penalty that increases with consecutive still steps
                r_stay_still_penalty = w_STAY_STILL_PENALTY * (1.5 ** self.still_counter)
            else:
                self.still_counter = 0  # Reset counter when moving
                r_stay_still_penalty = 0.0
            
            # Update tracking variables
            self.prev_tip_pos = applicator_tip_pos.copy()
            self.prev_distance_to_end = current_distance_end
            
            r_line_dev = 0.0
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        reward += r_start_distance + r_in_progress + r_line_dev + r_TIME + r_directional_movement + r_stay_still_penalty

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
            'r_directional_movement': r_directional_movement,
            'r_stay_still_penalty': r_stay_still_penalty,
        }
        
        return reward, reward_components
    
    def reward_2(self, applicator_tip_pos, current_distance_start, current_distance_end):
        r_START_TOUCH = 0
        r_END_TOUCH = 100
        r_TIME = -0.0001
        w_START_TOUCH = -2
        w_DISTANCE_IMPROVEMENT = -1
        w_LESS_MOVEMENT = -10  # Weight for joint movement penalty

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        r_less_movement = 0.0  # Penalty for joint movement
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
            r_in_progress = current_distance_end * w_DISTANCE_IMPROVEMENT
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        # Calculate less movement penalty (square sum of joint movement)
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            current_qpos = self.env.data.qpos[joint_idxs]
            
            if hasattr(self, 'prev_qpos') and self.prev_qpos is not None:
                joint_movement = current_qpos - self.prev_qpos
                r_less_movement = np.sum(joint_movement ** 2) * w_LESS_MOVEMENT
            
            # Update previous joint positions
            self.prev_qpos = current_qpos.copy()
            
        except Exception as e:
            # Fallback if joint access fails
            r_less_movement = 0.0
            if not hasattr(self, 'prev_qpos'):
                self.prev_qpos = None
        
        reward += r_start_distance + r_in_progress + r_TIME + r_less_movement

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
            'r_less_movement': r_less_movement,
        }
        
        return reward, reward_components
    
    def reward_3(self, applicator_tip_pos, current_distance_start, current_distance_end):
        r_START_TOUCH = 0
        r_END_TOUCH = 100
        r_TIME = -0.0001
        w_START_TOUCH = -2
        w_DISTANCE_IMPROVEMENT = -0.05
        w_LINE_DEVIATION = -0.0001  # Weight for line deviation penalty

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        r_line_deviation = 0.0  # Penalty for line deviation
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
            r_in_progress = current_distance_end * w_DISTANCE_IMPROVEMENT
            
            # Calculate line deviation penalty
            r_line_deviation = self._get_distance_from_line() * w_LINE_DEVIATION
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        reward += r_start_distance + r_in_progress + r_TIME + r_line_deviation

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
            'r_line_deviation': r_line_deviation,
        }
        
        return reward, reward_components
    
    def reward_4(self, applicator_tip_pos, current_distance_start, current_distance_end):
        r_START_TOUCH = 0
        r_END_TOUCH = 100
        r_TIME = -0.0001
        c_START_TOUCH = 0.0
        w_START_TOUCH = -2
        w_DISTANCE_IMPROVEMENT = -0.05
        w_BINARY_LINE_DEVIATION = 1.0  # Weight for binary line deviation reward
        line_success_threshold = 0.02  # Threshold for being "on the line" (2cm)

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        r_binary_line_deviation = 0.0  # Binary reward for being on line
        stop_rewarding = False

        if not self.start_reached:
            delta_d = current_distance_start
            r_start_distance = delta_d * w_START_TOUCH + c_START_TOUCH
            
            if current_distance_start < self.touch_threshold:
                print("Start point reached!")
                self.start_reached = True
                reward += r_START_TOUCH
                stop_rewarding = True
            
        if self.start_reached and not self.end_reached and not stop_rewarding:
            r_in_progress = current_distance_end * w_DISTANCE_IMPROVEMENT
            
            # Calculate binary line deviation reward (1 if on line, 0 if not)
            line_deviation = self._get_distance_from_line()
            if line_deviation <= line_success_threshold:
                r_binary_line_deviation = w_BINARY_LINE_DEVIATION  # Reward for being on the line
            else:
                r_binary_line_deviation = 0.0  # No reward for being off the line
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        reward += r_start_distance + r_in_progress + r_TIME + r_binary_line_deviation

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
            'r_binary_line_deviation': r_binary_line_deviation,
        }
        
        return reward, reward_components
    
    def reward_5(self, applicator_tip_pos, current_distance_start, current_distance_end):
        r_START_TOUCH = 0
        r_END_TOUCH = 100
        r_TIME = -0.0001
        c_START_TOUCH = 0.0
        w_START_TOUCH = -2
        w_DISTANCE_PROGRESS = 1.0  # Weight for distance progress reward
        w_BINARY_LINE_DEVIATION = 1.0  # Weight for binary line deviation reward
        line_success_threshold = 0.02  # Threshold for being "on the line" (2cm)

        reward = 0.0
        r_start_distance = 0.0
        r_distance_progress = 0.0
        r_binary_line_deviation = 0.0
        stop_rewarding = False

        if not self.start_reached:
            delta_d = current_distance_start
            r_start_distance = delta_d * w_START_TOUCH + c_START_TOUCH
            
            if current_distance_start < self.touch_threshold:
                print("Start point reached!")
                self.start_reached = True
                reward += r_START_TOUCH
                stop_rewarding = True
            
        if self.start_reached and not self.end_reached and not stop_rewarding:
            start_point = self.target_line[0]
            end_point = self.target_line[1]
            line_vector = end_point - start_point
            line_length = np.linalg.norm(line_vector)
            
            if line_length > 1e-6:
                tip_vector = applicator_tip_pos - start_point
                projection_scalar = np.dot(tip_vector, line_vector) / (line_length ** 2)
                
                if projection_scalar < 0 or projection_scalar > 1:
                    r_distance_progress = -0.05
                    r_binary_line_deviation = 0.0
                else:
                    distance_to_end = np.linalg.norm(applicator_tip_pos - end_point)
                    normalized_distance = distance_to_end / line_length
                    r_distance_progress = w_DISTANCE_PROGRESS * np.maximum((1 - normalized_distance), 0.0)
                    
                    line_deviation = self._get_distance_from_line()
                    if line_deviation <= line_success_threshold:
                        r_binary_line_deviation = w_BINARY_LINE_DEVIATION
                    else:
                        r_binary_line_deviation = 0.0
            else:
                r_distance_progress = 0.0
                r_binary_line_deviation = 0.0
            
            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
                print("End point reached!")
        
        reward += r_start_distance + r_distance_progress + r_TIME + r_binary_line_deviation

        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_distance_progress': r_distance_progress,
            'r_binary_line_deviation': r_binary_line_deviation,
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
        
        return reward, reward_components, state_info

    def step(self, action, max_time=np.inf):
        self.tick += 1
        scaled_action = self._normalize_action(action)
        
        try:
            ctrl_idxs = self.env.get_idxs_step(joint_names=self.joint_names)
            self.env.step(ctrl=scaled_action, ctrl_idxs=ctrl_idxs, nstep=self.mujoco_nstep)
        except:
            self.env.step(ctrl=scaled_action, nstep=self.mujoco_nstep)
        
        reward, reward_components, state_info = self.compute_reward(action, max_time)
        
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
            self.target_line_length = random.uniform(0.1, 0.15)
            z_offset = np.random.uniform(0.05, 0.15) * np.random.choice([-1, 1])
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
                    self.env.plot_line_fr2to(p_fr=tip_pos, p_to=self.target_line[0], rgba=[1, 0.5, 0, 0.4])
                elif not self.end_reached:
                    self.env.plot_line_fr2to(p_fr=tip_pos, p_to=self.target_line[1], rgba=[0, 0.5, 1, 0.4])
        
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
                    reward_text += f' | {key}: {value:.3f}'
            
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


def train_ur5e_sac(
    xml_path='../asset/makeup_frida/scene_table.xml',
    what_changed='',
    test_only=False,
    start_reached=False,
    n_test_episodes=3,
    dynamic_target=10.0,
    result_path=None,
    do_render=True,
    do_log=True,
    reward_mode='4',
    n_episode=1500,
    max_epi_sec=5.0,
    n_warmup_epi=10,
    buffer_limit=50000,
    init_alpha=0.2,  # <-- Increase init_alpha to encourage more exploration (default was 0.1)
    max_torque=1.0,
    lr_actor=0.0005,
    lr_alpha=0.0001,
    lr_critic=0.0001,
    n_update_per_tick=1,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    print_every=1,
    save_every_episode=50,
    seed=0,
    train_from_checkpoint=None,
    ):
    """
    Train UR5e with SAC algorithm for head touch task.
    
    Args:
        dynamic_target (float): Success rate threshold (%) for switching to dynamic targets.
                               When mean end success rate >= this value, targets become dynamic.
                               Example: 10.0 means switch when success rate reaches 10%.
                               This enables curriculum learning: static targets (easier) → dynamic targets (harder).
    """
    now_date = strftime("%Y-%m-%d")
    now_time = strftime("%H-%M-%S")
    if result_path is None:
        result_path = f'./result/weights/{now_date}_sac_ur5e/'

    if test_only:
        print("=== TESTING MODE ===")
        os.makedirs(result_path.replace('weights', 'gifs'), exist_ok=True)
        all_pth_files = glob.glob(os.path.join(result_path, '*.pth'))
        all_pth_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Found {len(all_pth_files)} model files to test:")
        for pth_file in all_pth_files:
            print(f"   - {pth_file}")
        
        if len(all_pth_files) == 0:
            print(f"No model files found in {result_path}")
            print("Make sure you have trained models saved in the weights directory.")
            return None
        
        print(f"Initializing environment for testing...")
        env = MuJoCoParserClass(name=f'UR5e_Test', rel_xml_path=xml_path, verbose=False)
        gym = UR5eHeadTouchEnv(env=env, HZ=50, reward_mode=reward_mode, start_reached=start_reached, dynamic_target=dynamic_target)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        best_model_path = None
        best_performance = float('-inf')
        all_results = []
        
        print(f"\nTesting {len(all_pth_files)} models with rendering enabled...")
        for i, pth_file in enumerate(all_pth_files):
            print(f"\n--- Testing Model {i+1}/{len(all_pth_files)}: {os.path.basename(pth_file)} ---")
            
            try:
                actor = ActorClass(
                    obs_dim=gym.o_dim, h_dims=[256, 256], out_dim=gym.a_dim, 
                    max_out=max_torque, init_alpha=init_alpha, lr_actor=lr_actor, 
                    lr_alpha=lr_alpha, device=device
                ).to(device)
                
                checkpoint = torch.load(pth_file, map_location=device)
                actor.load_state_dict(checkpoint)
                
                gif_path = f'{result_path.replace("weights", "gifs")}/{os.path.basename(pth_file).replace(".pth", ".gif")}'
                
                # Enable rendering and GIF generation for testing
                results = test_ur5e_sac(
                    gym=gym, 
                    actor=actor, 
                    device=device, 
                    max_epi_sec=max_epi_sec, 
                    do_render=True,  # Always enable rendering in test mode
                    do_gif=True,     # Always enable GIF generation
                    n_test_episodes=n_test_episodes,
                    gif_path=gif_path
                )
                
                performance_score = results['summary']['success_rate'] * 100 - results['summary']['mean_distance'] * 10
                
                all_results.append({
                    'path': pth_file,
                    'performance': performance_score,
                    'results': results
                })
                
                print(f"Results: Score {performance_score:.2f} (Success: {results['summary']['success_rate']:.1%}, Dist: {results['summary']['mean_distance']:.3f})")
                
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_model_path = pth_file
                    
            except Exception as e:
                print(f"Error testing model {pth_file}: {e}")
                continue
        
        if best_model_path:
            print(f"\n🏆 BEST MODEL: {os.path.basename(best_model_path)}")
            print(f"🏆 Performance Score: {best_performance:.2f}")
        else:
            print("No models were successfully tested.")
            
        return best_model_path


    print("Initializing UR5e environment...")
    print(f"XML path: {xml_path}")
    print(f"Rendering enabled: {do_render}")
    buffer_warmup=buffer_limit // 5

    env = MuJoCoParserClass(name=f'UR5e_{what_changed}_{now_time}', rel_xml_path=xml_path, verbose=True)
    gym = UR5eHeadTouchEnv(env=env, HZ=50, reward_mode=reward_mode, start_reached=start_reached, dynamic_target=dynamic_target)
    max_epi_tick=int(max_epi_sec * gym.HZ)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    replay_buffer = ReplayBufferClass(buffer_limit, device=device)
    
    actor_arg = {
        'obs_dim': gym.o_dim, 'h_dims': [256, 256], 'out_dim': gym.a_dim,
        'max_out': max_torque, 'init_alpha': init_alpha, 'lr_actor': lr_actor,
        'lr_alpha': lr_alpha, 'device': device
    }
    critic_arg = {
        'obs_dim': gym.o_dim, 'a_dim': gym.a_dim, 'h_dims': [256, 256], 'out_dim': 1,
        'lr_critic': lr_critic, 'device': device
    }
    
    actor = ActorClass(**actor_arg).to(device)
    critic_one = CriticClass(**critic_arg).to(device)
    critic_two = CriticClass(**critic_arg).to(device)
    critic_one_trgt = CriticClass(**critic_arg).to(device)
    critic_two_trgt = CriticClass(**critic_arg).to(device)
    
    # Load checkpoint if specified
    start_episode = 0
    if train_from_checkpoint is not None:
        if os.path.exists(train_from_checkpoint):
            print(f"Loading checkpoint from: {train_from_checkpoint}")
            try:
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
                        start_episode = checkpoint['episode']
                        print(f"Resuming from episode: {start_episode}")
                    
                    # Load optimizer states (will be set after optimizers are created)
                    loaded_optimizers = {}
                    if 'actor_optimizer_state_dict' in checkpoint and checkpoint['actor_optimizer_state_dict'] is not None:
                        loaded_optimizers['actor'] = checkpoint['actor_optimizer_state_dict']
                    if 'alpha_optimizer_state_dict' in checkpoint and checkpoint['alpha_optimizer_state_dict'] is not None:
                        loaded_optimizers['alpha'] = checkpoint['alpha_optimizer_state_dict']
                    if 'critic_one_optimizer_state_dict' in checkpoint and checkpoint['critic_one_optimizer_state_dict'] is not None:
                        loaded_optimizers['critic_one'] = checkpoint['critic_one_optimizer_state_dict']
                    if 'critic_two_optimizer_state_dict' in checkpoint and checkpoint['critic_two_optimizer_state_dict'] is not None:
                        loaded_optimizers['critic_two'] = checkpoint['critic_two_optimizer_state_dict']
                    
                    # Store for later loading after optimizers are initialized
                    actor._loaded_optimizer_states = loaded_optimizers
                    
                    print("Loaded full checkpoint with model states and optimizers")
                else:
                    # Simple checkpoint with just actor state_dict
                    actor.load_state_dict(checkpoint)
                    
                    # Try to extract episode number from filename
                    filename = os.path.basename(train_from_checkpoint)
                    if 'episode_' in filename:
                        try:
                            start_episode = int(filename.split('episode_')[-1].split('.')[0])
                            print(f"Extracted episode number from filename: {start_episode}")
                        except:
                            print("Could not extract episode number from filename")
                    
                    print("Loaded simple checkpoint with actor state only")
                
                print("Checkpoint loaded successfully!")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch...")
                start_episode = 0
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
        try:
            print("Initializing viewer...")
            gym.init_viewer()
            print("Viewer initialized successfully.")
        except Exception as e:
            print(f"Error initializing viewer: {e}")
            print("Continuing without rendering...")
            do_render = False
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Adjust episode range if resuming from checkpoint
    
    for epi_idx in tqdm(range(start_episode, end_episode + 1)):
        # Calculate progression from 0 to 1 based on current training progress
        progress = (epi_idx - start_episode) / n_episode
        zero_to_one = progress
        one_to_zero = 1 - progress
        
        s = gym.reset()
        
        USE_RANDOM_POLICY = (np.random.rand() < (0.1 * one_to_zero)) or (epi_idx < n_warmup_epi)
        
        reward_total = 0.0
        success_count = 0
        
        for tick in range(max_epi_tick):
            # Select action
            if USE_RANDOM_POLICY:
                a_np = gym.sample_action()
            else:
                a, log_prob = actor(np2torch(s, device=device))
                a_np = torch2np(a)
            
            # Step environment
            s_prime, reward, done, info = gym.step(a_np, max_time=max_epi_sec)
            replay_buffer.put((s, a_np, reward, s_prime, done))
            
            reward_total += reward
            if info['success']:
                success_count += 1
            
            s = s_prime
            
            if done:
                break

            if do_render and ((tick % 1) == 0):
                if gym.is_viewer_alive():
                    gym.render()
                else:
                    break  # Exit if viewer is closed
            
            if do_log:
                wandb.log({
                    "step": tick,
                    "reward": reward,
                    "success": info['success'],
                    "epi_len": tick,
                    "buffer_size": replay_buffer.size(),
                    "alpha": actor.log_alpha.exp(),
                    "episode": epi_idx,
                    "episode_reward": reward_total,
                    "replay_buffer_size": replay_buffer.size(),
                    **{k: v for k, v in info.items() if 'r_' in k}
                })

            if replay_buffer.size() > buffer_warmup:
                for _ in range(n_update_per_tick):
                    mini_batch = replay_buffer.sample(batch_size)
        
                    td_target = get_target(
                        actor, critic_one_trgt, critic_two_trgt,
                        gamma=gamma, mini_batch=mini_batch, device=device
                    )
                    critic_one.train(td_target, mini_batch)
                    critic_two.train(td_target, mini_batch)
                    
                    critic_one_grad_norm = gym.compute_gradient_norm(critic_one)
                    critic_two_grad_norm = gym.compute_gradient_norm(critic_two)
                    
                    actor.train(
                        critic_one, critic_two,
                        target_entropy=-gym.a_dim,
                        mini_batch=mini_batch
                    )
                    
                    actor_grad_norm = gym.compute_gradient_norm(actor)
                    
                    if do_log:
                        wandb.log({
                            "actor_grad_norm": actor_grad_norm,
                            "critic_one_grad_norm": critic_one_grad_norm,
                            "critic_two_grad_norm": critic_two_grad_norm,
                        })
                    
                    critic_one.soft_update(tau=tau, net_target=critic_one_trgt)
                    critic_two.soft_update(tau=tau, net_target=critic_two_trgt)
        
        
        if do_log:
            # Calculate mean success rates
            mean_success_episode_start = gym.total_start_reached / max(gym.total_episodes, 1)
            mean_success_episode_end = gym.total_end_reached / max(gym.total_episodes, 1)
            
            wandb.log({
                'episode_reward': reward_total,
                'mean_success_episode_start': mean_success_episode_start,
                'mean_success_episode_end': mean_success_episode_end,
                'total_episodes': gym.total_episodes,
                'total_start_reached': gym.total_start_reached,
                'total_end_reached': gym.total_end_reached
            })

        if (epi_idx % print_every) == 0:
            start_distance = info['distance_start']
            end_distance = info['distance_end']
            final_reward = reward
            print(f"[{epi_idx}/{n_episode}] final_reward: [{final_reward:.3f}] start: [{info['start_reached']}] end: [{info['end_reached']}] sum_reward:[{reward_total:.2f}] start_distance:[{start_distance:.3f}] end_distance:[{end_distance:.3f}] epi_len:[{tick}/{max_epi_tick}] buffer_size:[{replay_buffer.size()}] alpha:[{actor.log_alpha.exp():.2f}]")
        
        if (epi_idx % save_every_episode) == 0 and epi_idx > 0:
            pth_path = f'./result/weights/{now_date}_sac_{gym.name.lower()}/episode_{epi_idx}.pth'
            dir_path = os.path.dirname(pth_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            # Save simple checkpoint (backward compatibility)
            torch.save(actor.state_dict(), pth_path)
            
            # Save complete checkpoint for resuming training
            checkpoint_path = f'./result/weights/{now_date}_sac_{gym.name.lower()}/checkpoint_episode_{epi_idx}.pth'
            checkpoint = {
                'episode': epi_idx,
                'actor_state_dict': actor.state_dict(),
                'critic_one_state_dict': critic_one.state_dict(),
                'critic_two_state_dict': critic_two.state_dict(),
                'critic_one_trgt_state_dict': critic_one_trgt.state_dict(),
                'critic_two_trgt_state_dict': critic_two_trgt.state_dict(),
                'actor_optimizer_state_dict': actor.optimizer.state_dict() if hasattr(actor, 'optimizer') else None,
                'alpha_optimizer_state_dict': actor.alpha_optimizer.state_dict() if hasattr(actor, 'alpha_optimizer') else None,
                'critic_one_optimizer_state_dict': critic_one.optimizer.state_dict() if hasattr(critic_one, 'optimizer') else None,
                'critic_two_optimizer_state_dict': critic_two.optimizer.state_dict() if hasattr(critic_two, 'optimizer') else None,
                'hyperparameters': {
                    'lr_actor': lr_actor,
                    'lr_alpha': lr_alpha,
                    'lr_critic': lr_critic,
                    'gamma': gamma,
                    'tau': tau,
                    'init_alpha': init_alpha,
                    'max_torque': max_torque,
                    'reward_mode': reward_mode,
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  [Save] [{pth_path}] and [{checkpoint_path}] saved.")
    
    print("Training completed!")
    
    if do_render and gym.is_viewer_alive():
        gym.close_viewer()

    
def test_ur5e_sac(gym, actor, device, max_epi_sec=10.0, do_render=True, do_gif=False, n_test_episodes=3, gif_path='tmp.gif'):
    """
    Test the trained UR5e SAC agent and save the best episode as GIF
    """
    max_epi_tick = int(max_epi_sec * 50)
    
    # Initialize viewer for rendering
    if do_render or do_gif:
        try:
            print("Initializing viewer for testing...")
            gym.init_viewer()
            print("Viewer initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize viewer: {e}")
            do_render = False
            do_gif = False
    
    test_rewards = []
    test_distances = []  
    test_success_rates = []
    test_episode_lengths = []
    episode_frames = []
    
    print(f"Starting {n_test_episodes} test episodes...")
    
    for test_episode in range(n_test_episodes):
        print(f"\n=== Test Episode {test_episode + 1}/{n_test_episodes} ===")
        
        s = gym.reset()
        
        episode_reward = 0.0
        episode_success = False
        final_distance = 0.0
        best_distance = float('inf')
        frames = []
        
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
            
            # Render every few steps
            if do_render and ((tick % 2) == 0):
                if gym.is_viewer_alive():
                    gym.render(PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True)
                    time.sleep(0.02)  # Small delay for smooth visualization
                else:
                    print("Viewer closed, stopping rendering...")
                    do_render = False
                    break
            
            # Capture frames for GIF (simplified)
            if do_gif and ((tick % 5) == 0):
                try:
                    if gym.is_viewer_alive():
                        gym.render(PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True)
                        time.sleep(0.01)
                        
                        # Try to capture frame from viewer
                        if hasattr(gym.env, 'viewer') and gym.env.viewer:
                            try:
                                width, height = 480, 360
                                pixels = gym.env.viewer.read_pixels(width, height, depth=False)
                                
                                if pixels is not None and pixels.size > 0:
                                    pixels = np.flipud(pixels)
                                    if pixels.dtype != np.uint8:
                                        pixels = (pixels * 255).astype(np.uint8)
                                    
                                    if len(pixels.shape) == 3 and pixels.shape[2] == 3:
                                        frames.append(pixels.copy())
                            except Exception as e:
                                if tick == 0:
                                    print(f"GIF frame capture failed: {e}")
                                    do_gif = False
                except Exception as e:
                    if tick == 0:
                        print(f"GIF generation failed: {e}")
                        do_gif = False
            
            s = s_prime
            
            if done:
                break
        
        final_distance = current_distance
        test_rewards.append(episode_reward)
        test_distances.append(final_distance) 
        test_success_rates.append(1.0 if episode_success else 0.0)
        test_episode_lengths.append(tick + 1)
        if do_gif:
            episode_frames.append(frames)
        
        print(f"Episode {test_episode + 1} completed:")
        print(f"  Reward: {episode_reward:.3f}")
        print(f"  Success: {episode_success}")
        print(f"  Final distance: {final_distance:.3f}")
        print(f"  Episode length: {tick + 1}")
        
        if do_render:
            time.sleep(2.0)  # Pause between episodes
    
    # Find best episode for GIF
    best_episode_idx = 0
    best_score = float('-inf')
    
    for i in range(n_test_episodes):
        success_bonus = test_success_rates[i] * 1000
        reward_component = test_rewards[i]
        distance_penalty = test_distances[i] * 10
        
        score = success_bonus + reward_component - distance_penalty
        
        if score > best_score:
            best_score = score
            best_episode_idx = i
    
    # Save GIF from best episode
    if do_gif and episode_frames and len(episode_frames) > best_episode_idx:
        best_frames = episode_frames[best_episode_idx]
        print(f"\nSaving GIF with {len(best_frames)} frames from episode {best_episode_idx + 1}")
        
        if len(best_frames) > 0:
            try:
                imageio.mimsave(gif_path, best_frames, duration=0.1, loop=0)
                print(f"GIF saved successfully: {gif_path}")
            except Exception as e:
                print(f"Error saving GIF: {e}")
        else:
            print("No frames captured for GIF")
    
    # Close viewer
    if do_render or do_gif:
        try:
            gym.close_viewer()
        except:
            pass
    
    # Results summary
    result = {
        'rewards': test_rewards,
        'distances': test_distances,
        'success_rates': test_success_rates,
        'episode_lengths': test_episode_lengths,
        'summary': {
            'mean_reward': np.mean(test_rewards),
            'mean_distance': np.mean(test_distances),
            'success_rate': np.mean(test_success_rates),
            'best_distance': np.min(test_distances)
        }
    }
    
    print(f"\n=== Test Results Summary ===")
    print(f"Mean reward: {result['summary']['mean_reward']:.3f}")
    print(f"Success rate: {result['summary']['success_rate']:.1%}")
    print(f"Mean final distance: {result['summary']['mean_distance']:.3f}")
    print(f"Best distance: {result['summary']['best_distance']:.3f}")
    
    if do_gif:
        result['best_episode_idx'] = best_episode_idx
        result['gif_path'] = gif_path
    
    return result


if __name__ == "__main__":
    import fire
    fire.Fire(train_ur5e_sac)