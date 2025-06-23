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

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class UR5eHeadTouchEnv:
    """
    UR5e Head Touch Environment similar to homework.py HeadTouchEnv
    """
    def __init__(self, env, HZ=50, history_total_sec=1.0, history_intv_sec=0.1, 
                 object_table_position=[1.0, 0, 0], base_table_position=[0, 0, 0],
                 head_position=[1, 0.0, 0.53], waiting_time=0.0, normalize_obs=True, 
                 normalize_reward=True, VERBOSE=True):
        """
        Initialize UR5e Head Touch Environment
        """
        self.env = env  # MuJoCo environment
        self.HZ = HZ
        self.dt = 1/self.HZ
        self.history_total_sec = history_total_sec
        self.n_history = int(self.HZ * self.history_total_sec)
        self.history_intv_sec = history_intv_sec
        self.history_intv_tick = int(self.HZ * self.history_intv_sec)
        self.history_ticks = np.arange(0, self.n_history, self.history_intv_tick)
        
        self.mujoco_nstep = self.env.HZ // self.HZ
        self.VERBOSE = VERBOSE
        self.tick = 0
        
        self.name = env.name
        self.head_position = np.array(head_position)
        self.waiting_time = waiting_time
        self.object_table_position = np.array(object_table_position)
        self.base_table_position = np.array(base_table_position)
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        
        # Joint names for control (get from available revolute joints)
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # Validate joint names exist in the model
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
        
        # Initialize dynamic target system
        self.target_pos = self.head_position.copy()
        self.action_prev = self.sample_action()
        
        # Observation normalization statistics
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0
        
        # Reward normalization statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # Initialize dimensions
        self.state_dim = len(self._get_observation_raw())
        self.state_history = np.zeros((self.n_history, self.state_dim))
        self.tick_history = np.zeros((self.n_history, 1))
        self.o_dim = len(self.get_observation())
        self.a_dim = len(self.joint_names)
        
        self.prev_distance = self.get_distance_to_target()
        self.best_distance = float('inf')
        self.initial_distance = self.get_distance_to_target()
        
        if VERBOSE:
            print(f"[{self.name}] UR5e Head Touch Environment Instantiated")
            print(f"   [info] dt:[{self.dt:.4f}] HZ:[{self.HZ}], env-HZ:[{self.env.HZ}], mujoco_nstep:[{self.mujoco_nstep}]")
            print(f"   [info] state_dim:[{self.state_dim}], o_dim:[{self.o_dim}], a_dim:[{self.a_dim}]")
            print(f"   [info] joint_names:{self.joint_names}")
            print(f"   [history] total_sec:[{self.history_total_sec:.2f}]sec, n:[{self.n_history}], intv_sec:[{self.history_intv_sec:.2f}]sec")
            print(f"   [head] position:{self.head_position}, waiting_time:{self.waiting_time}s")
            print(f"   [normalization] obs:{self.normalize_obs}, reward:{self.normalize_reward}")
            
            # Print object detection status
            has_applicator = 'applicator_tip' in self.env.body_names
            has_head = 'obj_head' in self.env.body_names
            has_base_table = 'base_table' in self.env.body_names
            print(f"   [objects] applicator_tip:{has_applicator}, obj_head:{has_head}, base_table:{has_base_table}")
            if has_head:
                print(f"   [head_pos] obj_head position: {self.get_object_position('obj_head')}")
            if has_base_table:
                print(f"   [table_pos] base_table position: {self.get_object_position('base_table')}")

    def _dynamic_target(self, p_target, x=[0.0, 0.0], y=[0.0, 0.0], z=[0.0, 0.0]):
        offset = np.array([
            np.random.uniform(x[0], x[1]), 
            np.random.uniform(y[0], y[1]), 
            np.random.uniform(z[0], z[1])
        ])
        
        # Apply offset to base target position
        new_target = p_target + offset
        
        return new_target

    def get_end_effector_pos(self):
        """
        Get applicator tip position for distance calculation
        """
        try:
            # Primary: use applicator tip (like homework.py)
            if 'applicator_tip' in self.env.body_names:
                return self.env.get_p_body('applicator_tip')
        except:
            pass
        
        # Fallback to other possible end-effector names
        possible_ee_names = [
            'wrist_3_link', 'ee_link', 'tool0', 'gripper_base_link', 
            'gripper_finger1_finger_tip_link', 'gripper_finger2_finger_tip_link'
        ]
        
        for ee_name in possible_ee_names:
            try:
                if ee_name in self.env.body_names:
                    return self.env.get_p_body(ee_name)
            except:
                continue
        
        # Fallback: use the last controllable joint position
        try:
            return self.env.get_p_joint(self.joint_names[-1])
        except:
            # Final fallback
            return np.array([0.0, 0.0, 0.0])

    def get_applicator_tip_pos(self):
        """
        Get applicator tip position specifically (same as get_end_effector_pos for consistency)
        """
        return self.get_end_effector_pos()

    def get_object_position(self, object_name):
        """
        Get position of objects like obj_head, base_table
        """
        try:
            if object_name in self.env.body_names:
                return self.env.get_p_body(object_name)
            else:
                # Fallback to preset positions if object not found
                if object_name == 'obj_head':
                    return self.head_position.copy()
                elif object_name == 'base_table':
                    return self.base_table_position.copy()
                elif object_name == 'obj_table':
                    return self.object_table_position.copy()
                else:
                    return np.zeros(3)
        except:
            # Fallback to preset positions
            if object_name == 'obj_head':
                return self.head_position.copy()
            elif object_name == 'base_table':
                return self.base_table_position.copy()
            elif object_name == 'obj_table':
                return self.object_table_position.copy()
            else:
                return np.zeros(3)

    def get_distance_to_target(self):
        """
        Get distance from applicator tip to target
        """
        tip_pos = self.get_applicator_tip_pos()
        return np.linalg.norm(tip_pos - self.target_pos)

    def _get_observation_raw(self):
        """
        Get raw observation similar to homework.py
        """
        # Joint positions and velocities
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            qpos = self.env.data.qpos[joint_idxs]
            qvel = self.env.data.qvel[joint_idxs]
        except:
            qpos = self.env.data.qpos[:self.a_dim]
            qvel = self.env.data.qvel[:self.a_dim]
        
        # Applicator tip position
        tip_pos = self.get_applicator_tip_pos()
        
        # Object positions
        head_pos = self.get_object_position('obj_head')
        base_table_pos = self.get_object_position('base_table')
        
        # Current target position
        target_pos = self.target_pos.copy()
        
        # Distance to target
        distance = self.get_distance_to_target()
        
        # Vector from applicator tip to target
        tip_to_target = target_pos - tip_pos
        
        # Previous action
        prev_action = self.action_prev.copy()
        
        # Concatenate state
        state = np.concatenate([
            qpos,                    # Joint positions
            qvel,                    # Joint velocities  
            tip_pos,                 # Applicator tip position
            head_pos,                # Head object position
            base_table_pos,          # Base table position
            target_pos,              # Target position
            tip_to_target,           # Vector from tip to target
            [distance],              # Distance to target
            prev_action              # Previous action
        ])
        
        return state

    def _update_obs_stats(self, obs):
        """Update observation statistics for normalization"""
        self.obs_count += 1
        if self.obs_mean is None:
            self.obs_mean = obs.copy()
            self.obs_std = np.ones_like(obs)
        else:
            # Running average
            alpha = 1.0 / self.obs_count
            self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs
            self.obs_std = (1 - alpha) * self.obs_std + alpha * np.abs(obs - self.obs_mean)

    def _normalize_obs(self, obs):
        """Normalize observation"""
        if self.obs_mean is not None:
            # Avoid division by zero
            std = np.maximum(self.obs_std, 1e-6)
            return (obs - self.obs_mean) / std
        return obs

    def _update_reward_stats(self, reward):
        """Update reward statistics for normalization"""
        self.reward_count += 1
        
        # Running average for mean
        alpha = 1.0 / self.reward_count
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward
        
        # Running average for standard deviation
        if self.reward_count > 1:
            self.reward_std = (1 - alpha) * self.reward_std + alpha * abs(reward - self.reward_mean)

    def _normalize_reward(self, reward):
        """Normalize reward to approximately [-1, 1] range"""
        if self.reward_count > 10:  # Start normalizing after some samples
            # Avoid division by zero
            std = max(self.reward_std, 1e-6)
            # Z-score normalization then clip to [-1, 1]
            normalized_reward = (reward - self.reward_mean) / (3 * std)  # Divide by 3*std to make most values in [-1,1]
            return np.clip(normalized_reward, -1.0, 1.0)
        return reward

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

    def step(self, action, max_time=np.inf):
        """
        Step the environment
        """
        self.tick += 1
        self.action_prev = action.copy()
        
        # Store previous distance
        prev_distance = self.get_distance_to_target()
        
        # Apply action
        try:
            ctrl_idxs = self.env.get_idxs_step(joint_names=self.joint_names)
            self.env.step(ctrl=action, ctrl_idxs=ctrl_idxs, nstep=self.mujoco_nstep)
        except:
            # Fallback
            self.env.step(ctrl=action, nstep=self.mujoco_nstep)
        
        # Get current distance
        current_distance = self.get_distance_to_target()
        
        # Update best distance
        if current_distance < self.best_distance:
            self.best_distance = current_distance
        
        # Compute done signal  
        SUCCESS = current_distance < 0.03  # 3cm threshold for head touching
        TIMEOUT = self.get_sim_time() >= max_time
        
        # Check for joint limits violation
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            qpos = self.env.data.qpos[joint_idxs]
            joint_ranges = self.env.joint_ranges[:len(qpos)]
            JOINT_LIMIT_VIOLATION = np.any(qpos < joint_ranges[:, 0]) or np.any(qpos > joint_ranges[:, 1])
        except:
            JOINT_LIMIT_VIOLATION = False
        
        done = SUCCESS or TIMEOUT or JOINT_LIMIT_VIOLATION
        
        # Compute rewards similar to homework.py
        # 1. Distance-based reward (main component)
        distance_improvement = prev_distance - current_distance
        r_distance = distance_improvement * 20.0  # Higher scaling for head touching
        
        # 2. Proximity reward (exponential increase when close)
        r_proximity = np.exp(-current_distance * 10.0) - 1.0
        
        # 3. Success reward
        r_success = 50.0 if SUCCESS else 0.0
        
        # 4. Action smoothness (encourage smooth movements)
        r_smooth = -0.001 * np.sum(np.square(action))
        
        # 5. Stay close to head reward (using applicator tip)
        tip_pos = self.get_applicator_tip_pos()
        head_pos = self.get_object_position('obj_head')
        head_distance = np.linalg.norm(tip_pos - head_pos)
        r_head_vicinity = -head_distance * 0.1
        
        # 6. Time penalty 
        r_time = -0.001
        
        # 7. Joint limit penalty
        r_joint_limit = -10.0 if JOINT_LIMIT_VIOLATION else 0.0
        
        # Total reward
        reward = r_distance + r_proximity + r_success + r_smooth + r_head_vicinity + r_time + r_joint_limit
        
        # Apply reward normalization if enabled
        if self.normalize_reward:
            self._update_reward_stats(reward)
            reward = self._normalize_reward(reward)
        
        # Update history
        self.accumulate_state_history()
        
        # Get next observation
        obs_next = self.get_observation()
        
        # Info dictionary
        info = {
            'distance': current_distance,
            'success': SUCCESS,
            'timeout': TIMEOUT,
            'joint_limit_violation': JOINT_LIMIT_VIOLATION,
            'r_distance': r_distance,
            'r_proximity': r_proximity,
            'r_success': r_success,
            'r_smooth': r_smooth,
            'r_head_vicinity': r_head_vicinity,
            'r_time': r_time,
            'r_joint_limit': r_joint_limit,
            'best_distance': self.best_distance,
            'head_distance': head_distance
        }
        
        return obs_next, reward, done, info

    def reset(self):
        """
        Reset environment similar to homework.py
        """
        self.tick = 0
        self.best_distance = float('inf')
        
        # Reset environment
        self.env.reset(step=True)
        
        # Reset robot and table positions
        self.env.set_p_body(body_name='ur_base', p=[0, 0, 0.5])
        self.env.set_p_body(body_name='base_table', p=self.base_table_position)
        self.env.set_p_body(body_name='object_table', p=self.object_table_position)
        
        self.env.set_p_base_body(body_name='obj_head', p=self.head_position)
        self.env.set_R_base_body(body_name='obj_head', R=rpy2r(np.radians([90, 0, 270])))
        
        self.target_pos = self._dynamic_target(self.head_position, x=[0.03, 0.03], y=[-0.2, 0.2], z=[-0.05, 0.05])
        # self.target_pos = self._dynamic_target(self.head_position, x=[0, 0], y=[0, 0], z=[0, 0])
        
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            # Use a reasonable starting configuration
            init_qpos = np.array([0, -60, 90, -60, -90, 0]) * np.pi / 180  # Convert to radians
            if len(init_qpos) != len(self.joint_names):
                # Fallback to zero configuration
                init_qpos = np.zeros(len(self.joint_names))
            self.env.forward(q=init_qpos, joint_idxs=joint_idxs)
        except Exception as e:
            print(f"Warning: Could not set initial configuration: {e}")
            pass  # Use default reset
        
        # Wait for specified waiting time
        if self.waiting_time > 0:
            for _ in range(int(self.waiting_time * self.HZ)):
                self.env.step(ctrl=np.zeros(len(self.joint_names)), nstep=1)
        
        # Reset observation statistics
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0
        
        # Reset reward statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # Reset history
        self.state_history = np.zeros((self.n_history, self.state_dim))
        self.tick_history = np.zeros((self.n_history, 1))
        
        # Update distances
        self.prev_distance = self.get_distance_to_target()
        self.initial_distance = self.prev_distance
        
        return self.get_observation()

    def accumulate_state_history(self):
        """
        Accumulate state history
        """
        state = self._get_observation_raw()
        
        # Shift history
        self.state_history[1:, :] = self.state_history[:-1, :]
        self.state_history[0, :] = state
        
        self.tick_history[1:, :] = self.tick_history[:-1, :]
        self.tick_history[0, :] = self.tick
    
    def render(self, PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True):
        """
        Render the environment
        """
        # Check if viewer is initialized
        if not hasattr(self.env, 'viewer') or not self.env.use_mujoco_viewer:
            return
            
        if PLOT_TARGET:
            # Plot target as red sphere
            self.env.plot_sphere(p=self.target_pos, r=0.005, rgba=[1, 0, 0, 0.8], label='Target')
        
        if PLOT_EE:
            # Plot applicator tip as blue sphere
            tip_pos = self.get_applicator_tip_pos()
            self.env.plot_sphere(p=tip_pos, r=0.005, rgba=[0, 0, 1, 0.8], label='Tip')
            
            # Plot line from applicator tip to target
            self.env.plot_line_fr2to(p_fr=tip_pos, p_to=self.target_pos, rgba=[0, 1, 0, 0.5])
        
        if PLOT_DISTANCE:
            # Plot distance info
            distance = self.get_distance_to_target()
            self.env.plot_text(
                p=self.target_pos + np.array([0, 0, 0.1]),
                label=f'Dist: {distance:.3f}m'
            )
        
        self.env.render()

    def init_viewer(self):
        """Initialize viewer"""
        try:
            self.env.init_viewer(distance=2.0, elevation=-20, azimuth=45)
        except Exception as e:
            print(f"Failed to initialize MuJoCo viewer: {e}")
            raise e
    
    def close_viewer(self):
        """Close viewer"""
        self.env.close_viewer()
    
    def is_viewer_alive(self):
        """Check if viewer is alive"""
        if hasattr(self.env, 'viewer') and self.env.use_mujoco_viewer:
            return self.env.is_viewer_alive()
        return False
    
    def get_sim_time(self):
        """Get simulation time"""
        return self.env.get_sim_time()


def train_ur5e_sac(
    xml_path='../asset/makeup_frida/scene_table.xml',
    test_only=False,
    n_test_episodes=10,
    result_path=None,
    do_render=False,
    do_log=True,
    n_episode=4000,
    max_epi_sec=10.0,
    n_warmup_epi=10,
    buffer_limit=50000,
    init_alpha=0.1,
    max_torque=1.0,
    lr_actor=0.0005,
    lr_alpha=0.0001,
    lr_critic=0.0001,
    n_update_per_tick=1,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    print_every=1,
    eval_every=50,
    RENDER_EVAL=False,
    ):
    if result_path is None:
        now_date = strftime("%Y-%m-%d")
        result_path = f'./result/weights/{now_date}_sac_ur5e/'

    if test_only:
        os.makedirs(result_path.replace('weights', 'gifs'), exist_ok=True)
        all_pth_files = glob.glob(os.path.join(result_path, '*.pth'))
        all_pth_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Found {len(all_pth_files)} model files to test:")
        for pth_file in all_pth_files:
            print(f"   - {pth_file}")
        
        # Initialize environment and device
        env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path, verbose=False)
        gym = UR5eHeadTouchEnv(env=env, HZ=50, VERBOSE=False)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test all models and find the best one
        best_model_path = None
        best_performance = float('-inf')
        all_results = []
        
        print(f"\nTesting {len(all_pth_files)} models...")
        for i, pth_file in enumerate(all_pth_files):
            
            actor = ActorClass(
                obs_dim=gym.o_dim, h_dims=[256, 256], out_dim=gym.a_dim, 
                max_out=max_torque, init_alpha=init_alpha, lr_actor=lr_actor, 
                lr_alpha=lr_alpha, device=device
            ).to(device)
            
            checkpoint = torch.load(pth_file, map_location=device)
            actor.load_state_dict(checkpoint)
            
            results = test_ur5e_sac(gym=gym, actor=actor, device=device, max_epi_sec=max_epi_sec, 
                                    do_render=True, do_gif=True, n_test_episodes=n_test_episodes,
                                    gif_path=f'{result_path.replace("weights", "gifs")}/{now_date}_{os.path.basename(pth_file).replace(".pth", ".gif")}')
            
            # Calculate performance score (success rate + inverse distance)
            performance_score = results['summary']['success_rate'] * 100 - results['summary']['mean_distance'] * 10
            
            all_results.append({
                'path': pth_file,
                'performance': performance_score,
                'results': results
            })
            
            print(f"{i+1}. {os.path.basename(pth_file)}: Score {performance_score:.2f} (Success: {results['summary']['success_rate']:.1%}, Dist: {results['summary']['mean_distance']:.3f})")
            
            if performance_score > best_performance:
                best_performance = performance_score
                best_model_path = pth_file
        
        print(f"\nBEST MODEL: {os.path.basename(best_model_path)} Performance Score: {best_performance:.2f}")
        return best_model_path


    print("Initializing UR5e environment...")
    print(f"XML path: {xml_path}")
    print(f"Rendering enabled: {do_render}")
    max_epi_tick=int(max_epi_sec * 50)
    buffer_warmup=buffer_limit // 5


    # Initialize environment
    print("Creating MuJoCo parser...")
    env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path, verbose=True)
    print("Creating UR5e Head Touch environment wrapper...")
    gym = UR5eHeadTouchEnv(env=env, HZ=50, VERBOSE=True)
    print("Environment initialization complete.")
    
    # Initialize networks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    if do_log:
        import wandb
        wandb.init(project="ur5e-head-touch", name="sac")
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
        
    # Initialize target networks
    critic_one_trgt.load_state_dict(critic_one.state_dict())
    critic_two_trgt.load_state_dict(critic_two.state_dict())
    
    print("Starting training...")
    print(f"Episodes: {n_episode}, Max episode time: {max_epi_sec}s")
    print(f"Buffer size: {buffer_limit}, Warmup: {buffer_warmup}")
    
    # Initialize viewer if rendering is enabled
    if do_render:
        try:
            print("Initializing viewer...")
            gym.init_viewer()
            print("Viewer initialized successfully.")
        except Exception as e:
            print(f"Error initializing viewer: {e}")
            print("Continuing without rendering...")
            do_render = False
    
    # Training loop
    np.random.seed(seed=0)
    
    for epi_idx in range(n_episode + 1):
        zero_to_one = epi_idx / n_episode
        one_to_zero = 1 - zero_to_one
        
        # Reset environment
        s = gym.reset()
        
        # Determine if using random policy
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

            if do_render and ((tick % 5) == 0):
                if gym.is_viewer_alive():
                    gym.render()
                else:
                    break  # Exit if viewer is closed
            
            if do_log:
                wandb.log({
                    "step": tick,
                    "reward": reward,
                    "success": info['success'],
                    "distance": info['distance'],
                    "best_distance": info['best_distance'],
                    "epi_len": tick,
                    "buffer_size": replay_buffer.size(),
                    "alpha": actor.log_alpha.exp(),
                    "episode": epi_idx,
                    "episode_reward": reward_total,
                })

            # Update networks
            if replay_buffer.size() > buffer_warmup:
                for _ in range(n_update_per_tick):
                    mini_batch = replay_buffer.sample(batch_size)
                    
                    # Update critics
                    td_target = get_target(
                        actor, critic_one_trgt, critic_two_trgt,
                        gamma=gamma, mini_batch=mini_batch, device=device
                    )
                    critic_one.train(td_target, mini_batch)
                    critic_two.train(td_target, mini_batch)
                    
                    # Update actor
                    actor.train(
                        critic_one, critic_two,
                        target_entropy=-gym.a_dim,
                        mini_batch=mini_batch
                    )
                    
                    # Soft update targets
                    critic_one.soft_update(tau=tau, net_target=critic_one_trgt)
                    critic_two.soft_update(tau=tau, net_target=critic_two_trgt)
        
        
        if do_log:
            wandb.log({'episode_reward': reward_total})

        if (epi_idx % print_every) == 0:
            final_distance = info.get('distance', 0.0)
            best_distance = info.get('best_distance', 0.0)
            print(f"[{epi_idx}/{n_episode}] reward:[{reward_total:.1f}] distance:[{final_distance:.3f}] best:[{best_distance:.3f}] epi_len:[{tick}/{max_epi_tick}] buffer_size:[{replay_buffer.size()}] alpha:[{actor.log_alpha.exp():.2f}]")
        

        # Evaluation
        if (epi_idx % eval_every) == 0 and epi_idx > 0:
            if RENDER_EVAL:
                gym.init_viewer()
            
            s = gym.reset()
            eval_reward_total = 0.0
            eval_success = False
            
            for tick in range(max_epi_tick):
                a, _ = actor(np2torch(s, device=device), SAMPLE_ACTION=False)
                s_prime, reward, done, info = gym.step(torch2np(a), max_time=max_epi_sec)
                eval_reward_total += reward
                
                if info['success']:
                    eval_success = True
                
                if RENDER_EVAL and ((tick % 5) == 0):
                    gym.render()
                
                s = s_prime
                
                if RENDER_EVAL and not gym.is_viewer_alive():
                    break
                
                if done:
                    break


            if RENDER_EVAL:
                gym.close_viewer()
            
            eval_distance = info.get('distance', 0.0)
            print(f"  [Eval] reward:[{eval_reward_total:.3f}] distance:[{eval_distance:.3f}] success:[{eval_success}] epi_len:[{tick}/{max_epi_tick}]")
        
        # Save model
        if (epi_idx % 100) == 0 and epi_idx > 0:
            pth_path = f'./result/weights/{now_date}_sac_{gym.name.lower()}/episode_{epi_idx}.pth'
            dir_path = os.path.dirname(pth_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            torch.save(actor.state_dict(), pth_path)
            print(f"  [Save] [{pth_path}] saved.")
    
    print("Training completed!")
    
    # Close viewer if it was initialized
    if do_render and gym.is_viewer_alive():
        gym.close_viewer()

    
def test_ur5e_sac(gym, actor, device, max_epi_sec=10.0, do_render=True, do_gif=False, n_test_episodes=10, gif_path='tmp.gif'):
    """
    Test the trained UR5e SAC agent and save the best episode as GIF
    """
    max_epi_tick = int(max_epi_sec * 50)
    
    # Initialize viewer if rendering is enabled
    if do_render:
        try:
            gym.init_viewer()
        except Exception as e:
            do_render = False
    
    # Initialize renderer for GIF generation (simplified approach)
    renderer = None
    if do_gif:
        try:
            # Use a simpler approach - only create renderer when needed
            print("GIF generation enabled - will create renderer per frame")
        except Exception as e:
            print(f"Failed to setup GIF generation: {e}")
            do_gif = False
    
    test_rewards = []
    test_distances = []  
    test_success_rates = []
    test_episode_lengths = []
    episode_frames = []  # Store frames for each episode
    
    for test_episode in range(n_test_episodes):
        
        s = gym.reset()
        
        episode_reward = 0.0
        episode_success = False
        final_distance = 0.0
        best_distance = float('inf')
        frames = []  # Store frames for this episode
        
        for tick in range(max_epi_tick):
            with torch.no_grad():
                a, _ = actor(np2torch(s, device=device), SAMPLE_ACTION=False)
                a_np = torch2np(a)
            
            # Step environment
            s_prime, reward, done, info = gym.step(a_np, max_time=max_epi_sec)
            episode_reward += reward
            
            # Track metrics
            current_distance = info.get('distance', 0.0)
            if current_distance < best_distance:
                best_distance = current_distance
            
            if info.get('success', False):
                episode_success = True
            

            
            # Capture frame for GIF generation using safer approach
            if do_gif and (tick % 3) == 0:
                try:
                    # Use a minimal renderer with proper cleanup
                    width, height = 320, 240  # Smaller size to reduce memory issues
                    
                    # Check if we have valid model and data
                    if hasattr(gym.env, 'model') and hasattr(gym.env, 'data') and gym.env.model and gym.env.data:
                        # Create renderer with minimal configuration and proper cleanup
                        renderer = mujoco.Renderer(gym.env.model, height=height, width=width)
                        try:
                            renderer.update_scene(gym.env.data)
                            pixels = renderer.render()
                            
                            if pixels is not None and pixels.size > 0:
                                # Convert to proper format
                                if pixels.dtype != np.uint8:
                                    if pixels.max() <= 1.0:
                                        pixels = (pixels * 255).astype(np.uint8)
                                    else:
                                        pixels = pixels.astype(np.uint8)
                                
                                # Add frame if valid
                                if len(pixels.shape) == 3 and pixels.shape[2] == 3:
                                    frames.append(pixels.copy())
                                    
                                    if tick == 0:
                                        print(f"GIF frame captured: {pixels.shape}, range: {pixels.min()}-{pixels.max()}")
                        finally:
                            # Explicitly clean up
                            del renderer
                
                except Exception as e:
                    if tick == 0:
                        print(f"GIF generation failed, disabling: {e}")
                        do_gif = False
            
            # Render if enabled
            if do_render and ((tick % 5) == 0):
                if gym.is_viewer_alive():
                    gym.render()
                else:
                    print("Viewer closed, stopping rendering...")
                    do_render = False
                    break
            
            s = s_prime
            
            if done:
                break
        
        # Store episode data
        final_distance = info.get('distance', 0.0)
        test_rewards.append(episode_reward)
        test_distances.append(final_distance) 
        test_success_rates.append(1.0 if episode_success else 0.0)
        test_episode_lengths.append(tick + 1)
        if do_gif:
            episode_frames.append(frames)
        
        # Brief pause between episodes
        if do_render:
            time.sleep(1.0)
    
    # Find best episode (highest reward or success + lowest distance)
    best_episode_idx = 0
    best_score = float('-inf')
    
    for i in range(n_test_episodes):
        # Score: prioritize success, then reward, then inverse distance
        success_bonus = test_success_rates[i] * 1000
        reward_component = test_rewards[i]
        distance_penalty = test_distances[i] * 10
        
        score = success_bonus + reward_component - distance_penalty
        
        if score > best_score:
            best_score = score
            best_episode_idx = i
    
    # Save best episode as GIF
    if do_gif and episode_frames and episode_frames[best_episode_idx]:
        best_frames = episode_frames[best_episode_idx]
        print(f"Saving GIF with {len(best_frames)} frames from episode {best_episode_idx + 1}")
        
        if len(best_frames) > 0:
            # Validate frames before saving
            valid_frames = []
            for i, frame in enumerate(best_frames):
                if frame is not None and frame.size > 0:
                    # Ensure frame is in correct format (height, width, channels)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        valid_frames.append(frame)
                    else:
                        print(f"Skipping invalid frame {i}: shape={frame.shape}")
                else:
                    print(f"Skipping empty frame {i}")
            
            if len(valid_frames) > 0:
                try:
                    imageio.mimsave(gif_path, valid_frames, duration=0.1, loop=0)
                    print(f"GIF saved successfully: {gif_path} ({len(valid_frames)} frames)")
                except Exception as e:
                    print(f"Error saving GIF: {e}")
            else:
                print("No valid frames to save for GIF")
        else:
            print("No frames captured for GIF")
    
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
    
    if do_gif:
        result['best_episode_idx'] = best_episode_idx
        result['gif_path'] = gif_path if 'gif_path' in locals() else None
    
    return result


if __name__ == "__main__":
    import fire ; fire.Fire(train_ur5e_sac)