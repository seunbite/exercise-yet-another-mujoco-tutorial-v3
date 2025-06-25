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
            normalize_reward=True, 
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
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
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
        
        self.start_reached = False
        self.end_reached = False
        self.touch_threshold = 0.005
        
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
        r_START_TOUCH = 0.5
        r_END_TOUCH = 1
        r_TIME = -0.001
        w_START_TOUCH = -0.2
        w_END_TOUCH = -0.2
        w_LINE_DEV = -0.01

        reward = 0.0
        r_start_distance = 0.0
        r_in_progress = 0.0
        
        if not self.start_reached:
            delta_d = current_distance_start
            r_start_distance = delta_d * w_START_TOUCH
            
            if current_distance_start < self.touch_threshold:
                self.start_reached = True
                print("Start point reached!")
            
        if self.start_reached and not self.end_reached:
            reward += r_START_TOUCH
            delta_d = current_distance_end
            delta_l = self._get_distance_from_line()
            r_in_progress = delta_d * w_END_TOUCH + delta_l * w_LINE_DEV

            if current_distance_end < self.touch_threshold:
                self.end_reached = True
                reward += r_END_TOUCH
        
        reward += r_start_distance + r_in_progress + r_TIME
    
        if self.normalize_reward:
            reward = self._normalize_reward(reward)

        self.reward_count += 1
        self.reward_mean = (self.reward_mean * (self.reward_count - 1) + reward) / self.reward_count
        self.reward_std = np.sqrt((self.reward_std ** 2 * (self.reward_count - 1) + (reward - self.reward_mean) ** 2) / self.reward_count)
        
        reward_components = {
            'r_time': r_TIME,
            'r_start_distance': r_start_distance,
            'r_in_progress': r_in_progress,
        }
        
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
        self.tick = 0
        
        self.env.reset(step=True)
        
        self.env.set_p_body(body_name='ur_base', p=[0, 0, 0.5])
        self.env.set_p_body(body_name='base_table', p=self.base_table_position)
        self.env.set_p_body(body_name='object_table', p=self.object_table_position)
        
        # self.env.set_p_base_body(body_name='obj_head', p=self.head_position)
        # self.env.set_R_base_body(body_name='obj_head', R=rpy2r(np.radians([90, 0, 270])))
        
        self.target_pos_start = self._dynamic_target(self.head_position, x=[0.03, 0.03], y=[-0.2, 0.2], z=[-0.05, 0.05])
        
        line_length = random.uniform(0.1, 0.3)
        z_offset = np.random.uniform(0.05, 0.15) * np.random.choice([-1, 1])
        
        line_direction = np.array([np.random.choice([-1, 1]), np.random.uniform(-0.5, 0.5), z_offset / line_length])
        line_direction = line_direction / np.linalg.norm(line_direction)
        self.target_line = self._make_line(self.target_pos_start, direction=line_direction, length=line_length)
        
        self.start_reached = False
        self.end_reached = False
        
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            init_qpos = np.array([0, -60, 90, -60, -90, 0]) * np.pi / 180
            if len(init_qpos) != len(self.joint_names):
                init_qpos = np.zeros(len(self.joint_names))
            self.env.forward(q=init_qpos, joint_idxs=joint_idxs)
        except Exception as e:
            print(f"Warning: Could not set initial configuration: {e}")
            pass
        if self.waiting_time > 0:
            for _ in range(int(self.waiting_time * self.HZ)):
                self.env.step(ctrl=np.zeros(len(self.joint_names)), nstep=1)
        
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
    
    def render(self, PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True):
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
                info_text = f'{start_dist:.3f}m'
                info_pos = self.target_line[0] + np.array([0, 0, 0.05])
            else:
                info_text = f'{end_dist:.3f}m'
                info_pos = self.target_line[1] + np.array([0, 0, 0.05])
            
            self.env.plot_text(p=info_pos, label=info_text)
        
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
    test_only=False,
    n_test_episodes=3,
    result_path=None,
    do_render=False,
    do_log=True,
    n_episode=1000,
    max_epi_sec=5.0,
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
    save_every_episode=25,
    seed=0,
    ):
    now_date = strftime("%Y-%m-%d")
    if result_path is None:
        result_path = f'./result/weights/{now_date}_sac_ur5e/'

    if test_only:
        os.makedirs(result_path.replace('weights', 'gifs'), exist_ok=True)
        all_pth_files = glob.glob(os.path.join(result_path, '*.pth'))
        all_pth_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Found {len(all_pth_files)} model files to test:")
        for pth_file in all_pth_files:
            print(f"   - {pth_file}")
        
        env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path, verbose=False)
        gym = UR5eHeadTouchEnv(env=env, HZ=50)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
            
            gif_path = f'{result_path.replace("weights", "gifs")}/{os.path.basename(pth_file).replace(".pth", ".gif")}'
            results = test_ur5e_sac(gym=gym, actor=actor, device=device, max_epi_sec=max_epi_sec, 
                                    do_render=True, do_gif=True, n_test_episodes=n_test_episodes,
                                    gif_path=gif_path)
            
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
    buffer_warmup=buffer_limit // 5

    env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path, verbose=True)
    gym = UR5eHeadTouchEnv(env=env, HZ=50)
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
        
    critic_one_trgt.load_state_dict(critic_one.state_dict())
    critic_two_trgt.load_state_dict(critic_two.state_dict())
    
    print("Starting training...")
    print(f"Episodes: {n_episode}, Max episode time: {max_epi_sec}s")
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
    
    for epi_idx in tqdm(range(n_episode + 1)):
        zero_to_one = epi_idx / n_episode
        one_to_zero = 1 - zero_to_one
        
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
            wandb.log({'episode_reward': reward_total})

        if (epi_idx % print_every) == 0:
            start_distance = info['distance_start']
            end_distance = info['distance_end']
            final_reward = reward
            print(f"[{epi_idx}/{n_episode}] final_reward: [{final_reward:.3f}] sum_reward:[{reward_total:.2f}] start_distance:[{start_distance:.3f}] end_distance:[{end_distance:.3f}] epi_len:[{tick}/{max_epi_tick}] buffer_size:[{replay_buffer.size()}] alpha:[{actor.log_alpha.exp():.2f}]")
        

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
        
        if (epi_idx % save_every_episode) == 0 and epi_idx > 0:
            pth_path = f'./result/weights/{now_date}_sac_{gym.name.lower()}/episode_{epi_idx}.pth'
            dir_path = os.path.dirname(pth_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            torch.save(actor.state_dict(), pth_path)
            print(f"  [Save] [{pth_path}] saved.")
    
    print("Training completed!")
    
    if do_render and gym.is_viewer_alive():
        gym.close_viewer()

    
def test_ur5e_sac(gym, actor, device, max_epi_sec=10.0, do_render=True, do_gif=False, n_test_episodes=3, gif_path='tmp.gif'):
    """
    Test the trained UR5e SAC agent and save the best episode as GIF
    """
    max_epi_tick = int(max_epi_sec * 50)
    
    if do_render or do_gif:
        try:
            gym.init_viewer()
        except Exception as e:
            print(f"Failed to initialize viewer: {e}")
            do_render = False
            do_gif = False
    
    renderer = None
    if do_gif:
        try:
            print("GIF generation enabled - will create renderer per frame")
        except Exception as e:
            print(f"Failed to setup GIF generation: {e}")
            do_gif = False
    
    test_rewards = []
    test_distances = []  
    test_success_rates = []
    test_episode_lengths = []
    episode_frames = []
    
    for test_episode in range(n_test_episodes):
        
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
            
            current_distance = info.get('distance', 0.0)
            if current_distance < best_distance:
                best_distance = current_distance
            
            if info.get('success', False):
                episode_success = True
            

            
            if do_gif and (tick % 3) == 0:
                try:
                    if gym.is_viewer_alive():
                        gym.render(PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True)
                        
                        time.sleep(0.05)
                        
                        if hasattr(gym.env, 'viewer') and gym.env.viewer:
                            try:
                                width, height = 320, 240
                                pixels = gym.env.viewer.read_pixels(width, height, depth=False)
                                
                                if pixels is not None and pixels.size > 0:
                                    pixels = np.flipud(pixels)
                                    if pixels.dtype != np.uint8:
                                        if pixels.max() <= 1.0:
                                            pixels = (pixels * 255).astype(np.uint8)
                                        else:
                                            pixels = pixels.astype(np.uint8)
                                    
                                    if len(pixels.shape) == 3 and pixels.shape[2] == 3:
                                        frames.append(pixels.copy())
                                        
                                        if tick == 0:
                                            print(f"GIF frame captured from viewer: {pixels.shape}, range: {pixels.min()}-{pixels.max()}")
                            except Exception as viewer_error:
                                if hasattr(gym.env, 'model') and hasattr(gym.env, 'data'):
                                    renderer = mujoco.Renderer(gym.env.model, height=height, width=width)
                                    try:
                                        renderer.update_scene(gym.env.data)
                                        pixels = renderer.render()
                                        
                                        if pixels is not None and pixels.size > 0:
                                            if pixels.dtype != np.uint8:
                                                if pixels.max() <= 1.0:
                                                    pixels = (pixels * 255).astype(np.uint8)
                                                else:
                                                    pixels = pixels.astype(np.uint8)
                                            
                                            if len(pixels.shape) == 3 and pixels.shape[2] == 3:
                                                frames.append(pixels.copy())
                                                
                                                if tick == 0:
                                                    print(f"GIF frame captured from fallback renderer: {pixels.shape}")
                                    finally:
                                        del renderer
                
                except Exception as e:
                    if tick == 0:
                        print(f"GIF generation failed, disabling: {e}")
                        do_gif = False
            
            if do_render and ((tick % 5) == 0):
                if gym.is_viewer_alive():
                    gym.render(PLOT_TARGET=True, PLOT_EE=True, PLOT_DISTANCE=True)
                else:
                    print("Viewer closed, stopping rendering...")
                    do_render = False
                    break
            
            s = s_prime
            
            if done:
                break
        
        final_distance = info.get('distance', 0.0)
        test_rewards.append(episode_reward)
        test_distances.append(final_distance) 
        test_success_rates.append(1.0 if episode_success else 0.0)
        test_episode_lengths.append(tick + 1)
        if do_gif:
            episode_frames.append(frames)
        
        if do_render:
            time.sleep(1.0)
    
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
    
    if do_gif and episode_frames and episode_frames[best_episode_idx]:
        best_frames = episode_frames[best_episode_idx]
        print(f"Saving GIF with {len(best_frames)} frames from episode {best_episode_idx + 1}")
        
        if len(best_frames) > 0:
            valid_frames = []
            for i, frame in enumerate(best_frames):
                if frame is not None and frame.size > 0:
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
    import fire
    fire.Fire(train_ur5e_sac)