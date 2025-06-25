import sys,mujoco,time,os,json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import trange
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import torch.nn.functional as F
import wandb

sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/gpt_usage/')
sys.path.append('../package/detection_module/')
from mujoco_parser import *
from utility import *
from transformation import *
from gpt_helper import *
from owlv2 import *


class HeadTouchEnv:
    def __init__(
        self, 
        env, 
        HZ=50, 
        record_video=False, 
        initial_joint_angles=[0, -60, 90, -60, -90, 0], 
        waiting_time=0.0, 
        joint_names=None, 
        head_position=[1, 0, 0.55], normalize_obs=True
        ):
        self.env = env
        self.HZ = HZ
        self.record_video = record_video
        self.frames = []
        self.head_position = head_position
        self.normalize_obs = normalize_obs

        self.all_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_names = joint_names if joint_names is not None else self.all_joint_names
        self.waiting_time = waiting_time
        
        if len(initial_joint_angles) != 6:
            raise ValueError("initial_joint_angles must have exactly 6 values for all UR5e joints")
        self.initial_joint_angles_rad = np.deg2rad(initial_joint_angles)

        # location
        self.env.reset()
        self.env.set_p_body(body_name='ur_base', p=np.array([0, 0, 0.5]))
        self.env.set_p_body(body_name='object_table', p=np.array([1.0, 0, 0]))
        self.env.set_p_base_body(body_name='obj_head', p=self.head_position)
        self.env.set_qpos_joints(joint_names=self.all_joint_names, qpos=self.initial_joint_angles_rad)
        
        if self.record_video:
            try:
                self.env.init_viewer(
                    transparent=False,
                    azimuth=105,
                    distance=3.12,
                    elevation=-29,
                    lookat=[0.39, 0.25, 0.43],
                    maxgeom=100000
                )
                self.viewer_initialized = True
                print("MuJoCo viewer initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize MuJoCo viewer: {e}")
                print("Will use headless rendering for video capture")
                self.viewer_initialized = False
            
            self.env.grab_image_backup = np.zeros((400, 600, 3), dtype=np.uint8)

        else:
            self.viewer_initialized = False
        
        test_obs = self._get_observation_raw()
        self.o_dim = test_obs.shape[0]
        self.a_dim = len(self.joint_names)
        
        if self.normalize_obs:
            self.obs_mean = np.zeros(self.o_dim)
            self.obs_var = np.ones(self.o_dim)
            self.obs_count = 1e-4
            self.obs_min = np.full(self.o_dim, np.inf)
            self.obs_max = np.full(self.o_dim, -np.inf)
        
        self.success_threshold = 0.02
        
        print(f"[HeadTouch] Setting up initial state...")
        self.reset()
        
        print(f"[HeadTouch] Instantiated")
        print(f"   [info] dt:[{1.0/self.HZ:.4f}] HZ:[{self.HZ}], state_dim:[{self.o_dim}], a_dim:[{self.a_dim}]")
        print(f"   [info] All joints: {self.all_joint_names}")
        print(f"   [info] Controllable joints: {self.joint_names}")
        print(f"   [info] Fixed joints: {[j for j in self.all_joint_names if j not in self.joint_names]}")
        if self.record_video:
            print(f"   [info] Video recording enabled")
            if self.viewer_initialized:
                print(f"   [info] MuJoCo viewer available")
            else:
                print(f"   [info] Using headless rendering")
    

    
    def _get_observation_raw(self):
        """Get raw observation without normalization"""
        # Get joint positions for the controllable joints only
        joint_positions = self.env.get_qpos_joints(joint_names=self.joint_names)
        
        # Get joint velocities for controllable joints only
        joint_velocities = self.env.get_qvel_joints(joint_names=self.joint_names)
        
        # Get end-effector position
        p_tip = self.env.get_p_body('applicator_tip')
        try:
            p_target = self.p_target
        except:
            p_target = self.head_position
        
        # Combine all observations
        obs = np.concatenate([
            joint_positions,      # N values (N = number of controllable joints)
            joint_velocities,     # N values  
            p_tip,               # 3 values
            p_target            # 3 values
        ])
        
        return obs
    
    def _update_obs_stats(self, obs):
        """Update running statistics for observation normalization"""
        if not self.normalize_obs:
            return
            
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        self.obs_var += delta * (obs - self.obs_mean)
        
        # Update min/max for bounded normalization
        self.obs_min = np.minimum(self.obs_min, obs)
        self.obs_max = np.maximum(self.obs_max, obs)
    
    def _normalize_obs(self, obs):
        """Normalize observation to [-1, 1] range using min-max normalization"""
        if not self.normalize_obs:
            return obs
            
        # Use min-max normalization to bound values to [-1, 1]
        obs_range = self.obs_max - self.obs_min
        
        # Avoid division by zero for constant values
        obs_range = np.where(obs_range < 1e-8, 1.0, obs_range)
        
        # Normalize to [0, 1] then scale to [-1, 1]
        normalized_obs = (obs - self.obs_min) / obs_range  # [0, 1]
        normalized_obs = 2.0 * normalized_obs - 1.0       # [-1, 1]
        
        # Ensure values are strictly bounded
        normalized_obs = np.clip(normalized_obs, -1.0, 1.0)
        
        return normalized_obs
    
    def _get_observation(self):
        """Get normalized observation"""
        # Get raw observation
        raw_obs = self._get_observation_raw()
        
        # Update running statistics
        self._update_obs_stats(raw_obs)
        
        # Return normalized observation
        return self._normalize_obs(raw_obs)
    
    def _dynamic_target(self, p_target, x, y, z):
        """Dynamic target position with bounds checking"""
        # Generate random offset
        offset = np.array([
            np.random.uniform(x[0], x[1]), 
            np.random.uniform(y[0], y[1]), 
            np.random.uniform(z[0], z[1])
        ])
        
        # Apply offset to base target position
        new_target = p_target + offset
        
        # Ensure target stays within reasonable bounds (e.g., above table surface)
        new_target[2] = np.clip(new_target[2], 0.3, 1.5)  # Z between 30cm and 1.5m
        
        return new_target
    
    def reset(self):
        """Reset environment to initial state"""
        self.env.reset(step=True)
        
        self.prev_dist = None
        if hasattr(self, 'step_count'):
            self.step_count = 0
        
        self.env.set_p_body(body_name='ur_base', p=np.array([0, 0, 0.5]))
        self.env.set_p_body(body_name='object_table', p=np.array([1.0, 0, 0]))
        
        self.env.set_p_base_body(body_name='obj_head', p=self.head_position)
        self.env.set_R_base_body(body_name='obj_head', R=rpy2r(np.radians([0, 270, 0])))

        self.p_target = self._dynamic_target(self.head_position, x=[-0.01, 0.05], y=[-0.1, 0.1], z=[-0.02, 0.02])
        self.env.set_p_body(body_name='target_dot', p=self.p_target)
        self.env.set_qpos_joints(joint_names=self.all_joint_names, qpos=self.initial_joint_angles_rad)
        
        wait_steps = int(self.waiting_time / self.env.dt)
        for step_i in range(wait_steps):
            full_ctrl = np.zeros(6)
            for i, joint_name in enumerate(self.all_joint_names):
                if joint_name not in self.joint_names:
                    current_pos = self.env.get_qpos_joints(joint_names=[joint_name])
                    target_pos = self.initial_joint_angles_rad[i]
                    full_ctrl[i] = 5.0 * (target_pos - current_pos)
            
            idxs_step = self.env.get_idxs_step(joint_names=self.all_joint_names)
            self.env.step(ctrl=full_ctrl, ctrl_idxs=idxs_step)
        
        return self._get_observation()
    
    def step(self, action, max_time=60.0):
        """Execute action and get next observation
        
        Args:
            action: Normalized actions in [-1, 1] range for controllable joints
        """
        # Create full control vector for all joints
        full_ctrl = np.zeros(6)  # 6 joints total
        
        # Set fixed joints to their initial positions (hold them fixed)
        for i, joint_name in enumerate(self.all_joint_names):
            if joint_name not in self.joint_names:
                # This is a fixed joint - hold it at initial position
                current_pos = self.env.get_qpos_joints(joint_names=[joint_name])
                target_pos = self.initial_joint_angles_rad[i]
                # Use PD control to hold fixed joints at their initial positions
                full_ctrl[i] = 10.0 * (target_pos - current_pos)  # Strong position control
        
        
        # Set controllable joints to the provided actions with proper scaling
        controllable_idx = 0
        for i, joint_name in enumerate(self.all_joint_names):
            if joint_name in self.joint_names:
                full_ctrl[i] = action[controllable_idx]
                controllable_idx += 1
        
        # Apply control to all joints
        idxs_step = self.env.get_idxs_step(joint_names=self.all_joint_names)
        self.env.step(ctrl=full_ctrl, ctrl_idxs=idxs_step)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        p_tip = self.env.get_p_body('applicator_tip')  # Get applicator tip position
        
        dist = np.linalg.norm(p_tip - self.p_target)
        
        # NORMALIZED REWARD FUNCTION - bounded to [-1, 1]
        if dist < self.success_threshold:
            # Maximum reward for success
            reward = 1.0
        else:
            # Dense distance-based reward using exponential decay
            max_expected_dist = 2.0  # Maximum expected distance
            
            # Map distance to [0, 1] range using exponential decay
            # Close distances (0m) → 1.0, far distances (2m+) → ~0.0
            distance_reward = np.exp(-3.0 * dist / max_expected_dist)
            
            # Scale to [0.1, 0.9] to leave room for time penalty and success bonus
            # This ensures we don't hit the bounds too often during normal operation
            distance_reward = 0.1 + 0.8 * distance_reward
            
            # Small time penalty to encourage efficiency
            time_penalty = 0.01
            
            # Final reward bounded to [-1, 1]
            reward = distance_reward - time_penalty
            
        # Ensure reward is strictly bounded to [-1, 1]
        reward = np.clip(reward, -1.0, 1.0)
        
        # Episode is done if target is reached or max time exceeded
        done = (dist < self.success_threshold) or (self.env.get_sim_time() > max_time)
        
        info = {
            'dist': dist,
            'p_tip': p_tip,
            'p_target': self.p_target
        }
        
        return obs, reward, done, info
    
                    
    def render(self):
        """Render the current state"""
        if hasattr(self, 'viewer_initialized') and self.viewer_initialized:
            try:
                self.env.render()
            except Exception as e:
                print(f"Warning: Render failed: {e}")
    
# PPO implementation
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared layers - SMALLER network to prevent overfitting
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) head
        self.critic = nn.Linear(64, 1)
        
    def forward(self, state):
        shared_features = self.shared(state)
        
        # Actor - use tanh to bound actions to [-1, 1]
        mean = torch.tanh(self.actor_mean(shared_features))
        std = torch.exp(self.actor_logstd)
        
        # Critic
        value = self.critic(shared_features)
        
        return mean, std, value
    
    def get_value(self, state):
        """Get only the value estimate"""
        shared_features = self.shared(state)
        return self.critic(shared_features)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.3, exploration_noise=0.7, eps=1e-8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Show which device is being used
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)
        self.gamma = gamma
        self.epsilon = epsilon
        self.exploration_noise = exploration_noise  # Add exploration noise
    
    def calculate_detailed_gradient_norms(self):
        """Calculate gradient norms for different parts of the network"""
        total_norm = 0.0
        actor_norm = 0.0
        critic_norm = 0.0
        layer_norms = {}
        
        # Calculate gradient norms for each layer
        for name, param in self.actor_critic.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                layer_norms[f'grad_norm_{name}'] = param_norm.item()
                total_norm += param_norm.item() ** 2
                
                # Categorize by actor/critic
                if any(keyword in name.lower() for keyword in ['actor', 'mean', 'std']):
                    actor_norm += param_norm.item() ** 2
                elif any(keyword in name.lower() for keyword in ['critic', 'value']):
                    critic_norm += param_norm.item() ** 2
        
        result = {
            'gradient_norm_total': total_norm ** 0.5,
            'gradient_norm_actor': actor_norm ** 0.5,
            'gradient_norm_critic': critic_norm ** 0.5
        }
        result.update(layer_norms)
        return result
        
    def select_action(self, state, exploration=True):
        # Add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Check for NaN in input
        if torch.isnan(state).any():
            print("Warning: NaN detected in state input")
            state = torch.nan_to_num(state, nan=0.0)
        
        mean, std, _ = self.actor_critic(state)
        
        # Check for NaN in network output
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("Warning: NaN detected in network output")
            mean = torch.nan_to_num(mean, nan=0.0)
            std = torch.nan_to_num(std, nan=1.0)
        
        # Ensure std is positive and add exploration noise
        std = torch.clamp(std, min=1e-6)
        if exploration:
            # Increase exploration noise for better exploration
            std = std + self.exploration_noise * 2.0  # Doubled exploration noise
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample action from unbounded distribution
        unbounded_action = dist.sample()
        
        # Apply tanh to bound actions to [-1, 1]
        action = torch.tanh(unbounded_action)
        
        # Calculate log probability with tanh transformation correction
        log_prob = dist.log_prob(unbounded_action).sum(dim=-1)
        # Correct for tanh transformation: log_prob - log(1 - tanh^2(x))
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).detach().cpu().numpy()
    
    def update(self, states, actions, rewards, log_probs, values, dones, logging=True):
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        values = torch.FloatTensor(np.array(values)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        # Calculate returns
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(10):  # Multiple epochs
            mean, std, value = self.actor_critic(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(value.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            
            # Calculate gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            
            # Log gradient norm to wandb (only on last epoch to avoid spam)
            if logging and epoch == 9:  # Log only on the last epoch
                import wandb
                wandb.log({
                    "gradient_norm": grad_norm.item(),
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "total_loss": loss.item()
                })
            
            self.optimizer.step()


def train(
    record_video: bool = False,
    logging: bool = True,
    initial_joint_angles: list = [0, -60, 120, 30, 0, 0],
    epsilon: float = 0.3,
    gamma: float = 0.99,
    lr: float = 3e-4,
    n_episodes: int | None = 500,
    max_steps: int = 2000,
    batch_size: int = 64,
    waiting_time: float = 1.0,
    exploration_noise: float = 0.7,
    render_every: int = 50,
    joint_names: list = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
    head_position: list = [1, 0, 0.55],
    max_exploration_rate: float = 1,
    min_exploration_rate: float = 0.20,
    decay_rate: float = 0.00005,
    normalize_obs: bool = True
    ):
    """Train the PPO agent"""
    if logging:
        wandb.init(project="head-touch-ppo", name="head-touch-ppo")
        wandb.config.update({
            "epsilon": epsilon,
            "gamma": gamma,
            "lr": lr,
            "exploration_noise": exploration_noise,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "waiting_time": waiting_time,
            "max_exploration_rate": max_exploration_rate,
            "min_exploration_rate": min_exploration_rate,
            "decay_rate": decay_rate,
            "normalize_obs": normalize_obs,
        })

    print("Starting training")
    
    # Initialize environment
    head_touch_env = HeadTouchEnv(env=MuJoCoParserClass(
        name='ur5e_rg2',
        rel_xml_path='../asset/makeup_frida/scene_table.xml',
        verbose=True
    ), record_video=record_video, initial_joint_angles=initial_joint_angles, waiting_time=waiting_time, joint_names=joint_names, head_position=head_position, normalize_obs=normalize_obs)
    
    # Initialize PPO agent
    state_dim = head_touch_env.o_dim
    action_dim = head_touch_env.a_dim
    ppo = PPO(state_dim, action_dim, epsilon=epsilon, gamma=gamma, lr=lr, exploration_noise=exploration_noise)    
    print(f"viewer_initialized: {head_touch_env.viewer_initialized}")
    if logging:
        wandb.watch(ppo.actor_critic, log="all")
    
    # Training loop
    episode_rewards = []
    successful_episodes = []  # Track which episodes were successful
    
    episode = 0
    if n_episodes is None:
        # Infinite training until success
        episode_iterator = iter(lambda: episode, -1)  # This will never end
        print("Training indefinitely until consistent success is achieved...")
    else:
        episode_iterator = trange(n_episodes)
    
    for episode_n in episode_iterator:
        obs = head_touch_env.reset()
        episode_reward = []
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_success = False
        exploration_rate = max_exploration_rate
        
        for step in range(max_steps):
            # Select action with exploration
            action, log_prob = ppo.select_action(obs, exploration=(np.random.random() < exploration_rate))
            
            # Execute action
            next_obs, reward, done, info = head_touch_env.step(action)
            
            if reward > 0.8:  # Adjusted for normalized rewards
                exploration_rate = min_exploration_rate
            else:
                exploration_rate = max(min_exploration_rate, max_exploration_rate * (1 - (episode_n * decay_rate)))
            # Render if video recording (every render_every steps to reduce load)
            if record_video and step % render_every == 0:
                head_touch_env.env.plot_sphere(p=head_touch_env.env.get_p_body('applicator_tip'), r=0.005, rgba=(1,0,1,1), label='Tip')
                head_touch_env.env.plot_sphere(p=head_touch_env.p_target, r=0.005, rgba=(0,0,1,1), label='Target')
                head_touch_env.env.render()
            
            # Store transition
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(ppo.actor_critic.get_value(torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)).item())
            dones.append(done)
            
            obs = next_obs
            episode_reward.append(reward)
            if logging:
                wandb.log({
                    "step": step,
                    "reward": reward,
                    "exploration_rate": exploration_rate,
                    "episode_reward": sum(episode_reward),
                    "episode_success": episode_success,
                    "episode": episode,
                    "max_reward": max(episode_reward)
                })
            
            # End episode if target is reached or max time exceeded
            if done:
                episode_success = True
                torch.save(ppo.actor_critic.state_dict(), f'ppo_model_{episode_n}.pth')
                break
        
        # Track episode success and rewards
        episode_rewards.append(sum(episode_reward))
        successful_episodes.append(episode_success)
        
        # Calculate and log success metrics if logging is enabled
        if logging:
            # Calculate recent success rates
            recent_window = 50
            if len(successful_episodes) >= recent_window:
                recent_success_rate = sum(successful_episodes[-recent_window:]) / recent_window
            else:
                recent_success_rate = sum(successful_episodes) / len(successful_episodes)
            
            # Calculate last 10 episodes success rate
            if len(successful_episodes) >= 10:
                last_10_success_rate = sum(successful_episodes[-10:]) / 10
            else:
                last_10_success_rate = sum(successful_episodes) / len(successful_episodes)
            
            wandb.log({
                "episode_final_reward": sum(episode_reward),
                "episode_success_final": episode_success,
                "episode_number": episode + 1,
                "total_successes": sum(successful_episodes),
                "total_episodes": len(successful_episodes),
                "overall_success_rate": sum(successful_episodes) / len(successful_episodes),
                "recent_50_success_rate": recent_success_rate,
                "last_10_success_rate": last_10_success_rate,
                "exploration_rate_final": exploration_rate
            })
        
        # Update policy more frequently - smaller batch size for more frequent updates
        if len(states) >= batch_size:  # Update with smaller batches
            ppo.update(states, actions, rewards, log_probs, values, dones)
            # Clear buffer after update
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        # Print progress
        print_every = 1
        if (episode + 1) % print_every == 0:
            max_reward = max(episode_reward) if episode_reward else 0
            success_str = "SUCCESS" if episode_success else "FAILED"

            # Calculate recent success rate
            recent_window = 20  # Look at last 20 episodes
            if len(successful_episodes) >= recent_window:
                recent_successes = successful_episodes[-recent_window:]
                success_rate = sum(recent_successes) / len(recent_successes)
                success_rate_str = f"Success Rate: {success_rate:.1%}"
            else:
                success_rate_str = f"Success Rate: {sum(successful_episodes)}/{len(successful_episodes)}"

            if n_episodes is None:
                print(f"Episode {episode + 1}, Steps: {step + 1}, Max Reward: {max_reward:.2f}, {success_str}, Exploration: {exploration_rate:.2f}, {success_rate_str}")
            else:
                print(f"Episode {episode + 1}/{n_episodes}, Steps: {step + 1}, Max Reward: {max_reward:.2f}, {success_str}, Exploration: {exploration_rate:.2f}, {success_rate_str}")
            episode_reward = []
        
        # Enhanced early stopping logic - only stop when consistently successful
        if len(successful_episodes) >= 20:  # Need at least 20 episodes to evaluate
            recent_successes = successful_episodes[-20:]  # Last 20 episodes
            success_rate = sum(recent_successes) / len(recent_successes)
            
            # Stop only if more than half of recent episodes are successful
            if success_rate > 0.5:  # More than 50% success rate
                # Additional check: make sure we have enough consecutive successes
                last_10_successes = successful_episodes[-10:]
                last_10_success_rate = sum(last_10_successes) / len(last_10_successes)
                
                if last_10_success_rate >= 0.6:  # At least 60% success in last 10 episodes
                    print(f"Early stopping at episode {episode + 1} due to consistent success!")
                    print(f"Success rate in last 20 episodes: {success_rate:.1%}")
                    print(f"Success rate in last 10 episodes: {last_10_success_rate:.1%}")
                    break
            
        
        # Increment episode counter for infinite training
        episode += 1
    
    # Save model
    torch.save(ppo.actor_critic.state_dict(), 'ppo_model.pth')
    print("Training completed and model saved!")
    
    # Close viewer if it was opened for video recording
    if record_video and head_touch_env.viewer_initialized:
        head_touch_env.env.close_viewer()
        print("Video recording viewer closed")
    
    return ppo, episode_rewards

def test_model(
    checkpoint_path='ppo_model.pth', 
    record_video=False, 
    n_episodes=5, 
    initial_joint_angles=[0, -60, 90, -60, -90, 0], 
    joint_names=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], 
    render_every=50,
    normalize_obs=True
    ):
    """Test a trained model with optional video recording"""
    print(f"Testing model from {checkpoint_path}")
    
    # Initialize environment
    head_touch_env = HeadTouchEnv(env=MuJoCoParserClass(
        name='ur5e_rg2',
        rel_xml_path='../asset/makeup_frida/scene_table.xml',
        verbose=True
    ), record_video=record_video, initial_joint_angles=initial_joint_angles, joint_names=joint_names, normalize_obs=normalize_obs)
    
    # Load model
    state_dim = head_touch_env.o_dim
    action_dim = head_touch_env.a_dim
    ppo = PPO(state_dim, action_dim)
    
    try:
        ppo.actor_critic.load_state_dict(torch.load(checkpoint_path, map_location=ppo.device))
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Model file {checkpoint_path} not found. Please train first.")
        return
    
    # Video recording setup - ONLY for testing
    if record_video:
        print("Initializing video recording for testing...")
        try:
            head_touch_env.env.init_viewer(
                transparent=False,
                azimuth=105,
                distance=3.12,
                elevation=-29,
                lookat=[0.39, 0.25, 0.43],
                maxgeom=100000
            )
            head_touch_env.viewer_initialized = True
            print("Video recording initialized for testing")
        except Exception as e:
            print(f"Warning: Could not initialize video recording: {e}")
            record_video = False
    
    # Test episodes
    for episode in range(n_episodes):
        obs = head_touch_env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"Starting test episode {episode + 1}")
        
        while step_count < 1000:
            # Get action from model (no exploration during testing)
            try:
                action, _ = ppo.select_action(obs, exploration=False)
            except Exception as e:
                print(f"Error in action selection: {e}")
                break
            
            # Execute action
            try:
                next_obs, reward, done, info = head_touch_env.step(action)
            except Exception as e:
                print(f"Error in environment step: {e}")
                break
            
            # Render if video recording - with error handling
            if record_video and step_count % render_every == 0:
                try:
                    print(f"  Attempting to render at step {step_count}")
                    # Get positions safely
                    tip_pos = head_touch_env.env.get_p_body('applicator_tip')
                    target_pos = head_touch_env.env.get_p_body('target_dot')
                    
                    print(f"  Tip position: {tip_pos}")
                    print(f"  Target position: {target_pos}")
                    
                    # Try to plot spheres with error handling
                    head_touch_env.env.plot_sphere(p=tip_pos, r=0.005, rgba=(1,0,1,1), label='Tip')
                    head_touch_env.env.plot_sphere(p=target_pos, r=0.005, rgba=(0,0,1,1), label='Target')
                    
                    print(f"  Spheres plotted successfully")
                    
                    # Try to render
                    head_touch_env.env.render()
                    print(f"  Render completed successfully")
                    
                except Exception as e:
                    print(f"  Warning: Render failed at step {step_count}: {e}")
                    # Continue without rendering
                    pass
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count}, Distance: {info['dist']:.3f}m, Reward: {episode_reward:.2f}")
            
            if done:
                break
        
        print(f"Test episode {episode + 1} finished with reward: {episode_reward:.2f}")
    
    # Close viewer if it was opened
    if record_video and head_touch_env.viewer_initialized:
        try:
            head_touch_env.env.close_viewer()
            print("Test video recording completed")
        except Exception as e:
            print(f"Warning: Error closing viewer: {e}")

def main(
    test_only: bool = False,
    logging: bool = True,
    record_video: bool = False,
    render_every: int = 50,
    initial_joint_angles: list = [0, -30, 0, 30, 0, 30],
    epsilon: float = 0.1,
    gamma: float = 0.95,
    lr: float = 1e-4,
    batch_size: int = 32,
    max_steps: int = 1000,
    n_episodes: int | None = None,
    waiting_time: float = 1.0,
    exploration_noise: float = 0.6,
    # joint_names: list = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
    joint_names: list = ['elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
    head_position: list = [1, 0.1, 0.55],
    checkpoint_path: str = 'ppo_model.pth',
    normalize_obs: bool = True,
    headless_test: bool = False
    ):
    """Main function to handle training and testing"""
    
    if test_only:
        if headless_test:
            print("Running in headless test mode (no video/rendering)")
            test_model(checkpoint_path=checkpoint_path, record_video=False, initial_joint_angles=initial_joint_angles, joint_names=joint_names, render_every=50, normalize_obs=normalize_obs)
        else:
            test_model(checkpoint_path=checkpoint_path, record_video=True, initial_joint_angles=initial_joint_angles, joint_names=joint_names, render_every=50, normalize_obs=normalize_obs)
    else:
        ppo, rewards = train(
            record_video=record_video,
            logging=logging,
            n_episodes=n_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            initial_joint_angles=initial_joint_angles,
            epsilon=epsilon,
            gamma=gamma,
            lr=lr,
            waiting_time=waiting_time,
            joint_names=joint_names,
            exploration_noise=exploration_noise,
            render_every=render_every,
            head_position=head_position,
            normalize_obs=normalize_obs,
        )
        
        if headless_test:
            print("Running post-training test in headless mode")
            test_model(checkpoint_path=checkpoint_path, record_video=False, n_episodes=3, initial_joint_angles=initial_joint_angles, joint_names=joint_names, render_every=50, normalize_obs=normalize_obs)
        else:
            test_model(checkpoint_path=checkpoint_path, record_video=True, n_episodes=3, initial_joint_angles=initial_joint_angles, joint_names=joint_names, render_every=50, normalize_obs=normalize_obs)



if __name__ == "__main__":
    import fire
    fire.Fire(main)