import sys,mujoco,time,os,json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import torch.nn.functional as F

sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/gpt_usage/')
sys.path.append('../package/detection_module/')
from mujoco_parser import *
from utility import *
from transformation import *
from gpt_helper import *
from owlv2 import *

# Environment class
class HeadTouchEnv:
    def __init__(self, env, HZ=50, record_video=False, initial_joint_angles=[0, -60, 90, -60, -90, 0], waiting_time=0.0, joint_names=None):
        self.env = env
        self.HZ = HZ
        self.record_video = record_video
        self.frames = []  # Store frames for video
        
        # All joint names for the UR5e robot
        self.all_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # Controllable joints (subset of all joints)
        self.joint_names = joint_names if joint_names is not None else self.all_joint_names
        
        # Initial angles for ALL joints (must be 6 values)
        if len(initial_joint_angles) != 6:
            raise ValueError("initial_joint_angles must have exactly 6 values for all UR5e joints")
        self.initial_joint_angles = np.deg2rad(initial_joint_angles)
        
        self.waiting_time = waiting_time

        # Setup environment like in the notebook
        self._setup_environment()
        
        # Initialize viewer for recording if needed
        if self.record_video:
            try:
                # Try to initialize viewer normally
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
            
            # Initialize the backup image
            self.env.grab_image_backup = np.zeros((400, 600, 3), dtype=np.uint8)

        else:
            self.viewer_initialized = False
        
        # Get observation dimensions (headless version)
        test_obs = self._get_observation()
        self.o_dim = test_obs.shape[0]  # joint pos + joint vel + tip pos + target pos
        self.a_dim = len(self.joint_names)  # Only controllable joints
        
        # Success threshold
        self.success_threshold = 0.02  # 2cm
        
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
    
    def _setup_environment(self):
        """Setup environment with tables, objects, and initial positioning"""
        # Reset environment
        self.env.reset()
        
        # Move UR5e base to proper position
        self.env.set_p_body(body_name='ur_base', p=np.array([0, 0, 0.5]))
        
        # Move table to proper position
        self.env.set_p_body(body_name='object_table', p=np.array([1.0, 0, 0]))
        
        # Setup objects (head) - no rotation to keep it stable
        self.env.set_p_base_body(body_name='obj_head', p=[1, 0, 0.55])
        
        # Set initial joint positions for ALL joints
        self.env.set_qpos_joints(joint_names=self.all_joint_names, qpos=self.initial_joint_angles)
        print(f"Setting initial joint angles (degrees): {np.rad2deg(self.initial_joint_angles)}")
        
        # Store initial configuration for reset
        self.initial_qpos = self.initial_joint_angles
    
    def _get_observation(self):
        """Get current observation (headless version using joint positions)"""
        # Get joint positions for the controllable joints only
        joint_positions = self.env.get_qpos_joints(joint_names=self.joint_names)
        
        # Get joint velocities for controllable joints only
        joint_velocities = self.env.get_qvel_joints(joint_names=self.joint_names)
        
        # Get end-effector position
        p_tip = self.env.get_p_body('applicator_tip')
        
        # Get target position
        p_target = self.env.get_p_body('target_dot')
        
        # Combine all observations
        obs = np.concatenate([
            joint_positions,      # N values (N = number of controllable joints)
            joint_velocities,     # N values  
            p_tip,               # 3 values
            p_target             # 3 values
        ])
        
        return obs
    
    def reset(self, do_randomization: bool = False):
        """Reset environment to initial state"""
        # Reset MuJoCo environment
        self.env.reset(step=True)
        
        # Reset exploration state
        if hasattr(self, 'prev_tip_pos'):
            delattr(self, 'prev_tip_pos')
        
        # Re-setup environment positions
        self.env.set_p_body(body_name='ur_base', p=np.array([0, 0, 0.5]))
        self.env.set_p_body(body_name='object_table', p=np.array([1.0, 0, 0]))
        
        # Reset objects to initial positions - keep stable
        self.env.set_p_base_body(body_name='obj_head', p=[1, 0, 0.55])
        self.env.set_R_base_body(body_name='obj_head', R=rpy2r(np.radians([0, 270, 0])))
        
        # Set robot to initial joint configuration with optional randomization
        if do_randomization:
            # Add small random noise to initial positions for exploration
            # But only for controllable joints, others stay fixed
            initial_all_joints = self.initial_qpos.copy()
            
            # Get indices of controllable joints
            controllable_indices = [self.all_joint_names.index(name) for name in self.joint_names]
            
            # Add noise only to controllable joints
            for idx in controllable_indices:
                initial_all_joints[idx] += np.random.uniform(-0.1, 0.1)
            
            self.env.set_qpos_joints(joint_names=self.all_joint_names, qpos=initial_all_joints)
        else:
            self.env.set_qpos_joints(joint_names=self.all_joint_names, qpos=self.initial_qpos)
        
        # Fast physics settling without rendering (more efficient)
        print(f"Fast settling for {self.waiting_time} seconds...")
        
        # Run physics at full speed without rendering
        wait_steps = int(self.waiting_time / self.env.dt)
        for _ in range(wait_steps):
            self.env.step()
        
        # Target dot position is fixed in XML file - no randomization needed
        # The target_dot is already positioned at (0, 0, 0.3) relative to obj_head in the XML
        
        return self._get_observation()
    
    def step(self, action, max_time=60.0):
        """Execute action and get next observation"""
        # Create full control vector for all joints
        full_ctrl = np.zeros(6)  # 6 joints total
        
        # Set fixed joints to their initial positions (hold them fixed)
        for i, joint_name in enumerate(self.all_joint_names):
            if joint_name not in self.joint_names:
                # This is a fixed joint - hold it at initial position
                current_pos = self.env.get_qpos_joints(joint_names=[joint_name])
                target_pos = self.initial_joint_angles[i]
                # Use PD control to hold fixed joints at their initial positions
                full_ctrl[i] = 10.0 * (target_pos - current_pos)  # Strong position control
        
        # Set controllable joints to the provided actions
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
        p_target = self.env.get_p_body('target_dot')  # Get target position
        
        dist = np.linalg.norm(p_tip - p_target)
        
        # Improved reward function with better learning signals
        if dist < self.success_threshold:
            # Success reward
            reward = 100.0  # Much higher success reward
        else:
            # Distance-based reward (closer = better)
            distance_reward = -dist * 2.0  # Stronger distance penalty
            
            # Progress reward - reward for getting closer to target
            progress_reward = 0.0
            if hasattr(self, 'prev_dist'):
                progress = self.prev_dist - dist  # Positive if getting closer
                progress_reward = progress * 10.0  # Reward for progress
            self.prev_dist = dist
            
            # Movement reward - encourage exploration
            movement_reward = 0.0
            if hasattr(self, 'prev_tip_pos'):
                movement = np.linalg.norm(p_tip - self.prev_tip_pos)
                movement_reward = movement * 5.0  # Reward for movement
            self.prev_tip_pos = p_tip.copy()
            
            # Combine rewards
            reward = distance_reward + progress_reward + movement_reward
            
            # Small penalty for time to encourage efficiency
            reward -= 0.01
        
        # Episode is done if target is reached or max time exceeded
        done = (dist < self.success_threshold) or (self.env.get_sim_time() > max_time)
        
        info = {
            'dist': dist,
            'p_tip': p_tip,
            'p_target': p_target
        }
        
        return obs, reward, done, info
    
                    
    def render(self):
        """Render the current state"""
        if hasattr(self, 'viewer_initialized') and self.viewer_initialized:
            try:
                self.env.render()
            except Exception as e:
                print(f"Warning: Render failed: {e}")
    
    def capture_frame(self):
        """Capture current frame for video recording"""
        if self.record_video:
            try:
                if hasattr(self, 'viewer_initialized') and self.viewer_initialized:
                    # Use MuJoCo viewer if available
                    self.env.render()
                    img = self.env.grab_image(rsz_rate=0.5)
                else:
                    # Create a simple visualization frame
                    img = self._create_visualization_frame()
                
                if img is not None and img.sum() > 0:
                    self.frames.append(img.copy())
                else:
                    # Use backup image if current image is invalid
                    self.frames.append(self.env.grab_image_backup.copy())
            except Exception as e:
                print(f"Warning: Failed to capture frame: {e}")
                # Create a blank frame as fallback
                blank_frame = np.zeros((400, 600, 3), dtype=np.uint8)
                self.frames.append(blank_frame)
    
    def _create_visualization_frame(self):
        """Create a visualization frame for headless rendering"""
        try:
            # Create a simple visualization using matplotlib
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Get current positions
            p_tip = self.env.get_p_body('applicator_tip')
            p_target = self.env.get_p_body('target_dot')
            p_head = self.env.get_p_body('obj_head')
            
            # Plot positions
            ax.scatter([p_tip[0]], [p_tip[1]], c='red', s=100, label='Robot Tip')
            ax.scatter([p_target[0]], [p_target[1]], c='green', s=100, label='Target')
            ax.scatter([p_head[0]], [p_head[1]], c='blue', s=100, label='Head')
            
            # Draw line from tip to target
            ax.plot([p_tip[0], p_target[0]], [p_tip[1], p_target[1]], 'k--', alpha=0.5)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Robot Environment (Top View)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Convert matplotlib figure to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            plt.close()
            
            # Convert RGBA to RGB if needed
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            return img_array
            
        except Exception as e:
            print(f"Warning: Failed to create visualization frame: {e}")
            return None
    
    def save_video(self, filename='training_video.mp4', fps=10):
        """Save recorded frames as video"""
        if not self.record_video or len(self.frames) == 0:
            print("No frames to save")
            return
        
        print(f"Saving video with {len(self.frames)} frames...")
        
        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # Write frames
        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved as {filename}")
        
        # Clear frames to free memory
        self.frames = []

# PPO implementation
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) head
        self.critic = nn.Linear(256, 1)
        
    def forward(self, state):
        shared_features = self.shared(state)
        
        # Actor
        mean = self.actor_mean(shared_features)
        std = torch.exp(self.actor_logstd)
        
        # Critic
        value = self.critic(shared_features)
        
        return mean, std, value
    
    def get_value(self, state):
        """Get only the value estimate"""
        shared_features = self.shared(state)
        return self.critic(shared_features)

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Show which device is being used
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-8)
        self.gamma = gamma
        self.epsilon = epsilon
        self.exploration_noise = 1  # Add exploration noise
        
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
        
        # Sample action
        action = dist.sample()
        
        # Clip actions to reasonable range
        action = torch.clamp(action, -1.0, 1.0)
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).detach().cpu().numpy()
    
    def update(self, states, actions, rewards, log_probs, values, dones):
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
        for _ in range(10):  # Multiple epochs
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
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()


def train(
    record_video: bool = False,
    initial_joint_angles: list = [0, -60, 90, -60, -90, 0],
    epsilon: float = 0.3,
    gamma: float = 0.99,
    lr: float = 3e-4,
    n_episodes: int = 500,
    max_steps: int = 2000,
    batch_size: int = 64,
    waiting_time: float = 1.0,
    joint_names: list = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    ):
    """Train the PPO agent"""
    print("Starting training")
    
    # Initialize environment
    head_touch_env = HeadTouchEnv(env=MuJoCoParserClass(
        name='ur5e_rg2',
        rel_xml_path='../asset/makeup_frida/scene_table.xml',
        verbose=True
    ), record_video=record_video, initial_joint_angles=initial_joint_angles, waiting_time=waiting_time, joint_names=joint_names)
    
    # Initialize PPO agent
    state_dim = head_touch_env.o_dim
    action_dim = head_touch_env.a_dim
    ppo = PPO(state_dim, action_dim, epsilon=epsilon, gamma=gamma, lr=lr)    
    print(f"viewer_initialized: {head_touch_env.viewer_initialized}")
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = head_touch_env.reset()
        episode_reward = 0
        episode_length = 0
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        # Add exploration decay - slower decay for better exploration
        exploration_rate = max(0.3, 1.0 - episode / (n_episodes * 0.8))  # Slower decay, higher minimum
        
        for step in range(max_steps):
            # Select action with exploration
            action, log_prob = ppo.select_action(obs, exploration=(np.random.random() < exploration_rate))
            
            # Execute action
            next_obs, reward, done, info = head_touch_env.step(action)
            
            # Render if video recording (every 10 steps to reduce load)
            if record_video and step % 50 == 0:
                head_touch_env.env.plot_sphere(p=head_touch_env.env.get_p_body('applicator_tip'), r=0.005, rgba=(1,0,1,1), label='Tip')
                head_touch_env.env.plot_sphere(p=head_touch_env.env.get_p_body('target_dot'), r=0.005, rgba=(0,0,1,1), label='Target')
                head_touch_env.env.render()
            
            # Store transition
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(ppo.actor_critic.get_value(torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)).item())
            dones.append(done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # End episode if target is reached or time limit exceeded
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update policy more frequently - smaller batch size for more frequent updates
        if len(states) >= batch_size // 2:  # Update with smaller batches
            ppo.update(states, actions, rewards, log_probs, values, dones)
            # Clear buffer after update
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, Exploration: {exploration_rate:.2f}")
        
        # Early stopping if consistently successful
        if len(episode_rewards) >= 50:
            recent_rewards = episode_rewards[-50:]
            if np.mean(recent_rewards) > 5.0 and np.std(recent_rewards) < 2.0:
                print(f"Early stopping at episode {episode + 1} due to consistent success")
                break
    
    # Save model
    torch.save(ppo.actor_critic.state_dict(), 'ppo_model.pth')
    print("Training completed and model saved!")
    
    # Close viewer if it was opened for video recording
    if record_video and head_touch_env.viewer_initialized:
        head_touch_env.env.close_viewer()
        print("Video recording viewer closed")
    
    return ppo, episode_rewards

def test_model(model_path='ppo_model.pth', record_video=False, n_episodes=5, initial_joint_angles=[0, -60, 90, -60, -90, 0], joint_names=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']):
    """Test a trained model with optional video recording"""
    print(f"Testing model from {model_path}")
    
    # Initialize environment
    head_touch_env = HeadTouchEnv(env=MuJoCoParserClass(
        name='ur5e_rg2',
        rel_xml_path='../asset/makeup_frida/scene_table.xml',
        verbose=True
    ), record_video=record_video, initial_joint_angles=initial_joint_angles, joint_names=joint_names)
    
    # Load model
    state_dim = head_touch_env.o_dim
    action_dim = head_touch_env.a_dim
    ppo = PPO(state_dim, action_dim)
    
    try:
        ppo.actor_critic.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train first.")
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
        obs = head_touch_env.reset(do_randomization=False)
        episode_reward = 0
        step_count = 0
        
        print(f"Starting test episode {episode + 1}")
        
        while step_count < 1000:
            # Get action from model (no exploration during testing)
            action, _ = ppo.select_action(obs, exploration=False)
            
            # Execute action
            next_obs, reward, done, info = head_touch_env.step(action)
            
            # Render if video recording
            if record_video and step_count % 5 == 0:
                head_touch_env.env.render()
            
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
        head_touch_env.env.close_viewer()
        print("Test video recording completed")

def main(
    record_video: bool = True,
    test_only: bool = False,
    initial_joint_angles: list = [0, -60, 90, -60, -90, 0],
    epsilon: float = 0.3,
    gamma: float = 0.99,
    lr: float = 3e-4,
    batch_size: int = 32,
    max_steps: int = 2000,
    n_episodes: int = 200,
    waiting_time: float = 1.0,
    joint_names: list = ['wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    ):
    """Main function to handle training and testing"""
    
    if test_only:
        test_model(record_video=record_video, initial_joint_angles=initial_joint_angles, joint_names=joint_names)
    else:
        # Training - NO GUI
        ppo, rewards = train(
            record_video=record_video,
            n_episodes=n_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            initial_joint_angles=initial_joint_angles,
            epsilon=epsilon,
            gamma=gamma,
            lr=lr,
            waiting_time=waiting_time,
            joint_names=joint_names,
        )
        
        # Test with GUI if video mode or record_video is True
        test_model(record_video=True, n_episodes=3, initial_joint_angles=initial_joint_angles, joint_names=joint_names)

if __name__ == "__main__":
    import fire
    fire.Fire(main)