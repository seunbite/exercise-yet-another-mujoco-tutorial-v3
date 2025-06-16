import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from mujoco_parser import *
from gym.head_touch_env import HeadTouchGymClass

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and log_std for each action
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        # Get action distribution parameters
        action_params = self.actor(state)
        mean = action_params[:, :action_params.shape[1]//2]
        log_std = action_params[:, action_params.shape[1]//2:]
        std = torch.exp(log_std)
        
        # Get state value
        value = self.critic(state)
        
        return mean, std, value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, std, _ = self.actor_critic(state)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample action
        action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.cpu().numpy(), log_prob.cpu().numpy()
    
    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate returns and advantages
        with torch.no_grad():
            _, _, values = self.actor_critic(states)
            _, _, next_values = self.actor_critic(next_states)
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            mean, std, _ = self.actor_critic(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            _, _, values = self.actor_critic(states)
            critic_loss = nn.MSELoss()(values, returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train():
    # Initialize environment
    env = MuJoCoParserClass(
        name='ur5e_rg2',
        rel_xml_path='../asset/ur5e/ur5e_rg2.xml',
        VERBOSE=True
    )
    head_touch_env = HeadTouchGymClass(env=env)
    
    # Initialize PPO
    state_dim = head_touch_env.o_dim
    action_dim = head_touch_env.a_dim
    ppo = PPO(state_dim, action_dim)
    
    # Training parameters
    n_episodes = 1000
    max_steps = 200
    batch_size = 64
    
    # Training loop
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = head_touch_env.reset()
        episode_reward = 0
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        next_states = []
        dones = []
        
        for step in range(max_steps):
            # Select action
            action, log_prob = ppo.select_action(state)
            
            # Execute action
            next_state, reward, done, _ = head_touch_env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            # Update if batch is full
            if len(states) >= batch_size:
                ppo.update(
                    np.array(states),
                    np.array(actions),
                    np.array(log_probs),
                    np.array(rewards),
                    np.array(next_states),
                    np.array(dones)
                )
                states = []
                actions = []
                log_probs = []
                rewards = []
                next_states = []
                dones = []
            
            if done:
                break
        
        # Update with remaining transitions
        if len(states) > 0:
            ppo.update(
                np.array(states),
                np.array(actions),
                np.array(log_probs),
                np.array(rewards),
                np.array(next_states),
                np.array(dones)
            )
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
            
            # Plot rewards
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig('training_rewards.png')
            plt.close()
    
    # Save model
    torch.save(ppo.actor_critic.state_dict(), 'head_touch_model.pth')
    print("Training completed!")

if __name__ == "__main__":
    train()