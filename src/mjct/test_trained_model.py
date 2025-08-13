import torch
import numpy as np
from mujoco_parser import *
from gym.head_touch_env import HeadTouchGymClass
from train_head_touch import ActorCritic

def test_model(model_path, n_episodes=5):
    # Initialize environment
    env = MuJoCoParserClass(
        name='ur5e_rg2',
        rel_xml_path='../asset/ur5e/ur5e_rg2.xml',
        VERBOSE=True
    )
    head_touch_env = HeadTouchGymClass(env=env)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(head_touch_env.o_dim, head_touch_env.a_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    for episode in range(n_episodes):
        state = head_touch_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                mean, std, _ = model(state_tensor)
                action = mean.cpu().numpy()
            
            # Execute action
            next_state, reward, done, info = head_touch_env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Print progress
            print(f"Distance to target: {info['dist']:.3f}m")
            
            # Render
            head_touch_env.render()
        
        print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")

if __name__ == "__main__":
    test_model('head_touch_model.pth') 