import numpy as np
from mujoco_parser import *
from gym.head_touch_env import HeadTouchGymClass

def main():
    # Initialize MuJoCo environment
    env = MuJoCoParserClass(
        name='ur5e_rg2',
        rel_xml_path='../asset/ur5e/ur5e_rg2.xml',
        VERBOSE=True
    )
    
    # Initialize head touch environment
    head_touch_env = HeadTouchGymClass(env=env)
    
    # Run a few episodes with random actions
    n_episodes = 3
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        # Reset environment
        obs = head_touch_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Generate random action
            action = np.random.uniform(-1, 1, size=7)
            
            # Step environment
            obs, reward, done, info = head_touch_env.step(action)
            total_reward += reward
            
            # Render
            head_touch_env.render()
            
            # Print info
            print(f"Distance to target: {info['dist']:.3f}m")
        
        print(f"Episode finished with total reward: {total_reward:.2f}")

if __name__ == "__main__":
    main() 