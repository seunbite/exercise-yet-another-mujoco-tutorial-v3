import os, sys, time, random, glob
import numpy as np
import torch

# Append paths for custom packages
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/gym/')
sys.path.append('../package/rl/')

from mujoco_parser import MuJoCoParserClass
from utility import np2torch, torch2np
from sac import ReplayBufferClass, ActorClass, CriticClass, get_target
from rl_ur5e_line import UR5eHeadTouchEnv  # reuse original env

class UR5eLineWiseEnv(UR5eHeadTouchEnv):
    """Environment whose reward is the similarity between the *whole trajectory* that the applicator tip draws in an episode and the target line (start–end)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_path = []  # stores tip positions for current episode

    # ------------------------------------------------------------
    def reset(self):
        self._episode_path = []
        return super().reset()

    # ------------------------------------------------------------
    def step(self, action, max_time=np.inf):
        obs_next, _, done, info = super().step(action, max_time)
        # store tip position at every tick
        self._episode_path.append(self.get_applicator_tip_pos().copy())
        # override reward: only give at the final step
        if done:
            reward = self._compute_line_reward()
        else:
            reward = 0.0
        info['line_reward'] = reward
        return obs_next, reward, done, info

    # ------------------------------------------------------------
    def _compute_line_reward(self):
        """Compute negative mean squared distance between drawn path and goal line."""
        if len(self._episode_path) == 0:
            return -1.0  # trivial
        path = np.array(self._episode_path)
        start, end = self.target_line
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-9:
            return -1.0
        line_dir = line_vec / line_len
        # project every path point onto the line, compute perpendicular distance
        rel = path - start  # [N,3]
        proj = np.dot(rel, line_dir)  # [N]
        nearest = np.outer(proj, line_dir) + start  # [N,3]
        dists = np.linalg.norm(path - nearest, axis=1)
        mse = np.mean(dists**2)
        # Reward: +10 when mse==0, decreases with distance
        reward = 10.0 * np.exp(-mse / (0.02**2)) - 1.0  # in [-1, 10]
        return reward

# ------------------------------------------------------------

def train_linewise(
        xml_path='../asset/makeup_frida/scene_table.xml',
        n_episode=500,
        max_epi_sec=5.0,
        seed=0,
    ):
    env = MuJoCoParserClass(name='UR5e_LineWise', rel_xml_path=xml_path, verbose=False)
    gym = UR5eLineWiseEnv(env, HZ=50, reward_mode='1', dynamic_target=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    replay = ReplayBufferClass(50000, device)

    actor = ActorClass(obs_dim=gym.o_dim, h_dims=[256,256], out_dim=gym.a_dim,
                       max_out=1.0, init_alpha=0.2, lr_actor=5e-4,
                       lr_alpha=1e-4, device=device).to(device)
    critic1 = CriticClass(obs_dim=gym.o_dim, a_dim=gym.a_dim, h_dims=[256,256],
                          out_dim=1, lr_critic=1e-4, device=device).to(device)
    critic2 = CriticClass(obs_dim=gym.o_dim, a_dim=gym.a_dim, h_dims=[256,256],
                          out_dim=1, lr_critic=1e-4, device=device).to(device)
    c1_tgt = CriticClass(obs_dim=gym.o_dim, a_dim=gym.a_dim, h_dims=[256,256],
                         out_dim=1, lr_critic=1e-4, device=device).to(device)
    c2_tgt = CriticClass(obs_dim=gym.o_dim, a_dim=gym.a_dim, h_dims=[256,256],
                         out_dim=1, lr_critic=1e-4, device=device).to(device)
    c1_tgt.load_state_dict(critic1.state_dict())
    c2_tgt.load_state_dict(critic2.state_dict())

    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

    max_tick = int(max_epi_sec* gym.HZ)
    warmup = 1000

    for epi in range(n_episode):
        s = gym.reset()
        for t in range(max_tick):
            if replay.size() < warmup:
                a = gym.sample_action()
            else:
                a,_ = actor(np2torch(s,device=device))
                a = torch2np(a)
            s2, r, done, info = gym.step(a)
            replay.put((s,a,r,s2,done))
            s = s2
            if done:
                break
            # update networks after warmup
            if replay.size() > warmup:
                batch = replay.sample(256)
                td_target = get_target(actor,c1_tgt,c2_tgt,0.99,batch,device)
                critic1.train(td_target,batch)
                critic2.train(td_target,batch)
                actor.train(critic1,critic2,-gym.a_dim,batch)
                critic1.soft_update(0.005,c1_tgt)
                critic2.soft_update(0.005,c2_tgt)
        print(f"Episode {epi}: reward {info['line_reward']:.3f}, steps {t}")

if __name__ == '__main__':
    import fire
    fire.Fire(train_linewise)