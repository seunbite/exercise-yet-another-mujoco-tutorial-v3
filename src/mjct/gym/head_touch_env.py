import numpy as np
from mujoco_parser import *
from utility import *

class HeadTouchGymClass:
    def __init__(
        self,
        env,
        HZ = 50,
        VERBOSE = True
    ):
        self.env = env
        self.HZ = HZ
        self.VERBOSE = VERBOSE
        
        # State and action dimensions
        self.o_dim = 3  # RGB image size (will be flattened)
        self.a_dim = 7  # 6 joint angles + 1 gripper
        
        # Target dot position range (relative to head)
        self.target_x_range = [-0.1, 0.1]
        self.target_y_range = [-0.1, 0.1]
        self.target_z_range = [0.05, 0.2]
        
        # Success threshold
        self.success_threshold = 0.02  # 2cm
        
        if self.VERBOSE:
            print(f"[HeadTouch] Instantiated")
            print(f"   [info] dt:[{1.0/self.HZ:.4f}] HZ:[{self.HZ}], state_dim:[{self.o_dim}], a_dim:[{self.a_dim}]")
    
    def reset(self):
        """Reset environment and get initial observation"""
        self.env.reset()
        
        # Randomize target dot position
        target_x = np.random.uniform(*self.target_x_range)
        target_y = np.random.uniform(*self.target_y_range)
        target_z = np.random.uniform(*self.target_z_range)
        self.env.set_p_body('target_dot', np.array([target_x, target_y, target_z]))
        
        # Get initial observation (camera view)
        p_cam, R_cam = self.env.get_pR_body('ur_camera_center')
        p_ego = p_cam
        p_trgt = p_cam + R_cam[:,2]
        rgb_img, _, _, _, _ = self.env.get_egocentric_rgbd_pcd(
            p_ego=p_ego,
            p_trgt=p_trgt,
            rsz_rate_for_img=1/4,
            fovy=45
        )
        
        return rgb_img.flatten()  # Flatten the image as state
    
    def step(self, action, max_time=5.0):
        """Execute action and get next observation"""
        # Apply action to robot joints
        joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                      'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        idxs_step = self.env.get_idxs_step(joint_names=joint_names)
        self.env.step(ctrl=action, ctrl_idxs=idxs_step)
        
        # Get new observation
        p_cam, R_cam = self.env.get_pR_body('ur_camera_center')
        p_ego = p_cam
        p_trgt = p_cam + R_cam[:,2]
        rgb_img, _, _, _, _ = self.env.get_egocentric_rgbd_pcd(
            p_ego=p_ego,
            p_trgt=p_trgt,
            rsz_rate_for_img=1/4,
            fovy=45
        )
        
        # Calculate reward
        p_tip = self.env.get_p_body('applicator')  # Get applicator tip position
        p_target = self.env.get_p_body('target_dot')  # Get target position
        dist = np.linalg.norm(p_tip - p_target)
        
        # Reward is 1 if target is reached, -0.1 otherwise
        reward = 1.0 if dist < self.success_threshold else -0.1
        
        # Episode is done if target is reached or max time exceeded
        done = (dist < self.success_threshold) or (self.env.get_sim_time() > max_time)
        
        info = {
            'dist': dist,
            'p_tip': p_tip,
            'p_target': p_target
        }
        
        return rgb_img.flatten(), reward, done, info
    
    def render(self):
        """Render the current state"""
        self.env.render() 