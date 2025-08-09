import numpy as np
import mujoco
import time
import matplotlib.pyplot as plt
from package.mujoco_usage.mujoco_parser import MuJoCoParserClass
from package.helper.transformation import r2rpy, rpy2r, pr2t, t2p, t2r
from package.helper.utility import get_colors, d2r, trim_scale, np2torch, torch2np


class UR5eIKEnv:
    def __init__(
            self,
            xml_path='asset/makeup_frida/scene_table.xml',
            HZ=50,
            object_table_position=[1.0, 0, 0], 
            base_table_position=[0, 0, 0],
            head_position=[1, 0.0, 0.53], 
            waiting_time=0.0, 
            ):
        self.HZ = HZ
        self.dt = 1/self.HZ
        self.waiting_time = waiting_time
        self.head_position = np.array(head_position)
        self.object_table_position = np.array(object_table_position)
        self.base_table_position = np.array(base_table_position)
        
        # Initialize MuJoCo environment
        self.env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path)
        
        # Joint names for UR5e
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # Check available joints
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
        
        # Initialize viewer
        self.env.init_viewer(title='UR5e IK Demo', width=1400, height=1000)
        
        # Reset environment
        self.reset()
    
    def _make_line(self, start_point, direction, length):
        """Create a line segment from start_point in the given direction"""
        end_point = start_point + direction * length
        return np.array([start_point, end_point])
    
    def _generate_uniform_points(self, start_point, end_point, num_points=50):
        """Generate uniform points along the line from start to end"""
        # Create uniform parameter t from 0 to 1
        t_values = np.linspace(0, 1, num_points)
        
        # Interpolate points along the line
        points = []
        for t in t_values:
            point = start_point + t * (end_point - start_point)
            points.append(point)
        
        return np.array(points)
    
    def solve_ik_for_point(self, target_point, max_iterations=100, tolerance=1e-3):
        """Solve inverse kinematics for a single target point"""
        # Get current joint positions
        joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
        current_q = self.env.get_qpos_joints(self.joint_names)
        
        # Use the environment's IK solver (position only, not rotation)
        success = self.env.onestep_ik(
            body_name='applicator_tip',
            p_trgt=target_point,
            IK_P=True,
            IK_R=False,  # Don't solve for rotation
            joint_idxs=joint_idxs,
            stepsize=0.1,
            eps=tolerance,
            th=np.radians(1.0)
        )
        
        if success:
            return self.env.get_qpos_joints(self.joint_names)
        else:
            return None
    
    def solve_ik_trajectory(self, start_point, end_point, num_points=50):
        """Solve inverse kinematics for a trajectory of uniform points"""
        # Generate uniform points along the line
        trajectory_points = self._generate_uniform_points(start_point, end_point, num_points)
        
        # Solve IK for each point
        joint_trajectories = []
        successful_points = []
        
        print(f"Solving IK for {num_points} points along the trajectory...")
        
        for i, point in enumerate(trajectory_points):
            print(f"Solving IK for point {i+1}/{num_points}: {point}")
            
            # Solve IK for this point
            joint_config = self.solve_ik_for_point(point)
            
            if joint_config is not None:
                joint_trajectories.append(joint_config)
                successful_points.append(point)
                print(f"  ✓ IK solved successfully")
            else:
                print(f"  ✗ IK failed for this point")
        
        print(f"Successfully solved IK for {len(successful_points)}/{num_points} points")
        
        return np.array(joint_trajectories), np.array(successful_points)
    
    def render_target_line(self, start_point, end_point):
        """Render the target line with start and end points"""
        if not hasattr(self.env, 'viewer') or not self.env.use_mujoco_viewer:
            return
        
        # Plot start point (green sphere)
        self.env.plot_sphere(p=start_point, r=0.008, rgba=[0, 1, 0, 0.8], label='Start')
        
        # Plot end point (blue sphere)
        self.env.plot_sphere(p=end_point, r=0.008, rgba=[0, 0, 1, 0.8], label='End')
        
        # Plot line connecting start and end points
        self.env.plot_line_fr2to(p_fr=start_point, p_to=end_point, rgba=[0.5, 0.5, 0.5, 0.6])
        
        # Render the scene
        self.env.render()
        
        if hasattr(self.env, 'viewer') and self.env.viewer:
            try:
                self.env.viewer.sync()
            except:
                pass
    
    def animate_trajectory(self, joint_trajectory, trajectory_points, 
                          animation_speed=0.1, render_every=1):
        """Animate the robot following the solved trajectory"""
        print(f"Animating trajectory with {len(joint_trajectory)} waypoints...")
        
        for i, (joint_config, target_point) in enumerate(zip(joint_trajectory, trajectory_points)):
            if i % render_every == 0:
                # Set joint positions
                joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
                self.env.forward(q=joint_config, joint_idxs=joint_idxs)
                
                # Get current end-effector position
                current_tip_pos = self.env.get_p_body('applicator_tip')
                
                # Render
                self.env.render()
                
                # Plot current target point (red sphere)
                self.env.plot_sphere(p=target_point, r=0.008, rgba=[1, 0, 0, 0.8], label=f'Target')
                
                self.env.plot_line_fr2to(p_fr=current_tip_pos, p_to=target_point, rgba=[1, 0.5, 0, 0.8])
                
                # Small delay for visualization
                time.sleep(animation_speed)
        
        print("Animation completed!")
    
    def reset(self):
        """Reset the environment to initial state"""
        self.env.reset(step=True)
        
        # Set initial positions
        self.env.set_p_body(body_name='ur_base', p=[0, 0, 0.5])
        self.env.set_p_body(body_name='base_table', p=self.base_table_position)
        self.env.set_p_body(body_name='object_table', p=self.object_table_position)
        
        # Set initial joint configuration
        try:
            joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
            init_qpos = np.array([-30, -60, 90, -30, -30, 0]) * np.pi / 180
            if len(init_qpos) != len(self.joint_names):
                init_qpos = np.zeros(len(self.joint_names))
            self.env.forward(q=init_qpos, joint_idxs=joint_idxs)
        except Exception as e:
            print(f"Warning: Could not set initial configuration: {e}")
        
        # Wait if specified
        if self.waiting_time > 0:
            for _ in range(int(self.waiting_time * self.HZ)):
                self.env.step(ctrl=np.zeros(len(self.joint_names)), nstep=1)
    
    def create_target_line(self, use_dynamic_target=False):
        """Create a target line (start and end points)"""
        if use_dynamic_target:
            # Dynamic target positioning - harder, more varied
            target_pos_start = self.head_position + np.array([
                np.random.uniform(0.03, 0.03), 
                np.random.uniform(0.0, 0.03), 
                np.random.uniform(0.15, 0.18)
            ])
            line_length = np.random.uniform(0.1, 0.13)
            z_offset = np.random.uniform(0.05, 0.10) * np.random.choice([-1, 1])
            line_direction = np.array([
                np.random.choice([-1, 1]), 
                np.random.uniform(-0.5, 0.5), 
                z_offset / line_length
            ])
            line_direction = line_direction / np.linalg.norm(line_direction)
        else:
            # Static target positioning - easier, consistent
            target_pos_start = self.head_position + np.array([0.03, 0.015, 0.165])
            line_length = 0.125
            line_direction = np.array([1, 0, 0])  # Fixed direction (along x-axis)
        
        target_line = self._make_line(target_pos_start, direction=line_direction, length=line_length)
        return target_line
    
    def run_ik_demo(self, use_dynamic_target=False, num_points=50, animate=True):
        """Run the complete IK demonstration"""
        print("=== UR5e Inverse Kinematics Demo ===")
        
        # Create target line
        target_line = self.create_target_line(use_dynamic_target)
        start_point = target_line[0]
        end_point = target_line[1]
        
        print(f"Target line: Start={start_point}, End={end_point}")
        
        # Render the target line with start and end points
        self.render_target_line(start_point, end_point)
        
        # Solve IK for trajectory
        joint_trajectory, trajectory_points = self.solve_ik_trajectory(
            start_point, end_point, num_points
        )
        
        if len(joint_trajectory) > 0:
            print(f"Successfully generated trajectory with {len(joint_trajectory)} waypoints")
            
            if animate:
                # Animate the trajectory
                self.animate_trajectory(joint_trajectory, trajectory_points)
            
            return joint_trajectory, trajectory_points
        else:
            print("Failed to solve IK for any points along the trajectory")
            return None, None
    
    def close(self):
        """Close the environment"""
        self.env.close_viewer()


def main():
    """Main function to run the IK demo"""
    # Create IK environment
    ik_env = UR5eIKEnv(
        xml_path='asset/makeup_frida/scene_table.xml',
        HZ=50,
        object_table_position=[1.0, 0, 0],
        base_table_position=[0, 0, 0],
        head_position=[1, 0.0, 0.53],
        waiting_time=0.0
    )
    
    try:
        # Run IK demo with static target
        print("\n=== Running IK Demo with Static Target ===")
        joint_traj_static, points_static = ik_env.run_ik_demo(
            use_dynamic_target=False, 
            num_points=50, 
            animate=True
        )
        
        # Wait a bit before next demo
        time.sleep(2)
        
        # Run IK demo with dynamic target
        print("\n=== Running IK Demo with Dynamic Target ===")
        joint_traj_dynamic, points_dynamic = ik_env.run_ik_demo(
            use_dynamic_target=True, 
            num_points=50, 
            animate=True
        )
        
        # Keep viewer open for inspection
        print("\nDemo completed! Viewer will remain open for inspection.")
        print("Press 'q' to quit or close the viewer window.")
        
        # Keep the viewer open
        while ik_env.env.is_viewer_alive():
            ik_env.env.render()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        ik_env.close()


if __name__ == "__main__":
    import fire 
    fire.Fire(main)
