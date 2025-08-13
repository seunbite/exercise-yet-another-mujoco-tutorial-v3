import numpy as np
import mujoco
import time
import matplotlib.pyplot as plt
import imageio
import os
from mjct.mujoco_usage.mujoco_parser import MuJoCoParserClass
from mjct.helper.transformation import r2rpy, rpy2r, pr2t, t2p, t2r
from mjct.helper.utility import get_colors, d2r, trim_scale, np2torch, torch2np


class UR5eIKEnv:
    def __init__(
            self,
            xml_path='asset/makeup_frida/scene_table.xml',
            HZ=50,
            object_table_position=[1.0, 0, 0], 
            base_table_position=[0, 0, 0],
            head_position=[1.2, 0.0, 0.53], 
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
        try:
            self.env.init_viewer(title='UR5e IK Demo', width=1400, height=1000)
            print("✓ Interactive viewer initialized successfully")
            
            # Try to set viewer to window mode for better frame capture
            try:
                if hasattr(self.env.viewer, '_viewer'):
                    if hasattr(self.env.viewer._viewer, 'set_mode'):
                        self.env.viewer._viewer.set_mode('window')
                        print("✓ Set viewer to window mode")
                    elif hasattr(self.env.viewer._viewer, 'mode'):
                        self.env.viewer._viewer.mode = 'window'
                        print("✓ Set viewer mode to window")
                    elif hasattr(self.env.viewer._viewer, 'window_mode'):
                        self.env.viewer._viewer.window_mode = True
                        print("✓ Enabled window mode")
            except Exception as mode_error:
                print(f"Warning: Could not set window mode: {mode_error}")
                
        except Exception as e:
            print(f"Warning: Could not initialize interactive viewer: {e}")
            print("Falling back to headless rendering only")
        
        # Initialize renderer as None - will be created on-demand when needed
        self.renderer = None
        
        # Prime camera and reset environment
        self._prime_camera()
        self.reset()
    
    def _prime_camera(self):
        """Prime the camera for better initial view"""
        try:
            if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                # Set initial camera position
                if hasattr(self.env.viewer, 'cam'):
                    self.env.viewer.cam.distance = 2.0
                    self.env.viewer.cam.azimuth = 90
                    self.env.viewer.cam.elevation = -20
                    self.env.viewer.cam.lookat = np.array([0.5, 0, 0.5])
                elif hasattr(self.env.viewer, 'free_cam'):
                    self.env.viewer.free_cam.distance = 2.0
                    self.env.viewer.free_cam.azimuth = 90
                    self.env.viewer.free_cam.elevation = -20
                    self.env.viewer.free_cam.lookat = np.array([0.5, 0, 0.5])
                print("✓ Camera primed successfully")
        except Exception as e:
            print(f"Warning: Could not prime camera: {e}")
    
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
        if not hasattr(self.env, 'viewer') or self.env.viewer is None:
            return
        
        # Check if viewer is alive before rendering
        viewer_alive = False
        try:
            if hasattr(self.env.viewer, 'is_alive'):
                if callable(self.env.viewer.is_alive):
                    viewer_alive = self.env.viewer.is_alive()
                else:
                    viewer_alive = self.env.viewer.is_alive
            elif hasattr(self.env.viewer, 'window_open'):
                viewer_alive = self.env.viewer.window_open
            else:
                viewer_alive = True  # Assume alive if we can't check
        except Exception:
            viewer_alive = False
        
        if not viewer_alive:
            print("Warning: Viewer not alive, skipping target line render")
            return
        
        try:
            self.env.render()
            
            if hasattr(self.env.viewer, 'sync'):
                try:
                    self.env.viewer.sync()
                except Exception as e:
                    print(f"Warning: Could not sync viewer: {e}")
            elif hasattr(self.env.viewer, 'render'):
                try:
                    self.env.viewer.render()
                except Exception as e:
                    print(f"Warning: Could not render viewer: {e}")
        except Exception as e:
            print(f"Warning: Could not render target line: {e}")
    
    def animate_trajectory(self, joint_trajectory, trajectory_points, 
                          animation_speed=0.1, render_every=1, from_start=True, goal_start_point=None, goal_end_point=None,
                          save_gif=False, gif_path=None):
        """Animate the robot following the solved trajectory"""
        print(f"Animating trajectory with {len(joint_trajectory)} waypoints...")
        
        # Ensure viewer is alive before starting animation
        if not self._check_and_reinit_viewer():
            print("Warning: Could not initialize viewer, continuing with headless rendering only")
        
        frames = []  # Store frames for GIF creation
        
        for i, (joint_config, target_point) in enumerate(zip(joint_trajectory, trajectory_points)):
            if i % render_every == 0:
                # Set joint positions
                joint_idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
                self.env.forward(q=joint_config, joint_idxs=joint_idxs)
                
                # Get current end-effector position
                current_tip_pos = self.env.get_p_body('applicator_tip')
                
                # Check if viewer is still alive before rendering
                viewer_alive = False
                if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                    try:
                        if hasattr(self.env.viewer, 'is_alive'):
                            if callable(self.env.viewer.is_alive):
                                viewer_alive = self.env.viewer.is_alive()
                            else:
                                viewer_alive = self.env.viewer.is_alive
                        elif hasattr(self.env.viewer, 'window_open'):
                            viewer_alive = self.env.viewer.window_open
                        else:
                            viewer_alive = True  # Assume alive if we can't check
                    except Exception:
                        viewer_alive = False
                
                # Render with error handling
                if viewer_alive:
                    try:
                        self.env.render()
                    except Exception as render_error:
                        print(f"Warning: Could not render frame {i+1}: {render_error}")
                        viewer_alive = False
                else:
                    print(f"Warning: Viewer not alive for frame {i+1}, skipping render")
                
                # Try to reinitialize viewer if it was closed
                if not viewer_alive:
                    try:
                        print("Attempting to reinitialize viewer...")
                        self.env.init_viewer(title='UR5e IK Demo', width=1400, height=1000)
                        print("✓ Viewer reinitialized successfully")
                    except Exception as reinit_error:
                        print(f"Warning: Could not reinitialize viewer: {reinit_error}")
                        # Continue without rendering
                        pass
                
                if from_start:
                    # Traditional mode: show trajectory start and end points
                    trajectory_start = trajectory_points[0] if len(trajectory_points) > 0 else None
                    trajectory_end = trajectory_points[-1] if len(trajectory_points) > 0 else None
                    
                    if trajectory_start is not None:
                        self.env.plot_sphere(p=trajectory_start, r=0.008, rgba=[0, 1, 0, 0.8], label='Start')
                    if trajectory_end is not None:
                        self.env.plot_sphere(p=trajectory_end, r=0.008, rgba=[0, 0, 1, 0.8], label='End')
                else:
                    # Two-stage mode: show the two goal points (IK targets)
                    if goal_start_point is not None:
                        self.env.plot_sphere(p=goal_start_point, r=0.008, rgba=[1, 0, 0, 0.8], label='Goal Start')
                    if goal_end_point is not None:
                        self.env.plot_sphere(p=goal_end_point, r=0.008, rgba=[0, 0, 1, 0.8], label='Goal End')
                
                # Always show current tip position
                self.env.plot_sphere(p=current_tip_pos, r=0.006, rgba=[1, 1, 0, 0.8], label='Current')
                
                # Show line from current position to current target
                self.env.plot_line_fr2to(p_fr=current_tip_pos, p_to=target_point, rgba=[1, 0.5, 0, 1])
                
                # Capture frame for GIF if requested
                if save_gif:
                    try:
                        # Use the synchronized renderer for consistent frame capture
                        frame = self._capture_frame_with_renderer(width=640, height=480)
                        if frame is not None:
                            frames.append(frame)
                            print(f"  ✓ Captured frame {i+1} for GIF")
                        else:
                            print(f"Warning: Could not capture frame {i+1} for GIF")
                    except Exception as e:
                        print(f"Warning: Frame capture failed for frame {i+1}: {e}")
            
        if save_gif and len(frames) > 0:
            try:
                if gif_path is None:
                    # Create default filename with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    mode_str = "from_start" if from_start else "two_stage"
                    gif_path = f"ik_demo_{mode_str}_{timestamp}.gif"
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else '.', exist_ok=True)
                
                # Save GIF with reasonable frame rate
                duration = max(0.1, animation_speed)  # Minimum 0.1s per frame
                fps = max(1, int(1/duration))  # Ensure fps is at least 1
                imageio.mimsave(gif_path, frames, fps=fps)
                print(f"GIF saved: {gif_path} ({len(frames)} frames)")
            except Exception as e:
                print(f"Error saving GIF: {e}")
        elif save_gif:
            print("Warning: No frames captured for GIF")
        
        print("Animation completed!")
        
        # Ensure the viewer stays open and visible
        if hasattr(self.env.viewer, 'render'):
            try:
                self.env.viewer.render()
                # Give a moment for the viewer to update
                time.sleep(0.1)
            except Exception as e:
                print(f"Warning: Could not render final frame: {e}")
        elif hasattr(self.env.viewer, 'sync'):
            try:
                self.env.viewer.sync()
                time.sleep(0.1)
            except Exception as e:
                print(f"Warning: Could not sync final frame: {e}")
    
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
    
    def create_target_line(self, use_dynamic_target=False, from_start=True):
        """Create a target line (start and end points)"""
        if from_start:
            # Traditional approach: start from a fixed point
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
        else:
            current_tip_pos = self.env.get_p_body('applicator_tip')
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            line_length = np.random.uniform(0.08, 0.15)
            
            current_pos, target_pos_start = self._make_line(current_tip_pos, direction, line_length)
            direction = np.random.randn(3)
            line_direction = direction / np.linalg.norm(direction)
            line_length = np.random.uniform(0.03, 0.05)
            
        target_line = self._make_line(target_pos_start, direction=line_direction, length=line_length)
        return target_line
    
    def run_ik_demo(self, use_dynamic_target=False, num_points=50, animate=True, animation_speed=0.1, from_start=True, save_gif=False, gif_path=None):
        """Run the complete IK demonstration"""
        print("=== UR5e Inverse Kinematics Demo ===")
        
        if from_start:
            target_line = self.create_target_line(use_dynamic_target, from_start=True)
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
                    # Try to animate the trajectory
                    try:
                        self.animate_trajectory(joint_trajectory, trajectory_points, animation_speed=animation_speed, from_start=True, save_gif=save_gif, gif_path=gif_path)
                    except Exception as anim_error:
                        print(f"Warning: Animation failed: {anim_error}")
                        print("Continuing without animation...")
                        if save_gif:
                            print("Note: GIF creation also failed due to animation error")
                
                return joint_trajectory, trajectory_points
            else:
                print("Failed to solve IK for any points along the trajectory")
                return None, None
        
        else:
            # Two-stage approach: current tip -> start point, then start point -> end point
            print("Two-stage IK solving:")
            
            # Get current tip position
            current_tip_pos = self.env.get_p_body('applicator_tip')
            
            # Create target line (start point = current tip, end point = random nearby)
            target_line = self.create_target_line(use_dynamic_target, from_start=False)
            start_point = target_line[0]  # This is current tip position
            end_point = target_line[1]    # This is random nearby position
            
            print(f"Current tip position: {current_tip_pos}")
            print(f"Target end point: {end_point}")
            
            # Stage 1: Move from current tip to start point (should be minimal since they're the same)
            print("\n--- Stage 1: Current tip -> Start point ---")
            joint_traj_1, points_1 = self.solve_ik_trajectory(
                current_tip_pos, start_point, num_points 
            )
            
            # Stage 2: Move from start point to end point
            print("\n--- Stage 2: Start point -> End point ---")
            joint_traj_2, points_2 = self.solve_ik_trajectory(
                start_point, end_point, num_points
            )
            
            # Combine trajectories
            if len(joint_traj_1) > 0 and len(joint_traj_2) > 0:
                joint_trajectory = np.concatenate([joint_traj_1, joint_traj_2], axis=0)
                trajectory_points = np.concatenate([points_1, points_2], axis=0)
                
                print(f"Successfully generated combined trajectory with {len(joint_trajectory)} waypoints")
                print(f"  Stage 1: {len(joint_traj_1)} waypoints")
                print(f"  Stage 2: {len(joint_traj_2)} waypoints")
                
                # Render the target line
                self.render_target_line(start_point, end_point)
                
                if animate:
                    # Try to animate the combined trajectory with goal points
                    try:
                        self.animate_trajectory(joint_trajectory, trajectory_points, animation_speed=animation_speed, 
                                              from_start=False, goal_start_point=start_point, goal_end_point=end_point,
                                              save_gif=save_gif, gif_path=gif_path)
                    except Exception as anim_error:
                        print(f"Warning: Animation failed: {anim_error}")
                        print("Continuing without animation...")
                        if save_gif:
                            print("Note: GIF creation also failed due to animation error")
                
                return joint_trajectory, trajectory_points
            else:
                print("Failed to solve IK for one or both stages")
                print(f"  Stage 1 success: {len(joint_traj_1) > 0}")
                print(f"  Stage 2 success: {len(joint_traj_2) > 0}")
                return None, None
    
    def _ensure_renderer(self, width=640, height=480):
        """Create renderer on-demand when needed for GIF capture"""
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.env.model, width=width, height=height)
                print("✓ Headless renderer created on demand")
            except Exception as e:
                print(f"Warning: Could not create renderer: {e}")
                return False
        return True
    
    def _copy_viewer_cam_to_renderer(self):
        """Copy viewer camera parameters to renderer for consistent view"""
        try:
            if self.renderer is None:
                return None
                
            # Get camera parameters from viewer
            vcam = None
            if hasattr(self.env.viewer, 'cam'):
                vcam = self.env.viewer.cam
            elif hasattr(self.env.viewer, 'free_cam'):
                vcam = self.env.viewer.free_cam
            
            if vcam is None:
                return None
            
            # Create camera object for renderer
            try:
                cam = mujoco.MjvCamera()
                cam.lookat = np.array(vcam.lookat, dtype=float)
                cam.distance = float(getattr(vcam, 'distance', 1.8))
                cam.azimuth = float(getattr(vcam, 'azimuth', 90))
                cam.elevation = float(getattr(vcam, 'elevation', -20))
                return cam
            except Exception:
                # Fallback: try to set renderer camera directly
                try:
                    if hasattr(self.renderer, 'camera'):
                        self.renderer.camera.lookat = np.array(vcam.lookat, dtype=float)
                        self.renderer.camera.distance = float(getattr(vcam, 'distance', 1.8))
                        self.renderer.camera.azimuth = float(getattr(vcam, 'azimuth', 90))
                        self.renderer.camera.elevation = float(getattr(vcam, 'elevation', -20))
                        return "direct"
                except Exception:
                    pass
                return None
        except Exception as e:
            print(f"Warning: Could not copy camera parameters: {e}")
            return None
    
    def _capture_frame_with_renderer(self, width=640, height=480):
        """Capture frame using headless renderer with synchronized camera"""
        if not self._ensure_renderer(width, height):
            return None
            
        try:
            # Copy camera parameters from viewer
            cam = self._copy_viewer_cam_to_renderer()
            
            # Update scene and render
            if cam is not None and cam != "direct":
                self.renderer.update_scene(self.env.data, camera=cam)
            else:
                self.renderer.update_scene(self.env.data)
            
            frame = self.renderer.render()
            
            if frame is not None:
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Ensure correct shape
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Flip vertically to match expected orientation
                    frame = np.flipud(frame)
                    return frame
                else:
                    print(f"Warning: Frame has unexpected shape: {frame.shape}")
                    return None
            else:
                print("Warning: Renderer returned None frame")
                return None
                
        except Exception as e:
            print(f"Warning: Frame capture with renderer failed: {e}")
            return None
    
    def _capture_from_viewer_window_mode(self):
        """Try to capture frame from viewer in window mode"""
        try:
            # Set window mode if available
            if hasattr(self.env.viewer._viewer, 'set_mode'):
                self.env.viewer._viewer.set_mode('window')
            elif hasattr(self.env.viewer._viewer, 'mode'):
                self.env.viewer._viewer.mode = 'window'
            
            # Give a moment for mode change to take effect
            time.sleep(0.01)
            
            return self.env.viewer._viewer.read_pixels(depth=False)
        except Exception as e:
            raise Exception(f"Window mode capture failed: {e}")
    
    def _capture_from_viewer_offscreen(self):
        """Try to capture frame from viewer using offscreen context"""
        try:
            if hasattr(self.env.viewer._viewer, 'read_pixels_offscreen'):
                return self.env.viewer._viewer.read_pixels_offscreen(depth=False)
            else:
                raise Exception("Offscreen capture not available")
        except Exception as e:
            raise Exception(f"Offscreen capture failed: {e}")
    
    def _check_and_reinit_viewer(self):
        """Check if viewer is alive and reinitialize if needed"""
        viewer_alive = False
        try:
            if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                if hasattr(self.env.viewer, 'is_alive'):
                    if callable(self.env.viewer.is_alive):
                        viewer_alive = self.env.viewer.is_alive()
                    else:
                        viewer_alive = self.env.viewer.is_alive
                elif hasattr(self.env.viewer, 'window_open'):
                    viewer_alive = self.env.viewer.window_open
                else:
                    viewer_alive = True  # Assume alive if we can't check
        except Exception:
            viewer_alive = False
        
        if not viewer_alive:
            try:
                print("Viewer not alive, attempting to reinitialize...")
                self.env.init_viewer(title='UR5e IK Demo', width=1400, height=1000)
                print("✓ Viewer reinitialized successfully")
                return True
            except Exception as e:
                print(f"Warning: Could not reinitialize viewer: {e}")
                return False
        
        return True
    
    def close(self):
        """Close the environment"""
        try:
            # Clean up renderer
            if self.renderer is not None:
                del self.renderer
                self.renderer = None
                print("✓ Renderer cleaned up")
        except Exception as e:
            print(f"Warning: Could not delete renderer: {e}")
        
        try:
            if hasattr(self.env, 'close_viewer'):
                self.env.close_viewer()
            elif hasattr(self.env, 'viewer') and self.env.viewer is not None:
                if hasattr(self.env.viewer, 'close'):
                    self.env.viewer.close()
                elif hasattr(self.env.viewer, 'exit'):
                    self.env.viewer.exit()
        except Exception as e:
            print(f"Warning: Could not close viewer: {e}")


def main(
    xml_path='asset/makeup_frida/scene_table.xml',
    num_trials=50,
    num_points=50,
    animate=True,
    use_dynamic_target=True,
    from_start=True,
    save_gif=True,
    gif_dir='./gifs/',
    animation_speed=0.001,
    wait_between_trials=0.5,
    HZ=50,
    object_table_position=[1.0, 0, 0],
    base_table_position=[0, 0, 0],
    head_position=[1, 0.0, 0.53],
    waiting_time=0.0
):
    """
    Main function to run the IK demo
    
    Args:
        xml_path: Path to the MuJoCo XML file
        num_trials: Number of trials to run
        num_points: Number of points along each trajectory
        animate: Whether to animate the trajectory
        use_dynamic_target: Whether to use dynamic targets (True) or static targets (False)
        from_start: If True, start from fixed point. If False, start from current tip position (two-stage IK)
        save_gif: Whether to save animations as GIF files
        gif_dir: Directory to save GIF files
        animation_speed: Speed of animation (seconds between frames)
        wait_between_trials: Wait time between trials (seconds)
        HZ: Control frequency
        object_table_position: Position of object table
        base_table_position: Position of base table
        head_position: Position of head (target reference point)
        waiting_time: Initial waiting time
    """
    # Create IK environment
    ik_env = UR5eIKEnv(
        xml_path=xml_path,
        HZ=HZ,
        object_table_position=object_table_position,
        base_table_position=base_table_position,
        head_position=head_position,
        waiting_time=waiting_time
    )
    
    try:
        # Create GIF directory if saving GIFs
        if save_gif:
            os.makedirs(gif_dir, exist_ok=True)
        
        print(f"\n=== Running {num_trials} IK Demo Trials ===")
        print(f"Target type: {'Dynamic' if use_dynamic_target else 'Static'}")
        print(f"Start mode: {'From fixed point' if from_start else 'From current tip (two-stage)'}")
        print(f"Points per trajectory: {num_points}")
        print(f"Animation: {'Enabled' if animate else 'Disabled'}")
        print(f"Save GIF: {'Enabled' if save_gif else 'Disabled'}")
        if save_gif:
            print(f"GIF directory: {gif_dir}")
        
        successful_trials = 0
        total_points_solved = 0
        total_points_attempted = 0
        
        for trial in range(num_trials):
            print(f"\n--- Trial {trial + 1}/{num_trials} ---")
            
            # Reset environment for each trial
            ik_env.reset()
            
            # Generate unique GIF path for this trial if saving GIFs
            gif_path = None
            if save_gif:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                mode_str = "from_start" if from_start else "two_stage"
                target_str = "dynamic" if use_dynamic_target else "static"
                gif_filename = f"trial_{trial+1:02d}_{mode_str}_{target_str}_{timestamp}.gif"
                gif_path = os.path.join(gif_dir, gif_filename)
            
            # Run IK demo
            joint_trajectory, trajectory_points = ik_env.run_ik_demo(
                use_dynamic_target=use_dynamic_target, 
                num_points=num_points, 
                animate=animate,
                animation_speed=animation_speed,
                from_start=from_start,
                save_gif=save_gif,
                gif_path=gif_path
            )
            
            if joint_trajectory is not None and len(joint_trajectory) > 0:
                successful_trials += 1
                points_solved = len(joint_trajectory)
                total_points_solved += points_solved
                print(f"Trial {trial + 1} SUCCESS: {points_solved}/{num_points} points solved")
            else:
                print(f"Trial {trial + 1} FAILED: No points could be solved")
            
            total_points_attempted += num_points
            
            # Wait between trials (except for the last one)
            if trial < num_trials - 1 and wait_between_trials > 0:
                print(f"Waiting {wait_between_trials}s before next trial...")
                time.sleep(wait_between_trials)
        
        # Print summary statistics
        print(f"\n=== SUMMARY ===")
        print(f"Successful trials: {successful_trials}/{num_trials} ({successful_trials/num_trials*100:.1f}%)")
        print(f"Total points solved: {total_points_solved}/{total_points_attempted} ({total_points_solved/total_points_attempted*100:.1f}%)")
        print(f"Average points per successful trial: {total_points_solved/max(successful_trials, 1):.1f}")
        
        # Keep viewer open for inspection
        print("\nDemo completed! Viewer will remain open for inspection.")
        print("Press Ctrl+C to quit or close the viewer window.")
        
        # Keep the viewer open
        try:
            # Check if viewer exists and is alive using available methods
            viewer_alive = False
            if hasattr(ik_env.env, 'viewer') and ik_env.env.viewer is not None:
                if hasattr(ik_env.env, 'is_viewer_alive'):
                    # Check if it's a callable method or a property
                    if callable(ik_env.env.is_viewer_alive):
                        viewer_alive = ik_env.env.is_viewer_alive()
                    else:
                        viewer_alive = ik_env.env.is_viewer_alive
                elif hasattr(ik_env.env.viewer, 'is_alive'):
                    # Check if it's a callable method or a property
                    if callable(ik_env.env.viewer.is_alive):
                        viewer_alive = ik_env.env.viewer.is_alive()
                    else:
                        viewer_alive = ik_env.env.viewer.is_alive
                elif hasattr(ik_env.env.viewer, 'window_open'):
                    viewer_alive = ik_env.env.viewer.window_open
                else:
                    # Assume viewer is alive if we can't check
                    viewer_alive = True
            else:
                print("Warning: No viewer available, skipping viewer loop")
                viewer_alive = False
            
            if viewer_alive:
                print("Keeping viewer open for inspection...")
                while viewer_alive:
                    try:
                        ik_env.env.render()
                        # Update viewer alive status
                        if hasattr(ik_env.env, 'is_viewer_alive'):
                            # Check if it's a callable method or a property
                            if callable(ik_env.env.is_viewer_alive):
                                viewer_alive = ik_env.env.is_viewer_alive()
                            else:
                                viewer_alive = ik_env.env.is_viewer_alive
                        elif hasattr(ik_env.env.viewer, 'is_alive'):
                            # Check if it's a callable method or a property
                            if callable(ik_env.env.viewer.is_alive):
                                viewer_alive = ik_env.env.viewer.is_alive()
                            else:
                                viewer_alive = ik_env.env.viewer.is_alive
                        elif hasattr(ik_env.env.viewer, 'window_open'):
                            viewer_alive = ik_env.env.viewer.window_open
                    except Exception as e:
                        print(f"Viewer render error: {e}")
                        break
                    time.sleep(0.1)
            else:
                print("Viewer not available, demo completed successfully")
        except Exception as e:
            print(f"Viewer loop error: {e}")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        ik_env.close()


if __name__ == "__main__":
    import fire 
    fire.Fire(main)
