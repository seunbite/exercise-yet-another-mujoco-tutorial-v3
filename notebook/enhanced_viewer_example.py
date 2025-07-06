import sys, mujoco
import numpy as np
import time
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/bvh_parser/')
sys.path.append('../package/motion_retarget/')
from mujoco_parser import *
from slider import *
from utility import *
from transformation import *
from bvh_parser import *
from joi import *
from mr_cmu import *

def enhanced_viewer_example(
    xml_path = '../asset/unitree_g1/scene_g1.xml',
    bvh_path = "../bvh/cmu_mocap/05_14.bvh",
    p_rate = 0.056444*1.2,
    # Viewer configuration
    viewer_width = 1600,
    viewer_height = 1000,
    viewer_title = "Enhanced Motion Retargeting Viewer",
    # Camera settings
    camera_azimuth = 170,
    camera_distance = 4.0,
    camera_elevation = -25,
    camera_lookat = [0.0, 0.1, 0.7],
    # Rendering settings
    render_every = 5,
    playback_speed = 0.1,
    enable_transparency = True,
    show_contacts = True,
    show_joints = True,
    # Visualization options
    show_source_skeleton = True,
    show_target_skeleton = True,
    show_ik_targets = True,
    show_coordinate_frames = True,
    # Scaling and rotation options
    source_scale = 1.0,
    source_rotation_deg = 0.0,
    source_translation = [0.0, 0.0, 0.0],
):
    """
    Enhanced MuJoCo viewer example for motion retargeting with comprehensive
    visualization options and proper viewer configuration.
    """
    
    print("=== Enhanced MuJoCo Viewer for Motion Retargeting ===")
    print(f"Viewer: {viewer_width}x{viewer_height} - {viewer_title}")
    print(f"Camera: Azimuth={camera_azimuth}, Distance={camera_distance}, Elevation={camera_elevation}")
    print(f"Rendering: Every {render_every} frames, Speed={playback_speed}")
    print(f"Source scaling: {source_scale}, Rotation: {source_rotation_deg}°")
    print()
    
    # Initialize environment
    env = MuJoCoParserClass(name='EnhancedRetarget', rel_xml_path=xml_path, verbose=False)
    
    # Load motion data
    print("Loading BVH motion data...")
    secs, chains = get_chains_from_bvh_cmu_mocap(
        bvh_path=bvh_path,
        p_rate=p_rate,
        zup=True,
        plot_chain_graph=False,
        verbose=False,
    )
    secs, chains = secs[1:], chains[1:]  # exclude first frame
    L = len(chains)
    dt = secs[1] - secs[0] if len(secs) > 1 else 0.033
    HZ = int(1/dt)
    print(f"Loaded {L} frames at {HZ} Hz")
    
    # Calculate contact detection
    print("Calculating contact detection...")
    rtoe_traj, ltoe_traj = np.zeros((L, 3)), np.zeros((L, 3))
    ra_traj, la_traj = np.zeros((L, 3)), np.zeros((L, 3))
    
    for tick in range(L):
        chain = chains[tick]
        T_joi = get_T_joi_from_chain_cmu(chain)
        rtoe_traj[tick, :], ltoe_traj[tick, :] = t2p(T_joi['rtoe']), t2p(T_joi['ltoe'])
        ra_traj[tick, :], la_traj[tick, :] = t2p(T_joi['ra']), t2p(T_joi['la'])
    
    rcontact_segs, lcontact_segs = get_contact_segments(
        secs, rtoe_traj, ltoe_traj,
        zvel_th=0.1, min_seg_sec=0.1, smt_sigma=5.0, smt_radius=5.0,
        verbose=False, plot=False,
    )
    rcontact_segs_concat = np.concatenate(rcontact_segs) if rcontact_segs else np.array([])
    lcontact_segs_concat = np.concatenate(lcontact_segs) if lcontact_segs else np.array([])
    
    # Reset environment
    env.reset(step=True)
    
    # Enhanced viewer initialization
    print("Initializing enhanced viewer...")
    env.init_viewer(
        title=viewer_title,
        width=viewer_width,
        height=viewer_height,
        hide_menu=False,  # Show menu for debugging
        fontscale=mujoco.mjtFontScale.mjFONTSCALE_150.value,
        maxgeom=15000,  # Increase for more complex scenes
    )
    
    # Configure viewer settings
    env.set_viewer(
        azimuth=camera_azimuth,
        distance=camera_distance,
        elevation=camera_elevation,
        lookat=camera_lookat,
        transparent=enable_transparency,
        contactpoint=show_contacts,
        contactwidth=0.05,
        contactheight=0.05,
        contactrgba=np.array([1, 0, 0, 1]),
        joint=show_joints,
        jointlength=0.3,
        jointwidth=0.05,
        jointrgba=[0.2, 0.6, 0.8, 0.8],
        update=True,  # Apply settings immediately
    )
    
    print("Viewer initialized successfully!")
    print("Controls:")
    print("  - Mouse: Rotate camera")
    print("  - Scroll: Zoom in/out")
    print("  - Space: Pause/resume")
    print("  - Right arrow: Step forward")
    print("  - ESC: Close viewer")
    print()
    
    # Main rendering loop
    tick = 0
    qpos_list, R_ra_list, R_la_list = [], [], []
    
    while env.is_viewer_alive() and (tick < L):
        try:
            # Get source chain and apply scaling/rotation
            chain = chains[tick]
            T_joi_src = get_T_joi_from_chain_cmu(chain, hip_between_pelvis=True)
            
            # Apply scaling and rotation to source motion
            if source_scale != 1.0 or source_rotation_deg != 0.0 or np.any(np.array(source_translation) != 0):
                # Create transformation matrix for scaling and rotation
                R_scale_rot = rpy2r(np.radians([0, 0, source_rotation_deg]))
                T_transform = np.eye(4)
                T_transform[:3, :3] = R_scale_rot * source_scale
                T_transform[:3, 3] = np.array(source_translation)
                
                # Apply transformation to all source joints
                for key in T_joi_src:
                    T_joi_src[key] = T_transform @ T_joi_src[key]
            
            # Contact detection
            rcontact = tick in rcontact_segs_concat
            lcontact = tick in lcontact_segs_concat
            
            # Motion retargeting (simplified for example)
            # ... your existing IK logic here ...
            
            # Enhanced rendering
            if tick % render_every == 0:
                # Clear previous frame
                env.render()
                
                # Show coordinate frame at origin
                if show_coordinate_frames:
                    env.plot_T(p=np.zeros(3), R=np.eye(3), plot_axis=True, axis_len=0.5, axis_width=0.01)
                
                # Show source skeleton (red)
                if show_source_skeleton:
                    chain.plot_chain_mujoco(
                        env=env,
                        plot_joint=False,
                        plot_joint_name=False,
                        r_link=0.003,
                        rgba_link=(1, 0, 0, 0.6),
                    )
                    # Draw source joint spheres
                    for key in T_joi_src.keys():
                        env.plot_sphere(p=t2p(T_joi_src[key]), r=0.015, rgba=(1, 0, 0, 0.7))
                
                # Show target skeleton (blue) - if available
                if show_target_skeleton:
                    try:
                        T_joi_trgt = get_T_joi_from_g1(env)
                        env.plot_links_between_bodies(r=0.003, rgba=(0, 0, 1, 0.6))
                        for key in T_joi_trgt.keys():
                            env.plot_sphere(p=t2p(T_joi_trgt[key]), r=0.015, rgba=(0, 0, 1, 0.7))
                    except:
                        pass  # Target not available yet
                
                # Show IK targets (green)
                if show_ik_targets and 'ik_info_full_body' in locals():
                    plot_ik_info(env=env, ik_info=ik_info_full_body)
                
                # Show contact information
                if show_contacts:
                    contact_color = (1, 1, 0, 0.8) if rcontact or lcontact else (0.5, 0.5, 0.5, 0.3)
                    env.plot_text(p=[0, 0, 1.5], label=f"Frame: {tick+1}/{L}")
                    env.plot_text(p=[0, 0, 1.4], label=f"Contacts: R={rcontact}, L={lcontact}")
                
                # Show time and frame info
                env.plot_time()
                
                # Render the frame
                env.render()
                
                # Control playback speed
                if playback_speed > 0:
                    time.sleep(playback_speed)
            
            # Store data
            qpos_list.append(env.get_qpos())
            R_ra_list.append(env.get_R_body(body_name='right_ankle_roll_link'))
            R_la_list.append(env.get_R_body(body_name='left_ankle_roll_link'))
            
            tick += 1
            
            # Progress indicator
            if tick % 50 == 0:
                progress = (tick / L) * 100
                print(f"Progress: {progress:.1f}% ({tick}/{L} frames)")
                
        except Exception as e:
            print(f"Error in frame {tick}: {e}")
            break
    
    # Cleanup
    print("Closing viewer...")
    env.close_viewer()
    print("Enhanced viewer example completed!")
    
    return {
        'qpos_list': qpos_list,
        'R_ra_list': R_ra_list,
        'R_la_list': R_la_list,
        'frames_processed': tick,
        'total_frames': L,
    }

def viewer_camera_controls_example():
    """
    Example showing how to control the camera programmatically
    """
    env = MuJoCoParserClass(name='CameraControl', rel_xml_path='../asset/unitree_g1/scene_g1.xml')
    env.reset(step=True)
    
    # Initialize viewer
    env.init_viewer(title="Camera Control Example")
    
    # Set initial camera position
    env.set_viewer(
        azimuth=0,
        distance=3.0,
        elevation=-20,
        lookat=[0, 0, 0.7],
    )
    
    # Camera control loop
    tick = 0
    while env.is_viewer_alive() and tick < 1000:
        # Rotate camera around the scene
        azimuth = (tick * 0.5) % 360
        env.set_viewer(azimuth=azimuth, update=True)
        
        # Render
        env.render()
        time.sleep(0.05)
        tick += 1
    
    env.close_viewer()

def viewer_screenshot_example():
    """
    Example showing how to capture screenshots from the viewer
    """
    env = MuJoCoParserClass(name='Screenshot', rel_xml_path='../asset/unitree_g1/scene_g1.xml')
    env.reset(step=True)
    
    # Initialize viewer
    env.init_viewer(title="Screenshot Example")
    
    # Take screenshots
    for i in range(5):
        # Set different camera angles
        env.set_viewer(
            azimuth=i * 72,  # Rotate around scene
            distance=4.0,
            elevation=-20,
            lookat=[0, 0, 0.7],
            update=True
        )
        
        # Render and grab image
        env.render()
        rgb_img, depth_img = env.grab_rgbd_img()
        
        # Save screenshot
        import cv2
        cv2.imwrite(f'screenshot_{i:02d}.png', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        print(f"Saved screenshot_{i:02d}.png")
        
        time.sleep(0.5)
    
    env.close_viewer()

if __name__ == "__main__":
    import fire
    fire.Fire({
        'enhanced': enhanced_viewer_example,
        'camera_control': viewer_camera_controls_example,
        'screenshot': viewer_screenshot_example,
    }) 