import sys
import mujoco
import mujoco_viewer
import numpy as np
import time
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/bvh_parser/')
sys.path.append('../package/motion_retarget/')
from transformation import *
from bvh_parser import *
from joi import *
from mr_cmu import *

def standard_viewer_retargeting(
    xml_path = '../asset/unitree_g1/scene_g1.xml',
    bvh_path = "../bvh/cmu_mocap/05_14.bvh",
    p_rate = 0.056444*1.2,
    # Motion scaling and rotation
    source_scale = 0.8,  # Scale down the motion
    source_rotation_deg = 45.0,  # Rotate the motion
    source_translation = [0.5, 0.0, 0.0],  # Move the motion
    # Viewer settings
    max_frames = 100,
    render_every = 5,
    playback_speed = 0.1,
):
    """
    Motion retargeting using standard mujoco_viewer with scaling and rotation.
    
    This example shows how to:
    1. Use standard mujoco_viewer instead of custom wrapper
    2. Scale, rotate, and translate the source motion data
    3. Apply motion retargeting with proper visualization
    """
    
    print("=== Standard mujoco_viewer Motion Retargeting ===")
    print(f"Source scaling: {source_scale}")
    print(f"Source rotation: {source_rotation_deg}°")
    print(f"Source translation: {source_translation}")
    print()
    
    # Load MuJoCo model and data
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Create standard viewer
    print("Initializing mujoco_viewer...")
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
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
    L = min(len(chains), max_frames)
    print(f"Processing {L} frames")
    
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
    
    # Create transformation matrix for scaling and rotation
    R_scale_rot = rpy2r(np.radians([0, 0, source_rotation_deg]))
    T_transform = np.eye(4)
    T_transform[:3, :3] = R_scale_rot * source_scale
    T_transform[:3, 3] = np.array(source_translation)
    
    print("Starting motion retargeting loop...")
    print("Controls:")
    print("  - Mouse: Rotate camera")
    print("  - Scroll: Zoom in/out")
    print("  - ESC: Close viewer")
    print()
    
    # Main rendering loop
    tick = 0
    qpos_list = []
    
    while viewer.is_alive and tick < L:
        try:
            # Get source chain
            chain = chains[tick]
            T_joi_src = get_T_joi_from_chain_cmu(chain, hip_between_pelvis=True)
            
            # Apply scaling and rotation to source motion
            T_joi_src_transformed = {}
            for key, T in T_joi_src.items():
                T_joi_src_transformed[key] = T_transform @ T
            
            # Contact detection
            rcontact = tick in rcontact_segs_concat
            lcontact = tick in lcontact_segs_concat
            
            # Motion retargeting logic (simplified)
            # In a real implementation, you would do IK here
            # For this example, we'll just update the robot base position
            
            # Update robot base based on transformed source hip position
            hip_pos = t2p(T_joi_src_transformed['hip'])
            
            # Set robot base position (assuming free joint at index 0-2)
            data.qpos[0:3] = hip_pos
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Render visualization
            if tick % render_every == 0:
                # Add markers to show source skeleton (red)
                for key, T in T_joi_src_transformed.items():
                    pos = t2p(T)
                    viewer.add_marker(
                        pos=pos,
                        size=[0.02, 0.02, 0.02],
                        rgba=[1, 0, 0, 0.7],  # Red
                        label=f"src_{key}"
                    )
                
                # Add markers to show robot joints (blue)
                # This is a simplified version - you'd need to get actual robot joint positions
                robot_hip_pos = data.qpos[0:3]
                viewer.add_marker(
                    pos=robot_hip_pos,
                    size=[0.03, 0.03, 0.03],
                    rgba=[0, 0, 1, 0.8],  # Blue
                    label="robot_hip"
                )
                
                # Add contact indicators
                if rcontact or lcontact:
                    contact_pos = hip_pos + np.array([0, 0, 0.1])
                    viewer.add_marker(
                        pos=contact_pos,
                        size=[0.05, 0.05, 0.05],
                        rgba=[1, 1, 0, 0.9],  # Yellow
                        label=f"contact_R{rcontact}_L{lcontact}"
                    )
                
                # Add frame counter
                viewer.add_marker(
                    pos=hip_pos + np.array([0, 0, 0.3]),
                    size=[0.01, 0.01, 0.01],
                    rgba=[1, 1, 1, 1],  # White
                    label=f"Frame {tick+1}/{L}"
                )
                
                # Render the frame
                viewer.render()
                
                # Control playback speed
                if playback_speed > 0:
                    time.sleep(playback_speed)
            
            # Store data
            qpos_list.append(data.qpos.copy())
            tick += 1
            
            # Progress indicator
            if tick % 20 == 0:
                progress = (tick / L) * 100
                print(f"Progress: {progress:.1f}% ({tick}/{L} frames)")
                
        except Exception as e:
            print(f"Error in frame {tick}: {e}")
            break
    
    # Cleanup
    print("Closing viewer...")
    viewer.close()
    print("Standard viewer retargeting completed!")
    
    return {
        'qpos_list': qpos_list,
        'frames_processed': tick,
        'total_frames': L,
        'transformation_applied': {
            'scale': source_scale,
            'rotation_deg': source_rotation_deg,
            'translation': source_translation,
        }
    }

def simple_viewer_example():
    """
    Simple example showing basic mujoco_viewer usage
    """
    print("=== Simple mujoco_viewer Example ===")
    
    # Load model
    model = mujoco.MjModel.from_xml_path('../asset/unitree_g1/scene_g1.xml')
    data = mujoco.MjData(model)
    
    # Create viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Simple simulation loop
    for i in range(1000):
        if viewer.is_alive:
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Add some markers every 50 frames
            if i % 50 == 0:
                viewer.add_marker(
                    pos=[0, 0, 1.0],
                    size=[0.1, 0.1, 0.1],
                    rgba=[1, 0, 0, 0.8],
                    label=f"Marker {i//50}"
                )
            
            # Render
            viewer.render()
            time.sleep(0.01)
        else:
            break
    
    viewer.close()
    print("Simple viewer example completed!")

def camera_control_example():
    """
    Example showing camera control with standard viewer
    """
    print("=== Camera Control Example ===")
    
    model = mujoco.MjModel.from_xml_path('../asset/unitree_g1/scene_g1.xml')
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Camera control loop
    for i in range(500):
        if viewer.is_alive:
            # Rotate camera
            azimuth = (i * 0.5) % 360
            viewer.cam.azimuth = azimuth
            viewer.cam.distance = 3.0 + 0.5 * np.sin(i * 0.1)
            
            # Step and render
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.02)
        else:
            break
    
    viewer.close()
    print("Camera control example completed!")

if __name__ == "__main__":
    import fire
    fire.Fire({
        'standard': standard_viewer_retargeting,
        'simple': simple_viewer_example,
        'camera': camera_control_example,
    }) 