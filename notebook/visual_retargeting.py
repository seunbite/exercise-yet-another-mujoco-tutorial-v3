import sys,mujoco
import numpy as np
import time
import pickle
import os
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/bvh_parser/')
sys.path.append('../package/motion_retarget/')
from mujoco_parser import *
from utility import *
from transformation import *
from bvh_parser import *
from joi import *
from mr_cmu import *

def safe_mj_integratePos(model, qpos, dq, dt):
    """Safe wrapper for mj_integratePos with error checking"""
    try:
        if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
            return False, qpos
        if np.any(np.isnan(dq)) or np.any(np.isinf(dq)):
            return False, qpos
        
        qpos_new = qpos.copy()
        mujoco.mj_integratePos(model, qpos_new, dq, dt)
        
        if np.any(np.isnan(qpos_new)) or np.any(np.isinf(qpos_new)):
            return False, qpos
            
        return True, qpos_new
        
    except Exception as e:
        print(f"Exception in safe_mj_integratePos: {e}")
        return False, qpos

def main(
    xml_path = '../asset/unitree_g1/scene_g1.xml',
    bvh_path = "temp_mocap/bvh/58f23c62733b75711785a822cdb052c6.bvh",
    p_rate = 0.056444*1.2,
    max_frames = 20,
    enable_viewer = True,
    render_every = 1,
    playback_speed = 0.3,
    show_targets = True,
    repetition = 10,
    update_base_every_frame = True,
    safe_mode = True,
):
    """
    Full-body motion retargeting with 3D keypoint visualization.
    
    VISUALIZATION COLORS:
    - RED spheres: Source motion keypoints from BVH data
    - BLUE spheres: Current robot joint positions
    - GREEN spheres: IK target positions (if show_targets=True)
    - YELLOW spheres: Contact indicators for feet
    - RED lines: Connections between source keypoints
    - GREEN lines: Lines from robot joints to IK targets
    
    Use this to debug joint mapping by comparing RED (source) vs BLUE (robot) positions.
    """
    print("Starting visual retargeting with controlled rendering...")
    print("VISUALIZATION LEGEND:")
    print("  🔴 RED spheres/lines: Source motion keypoints (BVH data)")
    print("  🔵 BLUE spheres: Robot joint positions")
    print("  🟢 GREEN spheres/lines: IK targets and error vectors")
    print("  🟡 YELLOW spheres: Foot contact indicators")
    if safe_mode:
        print("SAFE MODE ENABLED: Using conservative settings to prevent segfaults")
    
    try:
        ########################################
        # SETUP AND INITIALIZATION
        ########################################
        env = MuJoCoParserClass(name='VisualRetarget', rel_xml_path=xml_path, verbose=False)

        ########################################
        # LOAD BVH AND CALCULATE CONTACT DETECTION
        ########################################
        print("Loading BVH file...")
        secs, chains = get_chains_from_bvh_cmu_mocap(
            bvh_path=bvh_path,
            p_rate=p_rate,
            zup=True,
            plot_chain_graph=False,
            verbose=False,
        )
        secs, chains = secs[1:], chains[1:]
        L = min(len(chains), max_frames)
        dt = secs[1] - secs[0] if len(secs) > 1 else 0.033
        HZ = int(1/dt)
        print(f"Processing {L} frames at {HZ} Hz...")

        print("Calculating contact detection...")
        rtoe_traj, ltoe_traj = np.zeros((L, 3)), np.zeros((L, 3))
        ra_traj, la_traj = np.zeros((L, 3)), np.zeros((L, 3))
        
        for tick in range(L):
            chain = chains[tick]
            T_joi = get_T_joi_from_chain_cmu(chain)
            rtoe_traj[tick, :], ltoe_traj[tick, :] = t2p(T_joi['rtoe']), t2p(T_joi['ltoe'])
            ra_traj[tick, :], la_traj[tick, :] = t2p(T_joi['ra']), t2p(T_joi['la'])
            
        rcontact_segs, lcontact_segs = get_contact_segments(
            secs,
            rtoe_traj,
            ltoe_traj,
            zvel_th=0.1,
            min_seg_sec=0.1,
            smt_sigma=5.0,
            smt_radius=5.0,
            verbose=False,
            plot=False,
        )
        rcontact_segs_concat = np.concatenate(rcontact_segs) if rcontact_segs else np.array([])
        lcontact_segs_concat = np.concatenate(lcontact_segs) if lcontact_segs else np.array([])
        
        print(f"Found {len(rcontact_segs)} right contact segments and {len(lcontact_segs)} left contact segments")

        ########################################
        # ENVIRONMENT SETUP
        ########################################
        env.reset(step=True)
        
        if safe_mode:
            print("Testing basic functionality...")
            try:
                test_qpos = env.get_qpos()
                env.forward(q=test_qpos)
                print("Basic functionality test passed")
            except Exception as e:
                print(f"Basic functionality test failed: {e}")
                return None
        
        if enable_viewer:
            print("Initializing viewer...")
            try:
                env.init_viewer(
                    title='Motion Retargeting - Visual (Full Body)',
                    transparent=False
                )
                print("Viewer initialized successfully")
                print("You should see the robot moving to match the motion data!")
            except Exception as e:
                print(f"Viewer initialization failed: {e}")
                enable_viewer = False

        ########################################
        # MAIN RETARGETING LOOP
        ########################################
        tick = 0
        current_repetition = 0
        total_frames_processed = 0
        qpos_list, R_ra_list, R_la_list = [], [], []
        
        print("Starting retargeting loop...")
        if enable_viewer:
            print("Press ESC in the viewer window to stop")
        print(f"Animation will repeat {repetition} time(s)")
        
        while (not enable_viewer or env.is_viewer_alive()) and (current_repetition < repetition):
            try:
                frame_idx = tick % L
                print(f"Repetition {current_repetition+1}/{repetition} - Frame {frame_idx+1}/{L} - Processing full-body retargeting...")
                
                chain = chains[frame_idx]
                T_joi_src = get_T_joi_from_chain_cmu(chain, hip_between_pelvis=True)

                rcontact = frame_idx in rcontact_segs_concat
                lcontact = frame_idx in lcontact_segs_concat

                if update_base_every_frame or tick == 0:
                    try:
                        T_base_src = T_joi_src['hip']
                        
                        if safe_mode:
                            p_base_src = t2p(T_base_src)
                            T_base_trgt = np.eye(4)
                            T_base_trgt[:3, 3] = [p_base_src[0], p_base_src[1], 0.702]
                        else:
                            T_base_trgt = T_yuzf2zuxf(T_base_src)
                        
                        if np.any(np.isnan(T_base_trgt)) or np.any(np.isinf(T_base_trgt)):
                            print(f"Warning: Invalid transformation at frame {frame_idx}, skipping base update")
                        else:
                            env.set_T_base_body(body_name='pelvis', T=T_base_trgt)
                            env.forward()
                    except Exception as e:
                        print(f"Error updating base at frame {frame_idx}: {e}")
                        if safe_mode:
                            continue
                        else:
                            break

                T_joi_trgt = get_T_joi_from_g1(env)

                len_hip2neck = len_T_joi(T_joi_trgt, 'hip', 'neck')
                len_neck2rs = len_T_joi(T_joi_trgt, 'neck', 'rs')
                len_rs2re = len_T_joi(T_joi_trgt, 'rs', 're')
                len_re2rw = len_T_joi(T_joi_trgt, 're', 'rw')
                len_neck2ls = len_T_joi(T_joi_trgt, 'neck', 'ls')
                len_ls2le = len_T_joi(T_joi_trgt, 'ls', 'le')
                len_le2lw = len_T_joi(T_joi_trgt, 'le', 'lw')
                len_hip2rp = len_T_joi(T_joi_trgt, 'hip', 'rp')
                len_rp2rk = len_T_joi(T_joi_trgt, 'rp', 'rk')
                len_rk2ra = len_T_joi(T_joi_trgt, 'rk', 'ra')
                len_hip2lp = len_T_joi(T_joi_trgt, 'hip', 'lp')
                len_lp2lk = len_T_joi(T_joi_trgt, 'lp', 'lk')
                len_lk2la = len_T_joi(T_joi_trgt, 'lk', 'la')

                rev_joint_names_for_ik_full_body = env.rev_joint_names
                joint_idxs_jac_full_body = env.get_idxs_jac(joint_names=rev_joint_names_for_ik_full_body)

                uv_hip2neck = uv_T_joi(T_joi_src, 'hip', 'neck')
                uv_neck2rs = uv_T_joi(T_joi_src, 'neck', 'rs')
                uv_rs2re = uv_T_joi(T_joi_src, 'rs', 're')
                uv_re2rw = uv_T_joi(T_joi_src, 're', 'rw')
                uv_neck2ls = uv_T_joi(T_joi_src, 'neck', 'ls')
                uv_ls2le = uv_T_joi(T_joi_src, 'ls', 'le')
                uv_le2lw = uv_T_joi(T_joi_src, 'le', 'lw')
                uv_hip2rp = uv_T_joi(T_joi_src, 'hip', 'rp')
                uv_rp2rk = uv_T_joi(T_joi_src, 'rp', 'rk')
                uv_rk2ra = uv_T_joi(T_joi_src, 'rk', 'ra')
                uv_hip2lp = uv_T_joi(T_joi_src, 'hip', 'lp')
                uv_lp2lk = uv_T_joi(T_joi_src, 'lp', 'lk')
                uv_lk2la = uv_T_joi(T_joi_src, 'lk', 'la')

                try:
                    if update_base_every_frame:
                        T_joi_trgt_current = get_T_joi_from_g1(env)
                        p_hip_current = t2p(T_joi_trgt_current['hip'])
                        p_hip_trgt = p_hip_current
                    else:
                        p_hip_trgt = t2p(T_joi_src['hip'])

                    p_neck_trgt = p_hip_trgt + len_hip2neck * uv_hip2neck
                    p_rs_trgt = p_neck_trgt + len_neck2rs * uv_neck2rs
                    p_re_trgt = p_rs_trgt + len_rs2re * uv_rs2re
                    p_rw_trgt = p_re_trgt + len_re2rw * uv_re2rw
                    p_ls_trgt = p_neck_trgt + len_neck2ls * uv_neck2ls
                    p_le_trgt = p_ls_trgt + len_ls2le * uv_ls2le
                    p_lw_trgt = p_le_trgt + len_le2lw * uv_le2lw
                    p_rp_trgt = p_hip_trgt + len_hip2rp * uv_hip2rp
                    p_rk_trgt = p_rp_trgt + len_rp2rk * uv_rp2rk
                    p_ra_trgt = p_rk_trgt + len_rk2ra * uv_rk2ra
                    p_lp_trgt = p_hip_trgt + len_hip2lp * uv_hip2lp
                    p_lk_trgt = p_lp_trgt + len_lp2lk * uv_lp2lk
                    p_la_trgt = p_lk_trgt + len_lk2la * uv_lk2la

                    target_positions = [p_neck_trgt, p_rs_trgt, p_re_trgt, p_rw_trgt,
                                      p_ls_trgt, p_le_trgt, p_lw_trgt, p_rp_trgt,
                                      p_rk_trgt, p_ra_trgt, p_lp_trgt, p_lk_trgt, p_la_trgt]

                    invalid_positions = False
                    for i, pos in enumerate(target_positions):
                        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                            print(f"Warning: Invalid target position {i} at frame {frame_idx}, skipping frame")
                            invalid_positions = True
                            break

                    if invalid_positions:
                        continue

                except Exception as e:
                    print(f"Error calculating target positions at frame {frame_idx}: {e}")
                    continue

                joi_body_name = get_joi_body_name_of_g1()
                ik_info_full_body = init_ik_info()
                
                add_ik_info(ik_info_full_body, body_name=joi_body_name['rs'], p_trgt=p_rs_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['re'], p_trgt=p_re_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['rw'], p_trgt=p_rw_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['ls'], p_trgt=p_ls_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['le'], p_trgt=p_le_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['lw'], p_trgt=p_lw_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['rp'], p_trgt=p_rp_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['rk'], p_trgt=p_rk_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['ra'], p_trgt=p_ra_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['lp'], p_trgt=p_lp_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['lk'], p_trgt=p_lk_trgt)
                add_ik_info(ik_info_full_body, body_name=joi_body_name['la'], p_trgt=p_la_trgt)

                max_ik_iterations = 30 if safe_mode else 50
                ik_converged = False
                
                try:
                    for ik_iter in range(max_ik_iterations):
                        dq, ik_err_stack = get_dq_from_ik_info(
                            env=env,
                            ik_info=ik_info_full_body,
                            stepsize=0.3 if safe_mode else 0.5,
                            eps=1e-2,
                            th=np.radians(10.0),
                            joint_idxs_jac=joint_idxs_jac_full_body,
                        )
                        
                        if np.any(np.isnan(dq)) or np.any(np.isinf(dq)):
                            if safe_mode:
                                print(f"Warning: Invalid dq at frame {frame_idx}, ik_iter {ik_iter} - continuing in safe mode")
                                break
                            else:
                                print(f"Warning: Invalid dq at frame {frame_idx}, ik_iter {ik_iter}")
                                break
                        
                        qpos = env.get_qpos()
                        qpos_backup = qpos.copy()
                        
                        success, qpos_new = safe_mj_integratePos(env.model, qpos, dq, 1)
                        
                        if success:
                            if safe_mode:
                                if np.any(np.abs(qpos_new) > 10):
                                    print(f"Warning: Extreme joint values detected, skipping integration")
                                    env.forward(q=qpos_backup)
                                    break
                            env.forward(q=qpos_new)
                        else:
                            env.forward(q=qpos_backup)
                            break
                        
                        convergence_threshold = 0.1 if safe_mode else 0.05
                        if np.linalg.norm(ik_err_stack) < convergence_threshold:
                            ik_converged = True
                            break
                            
                except Exception as e:
                    print(f"Critical error in IK solver at frame {frame_idx}: {e}")
                    if safe_mode:
                        print("Continuing in safe mode...")
                        continue
                    else:
                        break

                ########################################
                # RENDERING
                ########################################
                if enable_viewer and (tick % render_every == 0):
                    try:
                        T_joi_trgt = get_T_joi_from_g1(env)
                        
                        env.plot_T()
                        
                        key_joints = ['hip', 'neck', 'rs', 're', 'rw', 'ls', 'le', 'lw', 'rp', 'rk', 'ra', 'lp', 'lk', 'la']
                        for key in key_joints:
                            if key in T_joi_src:
                                env.plot_sphere(p=t2p(T_joi_src[key]), r=0.015, rgba=(1, 0, 0, 0.7))
                            if key in T_joi_trgt:
                                env.plot_sphere(p=t2p(T_joi_trgt[key]), r=0.010, rgba=(0, 0, 1, 0.7))
                        
                        if show_targets:
                            env.plot_sphere(p=p_rw_trgt, r=0.02, rgba=(1, 0, 0, 0.8))
                            env.plot_sphere(p=p_lw_trgt, r=0.02, rgba=(0, 1, 0, 0.8))
                            env.plot_sphere(p=p_ra_trgt, r=0.02, rgba=(1, 1, 0, 0.8))
                            env.plot_sphere(p=p_la_trgt, r=0.02, rgba=(1, 1, 0, 0.8))
                        
                        if rcontact:
                            env.plot_sphere(p=p_ra_trgt + np.array([0, 0, -0.05]), r=0.025, rgba=(1, 1, 0, 1.0))
                        if lcontact:
                            env.plot_sphere(p=p_la_trgt + np.array([0, 0, -0.05]), r=0.025, rgba=(1, 1, 0, 1.0))
                        
                        env.render()
                        
                        if playback_speed > 0:
                            time.sleep(playback_speed)
                            
                    except Exception as e:
                        print(f"Rendering error: {e}")

                if tick % 10 == 0:
                    status = "converged" if ik_converged else "partial"
                    error_norm = np.linalg.norm(ik_err_stack) if 'ik_err_stack' in locals() else 0.0
                    print(f"  -> Frame {frame_idx+1}/{L}: IK {status}, Error: {error_norm:.4f}, Contacts: R={rcontact}, L={lcontact}")

                qpos_list.append(env.get_qpos())
                R_ra_list.append(env.get_R_body(body_name='right_ankle_roll_link'))
                R_la_list.append(env.get_R_body(body_name='left_ankle_roll_link'))
                
                tick += 1
                total_frames_processed += 1
                
                if tick % L == 0:
                    current_repetition += 1
                    if current_repetition < repetition:
                        print(f"Completed repetition {current_repetition}/{repetition}. Starting next repetition...")
                
            except Exception as e:
                print(f"Error in frame {frame_idx}: {e}")
                break

        ########################################
        # CLEANUP AND RETURN
        ########################################
        if enable_viewer:
            try:
                print("Closing viewer...")
                env.close_viewer()
            except:
                pass
        
        expected_frames = min(L * repetition, total_frames_processed)
        if len(qpos_list) != expected_frames:
            print(f"Warning: Expected {expected_frames} frames but got {len(qpos_list)}")
        
        print(f"Completed {total_frames_processed} frames successfully!")
        print(f"Animation repeated {current_repetition} time(s) successfully!")
        print("Full-body motion retargeting completed with contact detection!")
        
        return {
            'qpos_list': qpos_list,
            'R_ra_list': R_ra_list,
            'R_la_list': R_la_list,
            'contact_info': {
                'rcontact_segs': rcontact_segs,
                'lcontact_segs': lcontact_segs,
            }
        }
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        return None

def just_bvh(
    xml_path = '../asset/unitree_g1/scene_g1.xml',
    keypoints_path = "temp_mocap/keypoints_raw.pkl",
    max_frames = 50,
    enable_viewer = True,
    render_every = 1,
    playback_speed = 0.3,
    scale_factor = 2.0,  # Scale factor for keypoints
    show_keypoints = True,
    show_connections = True,
    safe_mode = True,  # Add safety measures for viewer
    repetition = 10,  # Number of times to repeat the animation
    simple_rendering = False,  # Use simplified rendering in safe mode
):
    """
    Load raw 3D keypoints directly and visualize them in the simulator without BVH retargeting.
    This shows the same data as the 3D visualization GIF.
    
    Args:
        repetition (int): Number of times to repeat the animation sequence (default: 1)
    """
    print("Starting raw 3D keypoint visualization...")
    
    # MediaPipe landmark names
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
        'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
        'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]
    
    # MediaPipe pose connections for visualization
    POSE_CONNECTIONS = [
        # Face connections
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # Face outline
        (9, 10),  # Mouth
        # Body connections
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Shoulder to hip
        (23, 24),  # Hips
        # Legs
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        # Feet
        (27, 29), (27, 31),  # Left foot
        (28, 30), (28, 32),  # Right foot
    ]
    
    try:
        # Load raw 3D keypoints
        print(f"Loading raw 3D keypoints from {keypoints_path}")
        if not os.path.exists(keypoints_path):
            print(f"Error: Keypoints file not found: {keypoints_path}")
            print("Please run retarget_data.py first to generate keypoints!")
            return
        
        with open(keypoints_path, 'rb') as f:
            keypoints_3d = pickle.load(f)
        
        print(f"Loaded {len(keypoints_3d)} frames of 3D keypoints")
        L = min(len(keypoints_3d), max_frames)
        
        # Load environment
        env = MuJoCoParserClass(name='RawKeypoints', rel_xml_path=xml_path, verbose=False)
        env.reset(step=True)
        
        # Initialize viewer with safety measures
        if enable_viewer:
            print("Initializing viewer...")
            try:
                # Add safety delay and check
                import time
                time.sleep(0.1)
                
                # Use safer viewer settings
                viewer_settings = {
                    'title': 'Raw 3D Keypoints Visualization',
                    'transparent': False,
                    'width': 800,
                    'height': 600,
                    'maxgeom': 5000,  # Reduce max geometry count
                }
                
                if safe_mode:
                    # Additional safe settings
                    viewer_settings.update({
                        'azimuth': 170,
                        'distance': 5.0,
                        'elevation': -20,
                        'lookat': [0, 0, 1],
                    })
                    print("Safe mode enabled: using conservative viewer settings")
                
                env.init_viewer(**viewer_settings)
                print("Viewer initialized successfully")
                print("You should see the raw 3D keypoints from MediaPipe!")
                
                # Test render to ensure stability
                env.render()
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Viewer initialization failed: {e}")
                import traceback
                traceback.print_exc()
                enable_viewer = False
        
        tick = 0
        current_repetition = 0
        total_frames_processed = 0
        
        print("Starting keypoint visualization loop...")
        if enable_viewer:
            print("Press ESC in the viewer window to stop")
        print(f"Animation will repeat {repetition} time(s)")
        
        while (not enable_viewer or env.is_viewer_alive()) and (current_repetition < repetition):
            try:
                frame_idx = tick % L  # Use modulo to cycle through frames
                print(f"Repetition {current_repetition+1}/{repetition} - Frame {frame_idx+1}/{L} - Displaying raw 3D keypoints...")
                
                # Get current frame keypoints
                keypoints = keypoints_3d[frame_idx]
                
                if keypoints is not None and len(keypoints) >= 33:
                    # Transform keypoints to simulator coordinates
                    # MediaPipe: X=horizontal, Y=down, Z=depth
                    # Simulator: X=horizontal, Y=depth, Z=up
                    keypoints_sim = np.zeros_like(keypoints)
                    keypoints_sim[:, 0] = (keypoints[:, 0] - 0.5) * scale_factor  # Center X
                    keypoints_sim[:, 1] = keypoints[:, 2] * scale_factor  # Y = original Z (depth)
                    keypoints_sim[:, 2] = (1 - keypoints[:, 1]) * scale_factor  # Z = 1 - original Y (flip Y to make head up)
                    
                    # Rendering with safety measures
                    if enable_viewer and (tick % render_every == 0):
                        try:
                            # Check viewer is still alive before rendering
                            if not env.is_viewer_alive():
                                print("Viewer closed, stopping...")
                                break
                            
                            # Draw keypoints as spheres
                            if show_keypoints:
                                for i, (keypoint, name) in enumerate(zip(keypoints_sim, LANDMARK_NAMES)):
                                    if not np.allclose(keypoints[i], 0):  # Skip zero keypoints
                                        # Validate keypoint coordinates
                                        if np.any(np.isnan(keypoint)) or np.any(np.isinf(keypoint)):
                                            continue
                                            
                                        # Different colors for different body parts
                                        if 'face' in name or 'eye' in name or 'ear' in name or 'nose' in name or 'mouth' in name:
                                            color = (1, 0, 0, 0.8)  # Red for face
                                        elif 'shoulder' in name or 'elbow' in name or 'wrist' in name or 'thumb' in name or 'index' in name or 'pinky' in name:
                                            color = (0, 1, 0, 0.8)  # Green for arms
                                        elif 'hip' in name or 'knee' in name or 'ankle' in name or 'heel' in name or 'foot' in name:
                                            color = (0, 0, 1, 0.8)  # Blue for legs
                                        else:
                                            color = (1, 1, 0, 0.8)  # Yellow for others
                                        
                                        # Larger sphere for key landmarks
                                        radius = 0.05 if i in [0, 11, 12, 23, 24] else 0.03
                                        try:
                                            env.plot_sphere(p=keypoint, r=radius, rgba=color)
                                        except Exception as sphere_e:
                                            if safe_mode:
                                                print(f"Sphere rendering error for {name}: {sphere_e}")
                                            continue
                            
                            # Draw connections
                            if show_connections:
                                for connection in POSE_CONNECTIONS:
                                    start_idx, end_idx = connection
                                    if (start_idx < len(keypoints_sim) and end_idx < len(keypoints_sim) and
                                        not np.allclose(keypoints[start_idx], 0) and not np.allclose(keypoints[end_idx], 0)):
                                        
                                        start_point = keypoints_sim[start_idx]
                                        end_point = keypoints_sim[end_idx]
                                        
                                        # Validate connection points
                                        if (np.any(np.isnan(start_point)) or np.any(np.isinf(start_point)) or
                                            np.any(np.isnan(end_point)) or np.any(np.isinf(end_point))):
                                            continue
                                        
                                        # Draw line between keypoints
                                        try:
                                            env.plot_line_fr2to(p_fr=start_point, p_to=end_point, rgba=(0.5, 0.5, 0.5, 0.6))
                                        except Exception as line_e:
                                            if safe_mode:
                                                print(f"Line rendering error: {line_e}")
                                            continue
                            
                            # Add frame info (skip text rendering in safe mode to prevent segfaults)
                            if not safe_mode:
                                try:
                                    hip_center = (keypoints_sim[23] + keypoints_sim[24]) / 2 if not np.allclose(keypoints[23], 0) and not np.allclose(keypoints[24], 0) else np.array([0, 0, 1])
                                    if not (np.any(np.isnan(hip_center)) or np.any(np.isinf(hip_center))):
                                        env.plot_text(p=hip_center + np.array([0, 0, 0.5]), label=f"Rep {current_repetition+1}/{repetition} - Frame {frame_idx+1}/{L}")
                                except Exception as text_e:
                                    print(f"Text rendering error: {text_e}")
                            
                            # Render the scene with safety check
                            try:
                                env.render()
                            except Exception as render_e:
                                print(f"Render error: {render_e}")
                                if safe_mode:
                                    print("Render failed, continuing...")
                                    # Don't disable viewer, just continue with timing
                                    pass
                                else:
                                    raise
                            
                            # Timing control
                            if playback_speed > 0:
                                time.sleep(playback_speed)
                                
                        except Exception as e:
                            print(f"Rendering error: {e}")
                            if safe_mode:
                                print("Continuing in safe mode...")
                                # Don't break the loop in safe mode, just continue
                                continue
                            else:
                                import traceback
                                traceback.print_exc()
                                break
                else:
                    print(f"  -> Invalid keypoints data in frame {frame_idx+1}")
                
                tick += 1
                total_frames_processed += 1
                
                # Check if we've completed all frames in current repetition
                if tick % L == 0:
                    current_repetition += 1
                    if current_repetition < repetition:
                        print(f"Completed repetition {current_repetition}/{repetition}. Starting next repetition...")
                
            except Exception as e:
                print(f"Error in frame {frame_idx}: {e}")
                break
        
        # Close viewer safely
        if enable_viewer:
            try:
                print("Closing viewer...")
                env.close_viewer()
            except:
                pass
        
        print(f"Completed visualization of {total_frames_processed} frames ({current_repetition} repetitions)!")
        print("This shows the same raw 3D keypoints as in the visualization GIF.")
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import fire
    fire.Fire({
        'main': main,
        'just_bvh': just_bvh,
    })