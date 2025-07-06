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
    bvh_path = "temp_mocap/bvh/88848d2067d622de8e4f314e28dc431a.bvh",
    p_rate = 0.056444*0.5,
    max_frames = 20,
    enable_viewer = True,
    render_every = 5,
    playback_speed = 0.3,
    repetition = 10,
    update_base_every_frame = True,
    safe_mode = True,
    do = 'both',  # 'robot', 'keypoint', or 'both'
    radians = (90, 0, 0),
    title = 'Motion Retargeting - Visual (Full Body)',
):

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
            rpy_order=[0,1,2],
            zup=True,
            plot_chain_graph=False,
            verbose=False,
        )
        secs, chains = secs[1:], chains[1:]

        # ------------------------------------------------------------
        # Apply a single global rotation to every chain if requested
        # ------------------------------------------------------------
        if radians is not None:
            for tick, chain in enumerate(chains):
                p_root, R_root = chain.get_root_joint_pR()
                T_root = pr2t(p_root,R_root)
                T_root_rot = pr2t(np.zeros(3), rpy2r(np.radians(radians))) @ T_root
                p_root_rot, R_root_rot = t2pr(T_root_rot)
                # FK chain
                chain_rot = deepcopy(chain)
                chain_rot.set_root_joint_pR(p=p_root_rot,R=R_root_rot)
                chain_rot.forward_kinematics()
                chains[tick] = chain_rot


        L = min(len(chains), max_frames)
        dt = secs[1] - secs[0] if len(secs) > 1 else 0.033
        HZ = int(1/dt)
        print(f"Processing {L} frames at {HZ} Hz...")

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
        
        # Calculate keypoint scaling and positioning for upright display
        if do in ['keypoint', 'both']:
            print("Calculating keypoint scaling for upright display...")
            # Get average hip height and scale to match robot height
            avg_hip_height = np.mean([t2p(get_T_joi_from_chain_cmu(chain)['hip'])[2] for chain in chains[:min(10, len(chains))]])
            keypoint_scale = 0.7 / avg_hip_height if avg_hip_height > 0 else 1.0  # Scale to ~70 cm robot height
            keypoint_offset = np.array([1.5, 0, 0.02])  # Offset to the right of robot

        else:
            keypoint_scale = 1.0
            keypoint_offset = np.array([0, 0, 0])
        
        if enable_viewer:
            env.init_viewer(
                title=title,
                transparent=False,
                backend='native'
            )
            print("Viewer initialized successfully")
            print("You should see the robot moving to match the motion data!")

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
                mode_desc = {'robot': 'robot motion', 'keypoint': 'keypoint display', 'both': 'combined visualization'}[do]
                print(f"Repetition {current_repetition+1}/{repetition} - Frame {frame_idx+1}/{L} - Processing {mode_desc}...")
                
                chain = chains[frame_idx]
                T_joi_src = get_T_joi_from_chain_cmu(chain, hip_between_pelvis=True)

                # Create upright keypoint positions for visualization
                if do in ['keypoint', 'both']:
                    T_joi_keypoint = {}
                    for joint_name, T in T_joi_src.items():
                        p_src = t2p(T)
                        p_keypoint = (p_src * keypoint_scale) + keypoint_offset
                        T_keypoint = T.copy()
                        T_keypoint[:3, 3]  = p_keypoint
                        T_joi_keypoint[joint_name] = T_keypoint

                # Robot motion retargeting (only when needed)
                if do in ['robot', 'both'] and (update_base_every_frame or tick == 0):
                    T_base_src = T_joi_src['hip']
                    p_base_src = t2p(T_base_src)
                    T_base_trgt = np.eye(4)
                    T_base_trgt[:3, 3] = [p_base_src[0], p_base_src[1], 0.702]
                    env.set_T_base_body(body_name='pelvis', T=T_base_trgt)
                    env.forward()

                # Initialize robot joint positions (only when needed)
                if do in ['robot', 'both']:
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
                else:
                    T_joi_trgt = None

                # Calculate IK targets (only when needed for robot)
                if do in ['robot', 'both']:
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

                    ik_converged = False
                    for ik_iter in range(30):
                        dq, ik_err_stack = get_dq_from_ik_info(
                            env=env,
                            ik_info=ik_info_full_body,
                            stepsize=0.3,
                            eps=1e-2,
                            th=np.radians(10.0),
                            joint_idxs_jac=joint_idxs_jac_full_body,
                        )
                        
                        qpos = env.get_qpos()
                        success, qpos_new = safe_mj_integratePos(env.model, qpos, dq, 1)
                        
                        if success:
                            env.forward(q=qpos_new)
                        else:
                            break
                        
                        if np.linalg.norm(ik_err_stack) < 0.1:
                            ik_converged = True
                            break

                if enable_viewer and (tick % render_every == 0):
                    if not env.is_viewer_alive():
                        break
                    
                    connections = [('hip', 'neck'), ('neck', 'rs'), ('rs', 'rw'), ('neck', 'ls'), ('ls', 'lw')]
                    
                    if do in ['keypoint', 'both']:
                        chain.plot_chain_mujoco(
                            env             = env,
                            plot_joint      = True,
                            plot_joint_name = True,
                            r_link          = 0.0025,
                            rgba_link       = (1,0,0,0.5),
                        ) # source link with red
                        for key in T_joi_src.keys(): 
                            env.plot_sphere(p=t2p(T_joi_src[key]),r=0.01,rgba=(1,0,0,0.5))
                    
                    if do in ['robot', 'both']:
                        for start_joint, end_joint in connections:
                            if start_joint in T_joi_trgt and end_joint in T_joi_trgt:
                                p_start = t2p(T_joi_trgt[start_joint])
                                p_end = t2p(T_joi_trgt[end_joint])
                                env.plot_line_fr2to(p_fr=p_start, p_to=p_end, rgba=(0, 0, 1, 0.4))
                        env.plot_links_between_bodies(r=0.0025,rgba=(0,0,1,0.5)) # target link with blue
                        plot_ik_info(env=env,ik_info=ik_info_full_body)

                    env.render()
                    if playback_speed > 0:
                        time.sleep(playback_speed)

                if tick % 10 == 0:
                    if do in ['robot', 'both']:
                        status = "converged" if ik_converged else "partial"
                        error_norm = np.linalg.norm(ik_err_stack) if 'ik_err_stack' in locals() else 0.0
                        print(f"  -> Frame {frame_idx+1}/{L}: IK {status}, Error: {error_norm:.4f}")
                    else:
                        print(f"  -> Frame {frame_idx+1}/{L}: Keypoint display")

                # Store data (only for robot mode)
                if do in ['robot', 'both']:
                    qpos_list.append(env.get_qpos())
                    R_ra_list.append(env.get_R_body(body_name='right_ankle_roll_link'))
                    R_la_list.append(env.get_R_body(body_name='left_ankle_roll_link'))
                else:
                    qpos_list.append(None)
                    R_ra_list.append(None)
                    R_la_list.append(None)
                
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
        
        if do == 'robot':
            print("Robot motion retargeting completed with contact detection!")
        elif do == 'keypoint':
            print("Keypoint visualization completed!")
        else:
            print("Combined robot and keypoint visualization completed with contact detection!")
        
        return {
            'mode': do,
            'qpos_list': qpos_list,
            'R_ra_list': R_ra_list,
            'R_la_list': R_la_list,
            'keypoint_scale': keypoint_scale if do in ['keypoint', 'both'] else None,
            'keypoint_offset': keypoint_offset.tolist() if do in ['keypoint', 'both'] else None,
        }
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        return None

def meta_rotate():
    rotations = [
       (0, 180, 0),
       (90, 0, 0),
    ]
    for radians in rotations:
        print(f"\n**** Testing rotation ({radians}) ****")
        main(radians=radians, title=f"Rotation {radians}")




if __name__ == "__main__":
    import fire
    fire.Fire({
        'main': main,
        'meta_rotate': meta_rotate,
    })
