import sys,mujoco
import numpy as np
import time
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
    max_frames = 10,
    enable_viewer = True,
    render_every = 1,
    playback_speed = 0.2,  # Slower by default
):
    print("Starting simple retargeting with basic rendering...")
    
    try:
        # Load environment
        env = MuJoCoParserClass(name='SimpleRetarget', rel_xml_path=xml_path, verbose=False)

        # Load BVH
        print("Loading BVH file...")
        secs, chains = get_chains_from_bvh_cmu_mocap(
            bvh_path=bvh_path,
            p_rate=p_rate,
            zup=True,
            plot_chain_graph=False,
            verbose=False,
        )
        secs, chains = secs[1:], chains[1:]  # exclude first frame
        L = min(len(chains), max_frames)
        print(f"Processing {L} frames...")

        # Initialize environment
        env.reset(step=True)
        
        # Initialize viewer with minimal setup
        if enable_viewer:
            print("Initializing viewer...")
            try:
                env.init_viewer(title='Simple Motion Retargeting', transparent=False)
                print("Viewer initialized successfully")
            except Exception as e:
                print(f"Viewer initialization failed: {e}")
                enable_viewer = False

        tick = 0
        qpos_list = []
        
        print("Starting retargeting loop...")
        
        while (not enable_viewer or env.is_viewer_alive()) and (tick < L):
            try:
                print(f"Processing frame {tick+1}/{L}")
                
                # Get source motion
                chain = chains[tick]
                T_joi_src = get_T_joi_from_chain_cmu(chain, hip_between_pelvis=True)

                # Update base position only once
                if tick == 0:
                    try:
                        T_base_src = T_joi_src['hip']
                        T_base_trgt = T_yuzf2zuxf(T_base_src)
                        if not (np.any(np.isnan(T_base_trgt)) or np.any(np.isinf(T_base_trgt))):
                            env.set_T_base_body(body_name='pelvis', T=T_base_trgt)
                            env.forward()
                    except Exception as e:
                        print(f"Error updating base: {e}")

                # Get target robot state
                T_joi_trgt = get_T_joi_from_g1(env)

                # Calculate link lengths
                len_neck2rs = len_T_joi(T_joi_trgt, 'hip', 'rs')  # Simplified: direct hip to shoulder
                len_rs2re = len_T_joi(T_joi_trgt, 'rs', 're')
                len_re2rw = len_T_joi(T_joi_trgt, 're', 'rw')
                len_neck2ls = len_T_joi(T_joi_trgt, 'hip', 'ls')  # Simplified: direct hip to shoulder
                len_ls2le = len_T_joi(T_joi_trgt, 'ls', 'le')
                len_le2lw = len_T_joi(T_joi_trgt, 'le', 'lw')

                # Get simplified unit vectors
                uv_neck2rs = uv_T_joi(T_joi_src, 'hip', 'rs')  # Simplified
                uv_rs2re = uv_T_joi(T_joi_src, 'rs', 're')
                uv_re2rw = uv_T_joi(T_joi_src, 're', 'rw')
                uv_neck2ls = uv_T_joi(T_joi_src, 'hip', 'ls')  # Simplified
                uv_ls2le = uv_T_joi(T_joi_src, 'ls', 'le')
                uv_le2lw = uv_T_joi(T_joi_src, 'le', 'lw')

                # Calculate target positions (simplified chain)
                p_hip_trgt = t2p(T_joi_src['hip'])
                p_rs_trgt = p_hip_trgt + len_neck2rs * uv_neck2rs
                p_re_trgt = p_rs_trgt + len_rs2re * uv_rs2re
                p_rw_trgt = p_re_trgt + len_re2rw * uv_re2rw
                p_ls_trgt = p_hip_trgt + len_neck2ls * uv_neck2ls
                p_le_trgt = p_ls_trgt + len_ls2le * uv_ls2le
                p_lw_trgt = p_le_trgt + len_le2lw * uv_le2lw

                # Set up simple IK (arms only for speed)
                joi_body_name = get_joi_body_name_of_g1()
                ik_info = init_ik_info()
                add_ik_info(ik_info, body_name=joi_body_name['rs'], p_trgt=p_rs_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['re'], p_trgt=p_re_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['ls'], p_trgt=p_ls_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['le'], p_trgt=p_le_trgt)

                # Simple IK solve with fewer joints
                arm_joints = [name for name in env.rev_joint_names if 'shoulder' in name or 'elbow' in name][:8]
                joint_idxs_jac = env.get_idxs_jac(joint_names=arm_joints)

                # Quick IK solve
                max_ik_iterations = 5  # Much fewer iterations
                for ik_iter in range(max_ik_iterations):
                    try:
                        dq, ik_err_stack = get_dq_from_ik_info(
                            env=env,
                            ik_info=ik_info,
                            stepsize=0.3,
                            eps=1e-1,  # Relaxed
                            th=np.radians(30.0),
                            joint_idxs_jac=joint_idxs_jac,
                        )
                        
                        qpos = env.get_qpos()
                        success, qpos_new = safe_mj_integratePos(env.model, qpos, dq, 1)
                        
                        if success:
                            env.forward(q=qpos_new)
                        else:
                            break
                            
                        if np.linalg.norm(ik_err_stack) < 0.2:
                            break
                            
                    except Exception as e:
                        print(f"IK error: {e}")
                        break

                # Simple rendering - no complex plotting
                if enable_viewer and (tick % render_every == 0):
                    try:
                        # Just render the basic scene
                        env.render()
                        
                        # Add timing control
                        if playback_speed > 0:
                            time.sleep(playback_speed)
                            
                    except Exception as e:
                        print(f"Rendering error: {e}")

                # Store results
                qpos_list.append(env.get_qpos())
                tick += 1
                
            except Exception as e:
                print(f"Error in frame {tick}: {e}")
                break

        # Close viewer safely
        if enable_viewer:
            try:
                print("Closing viewer...")
                env.close_viewer()
            except:
                pass
        
        print(f"Completed {len(qpos_list)} frames successfully")
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import fire
    fire.Fire(main) 