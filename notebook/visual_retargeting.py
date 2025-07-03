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
    max_frames = 20,
    enable_viewer = True,
    render_every = 1,
    playback_speed = 0.3,  # Good speed for observation
    show_targets = True,   # Show IK target spheres
):
    print("Starting visual retargeting with controlled rendering...")
    
    try:
        # Load environment
        env = MuJoCoParserClass(name='VisualRetarget', rel_xml_path=xml_path, verbose=False)

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
        
        # Initialize viewer
        if enable_viewer:
            print("Initializing viewer...")
            try:
                env.init_viewer(title='Motion Retargeting - Visual', transparent=False)
                print("Viewer initialized successfully")
                print("You should see the robot moving its arms to match the motion data!")
            except Exception as e:
                print(f"Viewer initialization failed: {e}")
                enable_viewer = False

        tick = 0
        qpos_list = []
        
        print("Starting retargeting loop...")
        if enable_viewer:
            print("Press ESC in the viewer window to stop")
        
        while (not enable_viewer or env.is_viewer_alive()) and (tick < L):
            try:
                print(f"Frame {tick+1}/{L} - Processing motion retargeting...")
                
                # Get source motion
                chain = chains[tick]
                T_joi_src = get_T_joi_from_chain_cmu(chain, hip_between_pelvis=True)

                # Update base position only once at start
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

                # Calculate full body link lengths
                len_hip2neck = len_T_joi(T_joi_trgt,'hip','neck')
                len_neck2rs = len_T_joi(T_joi_trgt,'neck','rs')
                len_rs2re = len_T_joi(T_joi_trgt,'rs','re')
                len_re2rw = len_T_joi(T_joi_trgt,'re','rw')
                len_neck2ls = len_T_joi(T_joi_trgt,'neck','ls')
                len_ls2le = len_T_joi(T_joi_trgt,'ls','le')
                len_le2lw = len_T_joi(T_joi_trgt,'le','lw')

                # Get unit vectors from source motion
                uv_hip2neck = uv_T_joi(T_joi_src,'hip','neck')
                uv_neck2rs = uv_T_joi(T_joi_src,'neck','rs')
                uv_rs2re = uv_T_joi(T_joi_src,'rs','re')
                uv_re2rw = uv_T_joi(T_joi_src,'re','rw')
                uv_neck2ls = uv_T_joi(T_joi_src,'neck','ls')
                uv_ls2le = uv_T_joi(T_joi_src,'ls','le')
                uv_le2lw = uv_T_joi(T_joi_src,'le','lw')

                # Calculate target positions using proper kinematic chain
                p_hip_trgt = t2p(T_joi_src['hip'])
                p_neck_trgt = p_hip_trgt + len_hip2neck*uv_hip2neck
                p_rs_trgt = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_re_trgt = p_rs_trgt + len_rs2re*uv_rs2re
                p_rw_trgt = p_re_trgt + len_re2rw*uv_re2rw
                p_ls_trgt = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_le_trgt = p_ls_trgt + len_ls2le*uv_ls2le
                p_lw_trgt = p_le_trgt + len_le2lw*uv_le2lw

                # Set up IK for arms (good balance of performance and visual result)
                joi_body_name = get_joi_body_name_of_g1()
                ik_info = init_ik_info()
                add_ik_info(ik_info, body_name=joi_body_name['rs'], p_trgt=p_rs_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['re'], p_trgt=p_re_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['rw'], p_trgt=p_rw_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['ls'], p_trgt=p_ls_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['le'], p_trgt=p_le_trgt)
                add_ik_info(ik_info, body_name=joi_body_name['lw'], p_trgt=p_lw_trgt)

                # IK solve with arm and torso joints
                arm_torso_joints = [name for name in env.rev_joint_names 
                                   if 'shoulder' in name or 'elbow' in name or 'torso' in name]
                joint_idxs_jac = env.get_idxs_jac(joint_names=arm_torso_joints)

                # IK iterations
                max_ik_iterations = 10
                ik_converged = False
                for ik_iter in range(max_ik_iterations):
                    try:
                        dq, ik_err_stack = get_dq_from_ik_info(
                            env=env,
                            ik_info=ik_info,
                            stepsize=0.5,
                            eps=1e-2,
                            th=np.radians(20.0),
                            joint_idxs_jac=joint_idxs_jac,
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
                            
                    except Exception as e:
                        print(f"IK error: {e}")
                        break

                # Rendering with minimal but useful visualization
                if enable_viewer and (tick % render_every == 0):
                    try:
                        # Show target positions as colored spheres if requested
                        if show_targets:
                            # Red spheres for targets
                            env.plot_sphere(p=p_rs_trgt, r=0.02, rgba=(1,0,0,0.7))  # Right shoulder
                            env.plot_sphere(p=p_re_trgt, r=0.02, rgba=(1,0,0,0.7))  # Right elbow
                            env.plot_sphere(p=p_rw_trgt, r=0.02, rgba=(1,0,0,0.7))  # Right wrist
                            env.plot_sphere(p=p_ls_trgt, r=0.02, rgba=(0,1,0,0.7))  # Left shoulder
                            env.plot_sphere(p=p_le_trgt, r=0.02, rgba=(0,1,0,0.7))  # Left elbow
                            env.plot_sphere(p=p_lw_trgt, r=0.02, rgba=(0,1,0,0.7))  # Left wrist
                        
                        # Render the scene
                        env.render()
                        
                        # Timing control
                        if playback_speed > 0:
                            time.sleep(playback_speed)
                            
                    except Exception as e:
                        print(f"Rendering error: {e}")

                # Print progress
                status = "converged" if ik_converged else "partial"
                print(f"  -> IK {status}")

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
        
        print(f"Completed {len(qpos_list)} frames successfully!")
        print("You should have seen the robot's arms moving to follow the motion capture data.")
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import fire
    fire.Fire(main) 