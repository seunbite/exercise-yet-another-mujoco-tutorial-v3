import sys,mujoco
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
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



def main(
    xml_path = '../asset/unitree_g1/scene_g1.xml',
    bvh_path = "../bvh/cmu_mocap/05_14.bvh",
    p_rate = 0.056444*1.2,
):
    # np.set_printoptions(precision=2,suppress=True,linewidth=100)
    plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
    env = MuJoCoParserClass(name='',rel_xml_path=xml_path,verbose=True)

    # Load CMU-MoCap
    secs, chains = get_chains_from_bvh_cmu_mocap(
        bvh_path         = bvh_path,
        p_rate           = 0.056444*1.2, # 1.0
        zup              = True,
        plot_chain_graph = False,
        verbose          = True,
    )
    secs,chains = secs[1:],chains[1:] # exlude the first frame (T-pose)
    L = len(chains)
    dt = secs[1]-secs[0]
    HZ = int(1/dt)

    # Toe and ankle trajectories
    rtoe_traj,ltoe_traj = np.zeros((L,3)),np.zeros((L,3))
    ra_traj,la_traj = np.zeros((L,3)),np.zeros((L,3))
    for tick in range(L):
        chain = chains[tick]
        T_joi = get_T_joi_from_chain_cmu(chain)
        rtoe_traj[tick,:],ltoe_traj[tick,:] = t2p(T_joi['rtoe']),t2p(T_joi['ltoe'])
        ra_traj[tick,:],la_traj[tick,:] = t2p(T_joi['ra']),t2p(T_joi['la'])
    # Get contact segments
    rcontact_segs,lcontact_segs = get_contact_segments(
        secs,
        rtoe_traj,
        ltoe_traj,
        zvel_th     = 0.1, # toe z velocity threshold to detect contact
        min_seg_sec = 0.1, # minimum segment time (to filter out false-positve contact segments)
        smt_sigma   = 5.0, # Gaussian smoothing sigma
        smt_radius  = 5.0, # Gaussian smoothing radius (filter size)
        verbose     = True,
        plot        = True,
    )
    rcontact_segs_concat = np.concatenate(rcontact_segs) # [M]
    lcontact_segs_concat = np.concatenate(lcontact_segs) # [M]
    for c_idx,rcontact_seg in enumerate(rcontact_segs):
        print ("Right Contact [%d/%d] [%d]~[%d]"%
            (c_idx,len(rcontact_segs),rcontact_seg[0],rcontact_seg[-1]))
    for c_idx,lcontact_seg in enumerate(lcontact_segs):
        print ("Left Contact [%d/%d] [%d]~[%d]"%
            (c_idx,len(lcontact_segs),lcontact_seg[0],lcontact_seg[-1]))    
    print ("Done.")

    env.reset(step=True)
    env.init_viewer(
        title='Motion Retargeting using Unit Vector Method',
        transparent=True)
    tick,render_every = 0,10
    qpos_list,R_ra_list,R_la_list = [],[],[]
    while env.is_viewer_alive() and (tick<L):
        # Get chain source
        chain = chains[tick]
        T_joi_src = get_T_joi_from_chain_cmu(chain,hip_between_pelvis=True)

        # Contact label
        rcontact = tick in rcontact_segs_concat
        lcontact = tick in lcontact_segs_concat

        # Move target base
        if tick == 0: # move base once
            T_base_src  = T_joi_src['hip']
            T_base_trgt = T_yuzf2zuxf(T_base_src)
            env.set_T_base_body(body_name='pelvis',T=T_base_trgt)
            env.forward()

        # Start motion retargeting
        T_joi_trgt = get_T_joi_from_g1(env)

        # Get body link lengths of the target rig
        len_hip2neck   = len_T_joi(T_joi_trgt,'hip','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
        len_hip2rp     = len_T_joi(T_joi_trgt,'hip','rp')
        len_rp2rk      = len_T_joi(T_joi_trgt,'rp','rk')
        len_rk2ra      = len_T_joi(T_joi_trgt,'rk','ra')
        len_hip2lp     = len_T_joi(T_joi_trgt,'hip','lp')
        len_lp2lk      = len_T_joi(T_joi_trgt,'lp','lk')
        len_lk2la      = len_T_joi(T_joi_trgt,'lk','la')
        
        # IK configuration for the full body
        rev_joint_names_for_ik_full_body = env.rev_joint_names
        joint_idxs_jac_full_body = env.get_idxs_jac(
            joint_names=rev_joint_names_for_ik_full_body)
        joint_idxs_jac_full_body_with_base = np.concatenate(
            ([0,1,2,3,4,5],joint_idxs_jac_full_body)) # add base free joints

        # Get unit vectors of the source rig
        uv_hip2neck   = uv_T_joi(T_joi_src,'hip','neck')
        uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
        uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
        uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
        uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
        uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
        uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
        uv_hip2rp     = uv_T_joi(T_joi_src,'hip','rp')
        uv_rp2rk      = uv_T_joi(T_joi_src,'rp','rk')
        uv_rk2ra      = uv_T_joi(T_joi_src,'rk','ra')
        uv_hip2lp     = uv_T_joi(T_joi_src,'hip','lp')
        uv_lp2lk      = uv_T_joi(T_joi_src,'lp','lk')
        uv_lk2la      = uv_T_joi(T_joi_src,'lk','la')
        
        # Set positional targets
        p_hip_trgt   = t2p(T_joi_src['hip'])
        p_neck_trgt  = p_hip_trgt + len_hip2neck*uv_hip2neck
        p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
        p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
        p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
        p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
        p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
        p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
        p_rp_trgt    = p_hip_trgt + len_hip2rp*uv_hip2rp
        p_rk_trgt    = p_rp_trgt + len_rp2rk*uv_rp2rk
        p_ra_trgt    = p_rk_trgt + len_rk2ra*uv_rk2ra
        p_lp_trgt    = p_hip_trgt + len_hip2lp*uv_hip2lp
        p_lk_trgt    = p_lp_trgt + len_lp2lk*uv_lp2lk
        p_la_trgt    = p_lk_trgt + len_lk2la*uv_lk2la

        # Set IK targets
        joi_body_name = get_joi_body_name_of_g1()
        ik_info_full_body = init_ik_info()
        add_ik_info(ik_info_full_body,body_name=joi_body_name['rs'],p_trgt=p_rs_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['re'],p_trgt=p_re_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['rw'],p_trgt=p_rw_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['ls'],p_trgt=p_ls_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['le'],p_trgt=p_le_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['lw'],p_trgt=p_lw_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['rp'],p_trgt=p_rp_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['rk'],p_trgt=p_rk_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['ra'],p_trgt=p_ra_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['lp'],p_trgt=p_lp_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['lk'],p_trgt=p_lk_trgt)
        add_ik_info(ik_info_full_body,body_name=joi_body_name['la'],p_trgt=p_la_trgt)

        # Solve IK
        max_ik_tick = 100
        for ik_tick in range(max_ik_tick): # ik loop
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info_full_body,
                stepsize       = 1,
                eps            = 1e-2,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_idxs_jac_full_body_with_base,
            ) # dq:[43]
            qpos = env.get_qpos() # get current joint position  [44]
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            if np.linalg.norm(ik_err_stack) < 0.05: break
                
        # Render
        T_joi_trgt = get_T_joi_from_g1(env)
        if (tick)%render_every == 0:
            env.plot_T()
            env.plot_time()
            chain.plot_chain_mujoco(
                env             = env,
                plot_joint      = False,
                plot_joint_name = False,
                r_link          = 0.0025,
                rgba_link       = (1,0,0,0.5),
            ) # source link with red
            env.plot_links_between_bodies(r=0.0025,rgba=(0,0,1,0.5)) # target link with blue
            for key in T_joi_src.keys(): 
                env.plot_sphere(p=t2p(T_joi_src[key]),r=0.01,rgba=(1,0,0,0.5))
            for key in T_joi_trgt.keys():
                env.plot_sphere(p=t2p(T_joi_trgt[key]),r=0.01,rgba=(0,0,1,0.5))
            env.plot_body_T(body_name='right_ankle_roll_link')
            env.plot_body_T(body_name='left_ankle_roll_link')
            plot_ik_info(env=env,ik_info=ik_info_full_body)
            env.render()
        
        # Increase tick
        tick = tick + 1

        # Append joint position and ankle lists 
        qpos_list.append(env.get_qpos())
        R_ra_list.append(env.get_R_body(body_name='right_ankle_roll_link'))
        R_la_list.append(env.get_R_body(body_name='left_ankle_roll_link'))
    # Check
    assert (len(qpos_list)==L), "len(qpos_list):[%d] == L:[%d]"%(len(qpos_list),L)
    # Close viewer
    env.close_viewer()  
    print ("Done.")


if __name__ == "__main__":
    main()