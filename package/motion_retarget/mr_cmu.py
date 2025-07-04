import numpy as np
""" 
    Assume that the main notebook called 'sys.path.append('../../package/helper/')'
"""
from transformation import pr2t

def get_T_joi_from_chain_cmu(chain,hip_between_pelvis=True):
    """ 
        Get joints of interest of CMU mocap chain (IMPROVED MAPPING)
    """
    p_hip,R_hip = chain.get_joint_pR(joint_name='Hips')
    p_spine,R_spine = chain.get_joint_pR(joint_name='Spine')
    p_spine1,R_spine1 = chain.get_joint_pR(joint_name='Spine1')
    
    # Add missing shoulder joints - these were previously skipped!
    p_rsh,R_rsh = chain.get_joint_pR(joint_name='RightShoulder')
    p_lsh,R_lsh = chain.get_joint_pR(joint_name='LeftShoulder')

    p_rs,R_rs = chain.get_joint_pR(joint_name='RightArm')
    p_re,R_re = chain.get_joint_pR(joint_name='RightForeArm')
    # Better wrist mapping - create a virtual wrist position slightly beyond forearm
    wrist_offset = 0.1  # 10cm beyond forearm in the direction of arm extension
    arm_direction_r = (p_re - p_rs)
    if np.linalg.norm(arm_direction_r) > 1e-6:
        arm_direction_r = arm_direction_r / np.linalg.norm(arm_direction_r)
        p_rw = p_re + wrist_offset * arm_direction_r
    else:
        p_rw = p_re
    R_rw = R_re  # Use forearm rotation for wrist

    p_ls,R_ls = chain.get_joint_pR(joint_name='LeftArm')
    p_le,R_le = chain.get_joint_pR(joint_name='LeftForeArm')
    # Better wrist mapping - create a virtual wrist position slightly beyond forearm
    arm_direction_l = (p_le - p_ls)
    if np.linalg.norm(arm_direction_l) > 1e-6:
        arm_direction_l = arm_direction_l / np.linalg.norm(arm_direction_l)
        p_lw = p_le + wrist_offset * arm_direction_l
    else:
        p_lw = p_le
    R_lw = R_le  # Use forearm rotation for wrist

    p_neck,R_neck = chain.get_joint_pR(joint_name='Neck')
    # Better neck calculation using shoulders and spine1
    p_neck_shoulder_center = 0.5*(p_rsh+p_lsh)  # Use actual shoulder positions
    p_neck = np.array([p_neck_shoulder_center[0], p_neck_shoulder_center[1], p_neck[2]])

    p_head,R_head = chain.get_joint_pR(joint_name='Head')

    # Leg joints
    p_rp,R_rp = chain.get_joint_pR(joint_name='RightUpLeg')
    p_rk,R_rk = chain.get_joint_pR(joint_name='RightLeg')
    p_ra,R_ra = chain.get_joint_pR(joint_name='RightFoot')

    p_lp,R_lp = chain.get_joint_pR(joint_name='LeftUpLeg')
    p_lk,R_lk = chain.get_joint_pR(joint_name='LeftLeg')
    p_la,R_la = chain.get_joint_pR(joint_name='LeftFoot')

    p_r_toe,R_r_toe = chain.get_joint_pR(joint_name='RightFoot')  
    p_l_toe,R_l_toe = chain.get_joint_pR(joint_name='LeftFoot')  

    # Modify Hip positions
    if hip_between_pelvis:
        p_hip = 0.5*(p_rp+p_lp)
        p_hip_z = 0.5*(p_rp+p_lp) # z hip position to be the center of two pelvis positions
        p_hip = np.array([p_hip[0],p_hip[1],p_hip_z[2]])
    else:
        p_hip = p_hip

    # Better torso mapping - interpolate between hip and spine1
    torso_ratio = 0.3  # 30% up from hip to spine1
    p_torso = p_hip + torso_ratio * (p_spine1 - p_hip)
    R_torso = R_spine1

    T_joi = {
        'hip': pr2t(p_hip,R_hip),
        'spine': pr2t(p_spine,R_spine),
        'spine1': pr2t(p_spine1,R_spine1),
        'torso': pr2t(p_torso,R_torso),  # Better torso position
        'rsh': pr2t(p_rsh,R_rsh),      # Added missing right shoulder
        'lsh': pr2t(p_lsh,R_lsh),      # Added missing left shoulder
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),         # Improved wrist mapping
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),         # Improved wrist mapping
        'neck':pr2t(p_neck,R_neck),
        'head':pr2t(p_head,R_head),
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
        'rtoe': pr2t(p_r_toe,R_r_toe),
        'ltoe': pr2t(p_l_toe,R_l_toe),
    }
    return T_joi

