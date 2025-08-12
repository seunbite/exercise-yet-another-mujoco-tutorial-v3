""" 
    Assume that the main notebook called 'sys.path.append('../../package/helper/')'
"""
import numpy as np
from transformation import pr2t

def get_T_joi_from_g1(env):
    """
        Get joints of interest of G1 (IMPROVED MAPPING)
    """
    p_base,R_base = env.get_pR_body(body_name='pelvis')
    
    # Add missing shoulder pitch joints that were previously skipped
    p_rsh,R_rsh = env.get_pR_body(body_name='right_shoulder_pitch_link')
    p_lsh,R_lsh = env.get_pR_body(body_name='left_shoulder_pitch_link')
    
    p_rs,R_rs = env.get_pR_body(body_name='right_shoulder_roll_link')
    p_re,R_re = env.get_pR_body(body_name='right_elbow_pitch_link')
    # Better wrist mapping - use elbow_roll_link instead of hand joints
    p_rw,R_rw = env.get_pR_body(body_name='right_elbow_roll_link')
    
    p_ls,R_ls = env.get_pR_body(body_name='left_shoulder_roll_link')
    p_le,R_le = env.get_pR_body(body_name='left_elbow_pitch_link')
    # Better wrist mapping - use elbow_roll_link instead of hand joints
    p_lw,R_lw = env.get_pR_body(body_name='left_elbow_roll_link')
    
    # Better torso mapping
    p_torso,R_torso = env.get_pR_body(body_name='torso_link')
    
    p_rc,_ = env.get_pR_body(body_name='right_shoulder_pitch_link')
    p_lc,_ = env.get_pR_body(body_name='left_shoulder_pitch_link')

    # Better neck calculation using actual shoulder positions
    p_neck = 0.5 * (p_rsh+p_lsh)  # Use shoulder pitch links for more accurate positioning
    
    p_rp,R_rp = env.get_pR_body(body_name='right_hip_roll_link')
    p_rk,R_rk = env.get_pR_body(body_name='right_knee_link')
    p_ra,R_ra = env.get_pR_body(body_name='right_ankle_pitch_link')
    
    p_lp,R_lp = env.get_pR_body(body_name='left_hip_roll_link')
    p_lk,R_lk = env.get_pR_body(body_name='left_knee_link')
    p_la,R_la = env.get_pR_body(body_name='left_ankle_pitch_link')
    
    T_joi = {
        'base': pr2t(p_base,R_base),
        'hip': pr2t(0.5*(p_rp+p_lp),np.eye(3,3)),
        'spine': pr2t(p_base,R_base),  # Use pelvis as spine base
        'spine1': pr2t(p_torso,R_torso),  # Map to torso_link  
        'torso': pr2t(p_torso,R_torso),  # Better torso position
        'neck': pr2t(p_neck,np.eye(3,3)),
        'rsh': pr2t(p_rsh,R_rsh),      # Added missing right shoulder
        'lsh': pr2t(p_lsh,R_lsh),      # Added missing left shoulder
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),         # Improved wrist mapping
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),         # Improved wrist mapping
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
    }
    return T_joi

def get_joi_body_name_of_g1():
    """
        Get the body name of JOI (IMPROVED MAPPING)
    """
    joi_body_name = {
        'torso':'pelvis',
        'spine1':'torso_link',          # Map spine1 to torso_link
        'rsh':'right_shoulder_pitch_link',  # Added missing right shoulder
        'lsh':'left_shoulder_pitch_link',   # Added missing left shoulder
        'rs':'right_shoulder_roll_link',
        're':'right_elbow_pitch_link',
        'rw':'right_elbow_roll_link',   # Better wrist mapping
        'ls':'left_shoulder_roll_link',
        'le':'left_elbow_pitch_link',
        'lw':'left_elbow_roll_link',    # Better wrist mapping
        'rp':'right_hip_roll_link',
        'rk':'right_knee_link',
        'ra':'right_ankle_pitch_link',
        'lp':'left_hip_roll_link',
        'lk':'left_knee_link',
        'la':'left_ankle_pitch_link',
    }
    return joi_body_name