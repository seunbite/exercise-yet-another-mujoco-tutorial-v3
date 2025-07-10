import os
import time
import pickle
from copy import deepcopy

import numpy as np
from retarget_data import *
# Helper imports from existing packages
from mujoco_parser import MuJoCoParserClass, solve_ik
from transformation import rpy2r, pr2t, t2pr


# -----------------------------------------------------------------------------
# Mapping from MediaPipe landmark index → body name in the Unitree-G1 MuJoCo model
# -----------------------------------------------------------------------------
# We only include landmarks for which a clearly corresponding body exists in the
# robot description.  Trying to attach a landmark to a non-existent body raises
# an "Invalid name" error inside `solve_ik`, so the list below is intentionally
# conservative.

LMK2BODY_DEFAULT = {
    # Upper body --------------------------------------------------------------
    MEDIAPIPE_LANDMARKS['left_shoulder']:  'left_shoulder_pitch_link',
    MEDIAPIPE_LANDMARKS['right_shoulder']: 'right_shoulder_pitch_link',
    MEDIAPIPE_LANDMARKS['left_elbow']:     'left_elbow_pitch_link',
    MEDIAPIPE_LANDMARKS['right_elbow']:    'right_elbow_pitch_link',
    MEDIAPIPE_LANDMARKS['left_wrist']:     'left_elbow_roll_link',   # closest body
    MEDIAPIPE_LANDMARKS['right_wrist']:    'right_elbow_roll_link',  # closest body

    # Trunk ------------------------------------------------------------------
    MEDIAPIPE_LANDMARKS['left_hip']:       'left_hip_pitch_link',
    MEDIAPIPE_LANDMARKS['right_hip']:      'right_hip_pitch_link',

    # Lower body -------------------------------------------------------------
    MEDIAPIPE_LANDMARKS['left_knee']:      'left_knee_link',
    MEDIAPIPE_LANDMARKS['right_knee']:     'right_knee_link',
    MEDIAPIPE_LANDMARKS['left_ankle']:     'left_ankle_roll_link',
    MEDIAPIPE_LANDMARKS['right_ankle']:    'right_ankle_roll_link',

    # Feet (use ankle roll link as closest rigid body) ------------------------
    MEDIAPIPE_LANDMARKS['left_heel']:          'left_ankle_roll_link',
    MEDIAPIPE_LANDMARKS['right_heel']:         'right_ankle_roll_link',
    MEDIAPIPE_LANDMARKS['left_foot_index']:    'left_ankle_roll_link',
    MEDIAPIPE_LANDMARKS['right_foot_index']:   'right_ankle_roll_link',
}

print(LMK2BODY_DEFAULT)
JOINT_NAMES_FOR_IK = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'left_elbow_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_elbow_roll_joint',
]


def convert_landmarks_to_world(landmarks, p_rate=0.056444 * 15, z_offset=0.7, radians=(90, 180, -90), use_ankle_root=True):
    """Transform MediaPipe landmarks to world-space coordinates.
    
    Args:
        landmarks: MediaPipe landmarks array
        p_rate: Scale factor
        z_offset: Vertical offset
        radians: Rotation angles
        use_ankle_root: If True, uses ankle center as root (🦶). If False, uses hip center.
    
    Returns an (N,3) np.array of world-space points.
    """
    if use_ankle_root:
        # 🦶 NEW: Use ankle center as root (better for walking/dancing)
        left_anchor, right_anchor = landmarks[27], landmarks[28]  # Ankle indices
        root_center = (left_anchor + right_anchor) / 2.0
        print(f"🦶 Ankle root center: {root_center}")
    else:
        # 🕺 ORIGINAL: Use hip center as root (better for upper body movement)
        left_anchor, right_anchor = landmarks[23], landmarks[24]  # Hip indices
        root_center = (left_anchor + right_anchor) / 2.0
        print(f"🕺 Hip root center: {root_center}")

    # Root transform
    p_root = root_center * p_rate
    R_root = np.eye(3)
    T_root = pr2t(p_root, R_root)
    T_root_rot = pr2t(np.zeros(3), rpy2r(np.radians(radians))) @ T_root
    p_root_rot, R_root_rot = t2pr(T_root_rot)

    pts_global = []
    for lm in landmarks:
        # Transform all landmarks relative to root center
        p_local = (lm - root_center) * p_rate
        p_glob = R_root_rot @ p_local + p_root_rot + np.array([0.0, 0.0, z_offset])
        pts_global.append(p_glob)
    return np.asarray(pts_global)


# ----------------------------------------------------------------------
# Main retargeting routine
# ----------------------------------------------------------------------

def ik_retargeting(
    keypoints_pkl: str = 'temp_mocap/88848d2067d622de8e4f314e28dc431a.pkl',
    do_render: bool = True,
    xml_path: str = '../asset/unitree_g1/scene_g1.xml',
    landmark_to_body: dict = None,
    joint_names_for_ik: list = None,
    p_rate: float = 0.056444 * 15,
    z_offset: float = 0.7,
    radians: tuple = (90, 180, -90),
    ik_steps: int = 50,
    ik_err_th: float = 1e-3,
    playback_speed: float = 0.005,
    repetition: int = 5,
    use_ankle_root: bool = True,  # 🆕 NEW: Choose root reference point
):
    """Perform IK so the robot mimics the motion captured in `keypoints_pkl`.

    1. Loads 3-D MediaPipe landmarks saved by retarget_data.py.
    2. For each frame, picks a subset of those landmarks (mapping dict) and
       solves IK so the corresponding robot body attaches to that point.
    3. Animates the result in MuJoCo viewer.
    """
    if landmark_to_body is None:
        landmark_to_body = LMK2BODY_DEFAULT
    if joint_names_for_ik is None:
        joint_names_for_ik = JOINT_NAMES_FOR_IK

    if not os.path.isfile(keypoints_pkl):
        raise FileNotFoundError(f'Landmark pickle not found: {keypoints_pkl}')

    with open(keypoints_pkl, 'rb') as f:
        keypoints_3d = pickle.load(f)
    if not keypoints_3d:
        raise RuntimeError('Loaded key-point list is empty!')

    # ---------------- MuJoCo environment ----------------
    env = MuJoCoParserClass(name='IKRetarget', rel_xml_path=xml_path, verbose=False)
    env.reset(step=True)
    
    # Dynamic title based on root choice
    title = f"IK Retarget - {'Ankle Root 🦶' if use_ankle_root else 'Hip Root 🕺'}"
    env.init_viewer(title=title, transparent=False, backend='native')
    if do_render:
        env.init_viewer(title='IK Retarget', transparent=False, backend='native')

    frame_count = len(keypoints_3d)
    rep = tick = 0
    while (not do_render or env.is_viewer_alive()) and rep < repetition:
        landmarks = keypoints_3d[tick]
        tick += 1
        if tick >= frame_count:
            tick = 0
            rep += 1

        if landmarks is None:
            if do_render:
                env.render()
                time.sleep(playback_speed)
            continue

        pts_world = convert_landmarks_to_world(landmarks, p_rate=p_rate, z_offset=z_offset, radians=radians, use_ankle_root=use_ankle_root)


        # 🎯 Visualize root reference point
        if use_ankle_root:
            # 🦶 Ankle root: Show ankle center and individual ankles
            left_ankle, right_ankle = landmarks[27], landmarks[28]
            ankle_center_world = (pts_world[27] + pts_world[28]) / 2.0
            env.plot_sphere(p=ankle_center_world, r=0.08, rgba=[1, 0, 0, 0.8])  # Large red sphere for ankle center
            
            # Show individual ankles
            env.plot_sphere(p=pts_world[27], r=0.04, rgba=[0, 0, 1, 0.8])  # Blue for left ankle
            env.plot_sphere(p=pts_world[28], r=0.04, rgba=[0, 1, 1, 0.8])  # Cyan for right ankle
        else:
            # 🕺 Hip root: Show hip center and individual hips
            left_hip, right_hip = landmarks[23], landmarks[24]
            hip_center_world = (pts_world[23] + pts_world[24]) / 2.0
            env.plot_sphere(p=hip_center_world, r=0.08, rgba=[1, 0, 0, 0.8])  # Large red sphere for hip center
            
            # Show individual hips
            env.plot_sphere(p=pts_world[23], r=0.04, rgba=[0, 0, 1, 0.8])  # Blue for left hip
            env.plot_sphere(p=pts_world[24], r=0.04, rgba=[0, 1, 1, 0.8])  # Cyan for right hip

        # --------------- Solve IK for each mapping ----------------
        for lm_idx, body_name in landmark_to_body.items():
            p_trgt = pts_world[lm_idx]

            # IK towards this single body
            try:
                q_init, ik_err_stack, ik_info = solve_ik(
                    env=env,
                    do='forward',
                    joint_names_for_ik=joint_names_for_ik,
                    body_name_trgt=body_name,
                    p_trgt=p_trgt,
                    R_trgt=None,
                    max_ik_tick=ik_steps,
                    ik_err_th=ik_err_th,
                    restore_state=False,
                    verbose=False,
                )
                # IK solver already moved env's qpos via forward(); nothing else is needed.
                pass  # keep for clarity – no torque stepping here
            except Exception as e:
                print(f'[IK] Warning: IK failed for {body_name}: {e}')

        if do_render:
            env.render()
            env.plot_sphere(p=p_trgt, r=0.02, rgba=[0, 1, 0, 0.6])  # visualise target
            time.sleep(playback_speed)
            env.remove_all_geoms()
            env.plot_T()


    if do_render:
        env.close_viewer()
        print('Viewer closed.')


if __name__ == '__main__':
    import fire
    fire.Fire(ik_retargeting)