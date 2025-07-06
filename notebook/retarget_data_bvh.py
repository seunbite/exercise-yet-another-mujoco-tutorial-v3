import os
import pickle
import numpy as np

# --------------------------------------------------
# MediaPipe landmark indices for convenience
# --------------------------------------------------
MEDIAPIPE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}


# ----------------------------------------------------------------------------------
# Very small helper – identical to the one used in the notebook GIF creation section
# ----------------------------------------------------------------------------------

def upright_transform(mp_coords):
    """Transform MediaPipe coordinates (X right, Y down, Z into-screen)
    into a simple upright frame (X right, Y depth, Z up).

    mp_coords : ndarray, shape (33, 3)
    returns    : ndarray, shape (33, 3) – same points, upright.
    """
    out = np.zeros_like(mp_coords)
    out[:, 0] = mp_coords[:, 0]          # X stays the same
    out[:, 1] = mp_coords[:, 2]          # Depth  = original Z
    out[:, 2] = 1.0 - mp_coords[:, 1]    # Height = 1 – original Y (flip)
    return out


# -------------------------------------------------------------------------------
# BVH helper – copied in spirit from create_simple_bvh_content but **unaltered**
# scaling/angles so that the raw numbers appear in the BVH exactly as in the GIF.
# -------------------------------------------------------------------------------

def create_simple_bvh(frames_data, fps=20):
    """Generate a minimal BVH with the CMU-style 15-joint hierarchy.

    frames_data : list[list[float]] – each inner list is one frame's 63 channel
                  values (root X Y Z + 20×3 rotations).  We do **not** compute
                  fancy rotations – they remain 0.  What *does* vary is the root
                  translation which carries the raw hip-centre coordinates.
    """
    frame_time = 1.0 / fps

    header = f"""HIERARCHY
ROOT Hips
{{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {{
        OFFSET 0.0 5.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Spine1
        {{
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Neck
            {{
                OFFSET 0.0 10.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Head
                {{
                    OFFSET 0.0 5.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {{
                        OFFSET 0.0 5.0 0.0
                    }}
                }}
            }}
            JOINT LeftShoulder
            {{
                OFFSET -10.0 8.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftArm
                {{
                    OFFSET -15.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT LeftForeArm
                    {{
                        OFFSET -15.0 0.0 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {{
                            OFFSET -10.0 0.0 0.0
                        }}
                    }}
                }}
            }}
            JOINT RightShoulder
            {{
                OFFSET 10.0 8.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightArm
                {{
                    OFFSET 15.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT RightForeArm
                    {{
                        OFFSET 15.0 0.0 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {{
                            OFFSET 10.0 0.0 0.0
                        }}
                    }}
                }}
            }}
        }}
    }}
    JOINT LeftUpLeg
    {{
        OFFSET -5.0 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {{
            OFFSET 0.0 -20.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {{
                OFFSET 0.0 -20.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.0 -5.0 10.0
                }}
            }}
        }}
    }}
    JOINT RightUpLeg
    {{
        OFFSET 5.0 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightLeg
        {{
            OFFSET 0.0 -20.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightFoot
            {{
                OFFSET 0.0 -20.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.0 -5.0 10.0
                }}
            }}
        }}
    }}
}}
MOTION
Frames: {len(frames_data)}
Frame Time: {frame_time:.6f}
"""

    for frame in frames_data:
        header += ' '.join(f"{v:.6f}" for v in frame) + "\n"

    return header


# --------------------------------------------------------------------------
# Main conversion utility
# --------------------------------------------------------------------------

def convert_raw_keypoints_to_bvh(
    keypoints_pkl: str = "temp_mocap/keypoints_raw.pkl",
    output_bvh: str = "temp_mocap/bvh/raw_keypoints.bvh",
    fps: int = 20,
):
    """Load the pickled 3-D MediaPipe key-points and dump a BVH whose root
    translation reproduces the same motion seen in the GIF.  No extra scaling,
    no fitting to robot joints – pure data.
    """
    if not os.path.isfile(keypoints_pkl):
        raise FileNotFoundError(f"Key-point file not found: {keypoints_pkl}")

    with open(keypoints_pkl, "rb") as f:
        keypoints_3d = pickle.load(f)

    if not keypoints_3d:
        raise RuntimeError("Loaded key-point list is empty!")

    print(f"[convert] Loaded {len(keypoints_3d)} frames of key-points")

    # Build BVH channel frames (root translation + 60 dummy rotation values)
    frames_data = []

    for idx, pose in enumerate(keypoints_3d):
        if pose is None or not isinstance(pose, np.ndarray) or pose.shape[0] < 33:
            print(f"[convert] Frame {idx+1}: invalid pose – inserting zeros")
            frames_data.append([0.0] * 63)  # 63 channels as in the header
            continue

        # Apply the same upright transform used by the GIF visualiser
        pose_upright = upright_transform(pose)

        # Hip centre becomes the root translation
        left_hip  = pose_upright[MEDIAPIPE_LANDMARKS['left_hip']]
        right_hip = pose_upright[MEDIAPIPE_LANDMARKS['right_hip']]
        hip_center = (left_hip + right_hip) / 2.0

        # Root translation in *arbitrary units* (same as original 0-1 range)
        root_x, root_y, root_z = hip_center.tolist()

        # Assemble frame (root translation + 60 zeros for rotations)
        frame = [root_x, root_y, root_z] + [0.0] * 60
        frames_data.append(frame)

    # Write BVH
    os.makedirs(os.path.dirname(output_bvh), exist_ok=True)
    bvh_content = create_simple_bvh(frames_data, fps=fps)
    with open(output_bvh, "w") as f:
        f.write(bvh_content)

    print(f"[convert] BVH written to {output_bvh} (frames: {len(frames_data)})")


if __name__ == "__main__":
    import fire

    fire.Fire({
        "convert": convert_raw_keypoints_to_bvh,
    }) 