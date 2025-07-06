import os
import cv2
import numpy as np
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from PIL import Image
from utils import download_video, extract_frames
import mediapipe as mp
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


def load_video_to_frames(input_mp4: str, fps: int = 20):
    input_basename = os.path.basename(input_mp4)

    for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm']:
        if input_basename.lower().endswith(ext):
            input_basename = input_basename[:-len(ext)]
            break

    if input_mp4.startswith(('http://', 'https://')):
        video_filename = os.path.basename(urlparse(input_mp4).path) or "video.mp4"
            
        if not video_filename.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
            if '.gif' in input_mp4.lower():
                video_filename += '.gif'
            else:
                video_filename += '.mp4'
            
        video_path = download_video(input_mp4, os.path.join('temp_mocap', video_filename))
        if not video_path:
            print("Video/GIF download failed. Skipping processing.")
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[extract_frames_to_memory] Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"[extract_frames_to_memory] Video info:")
    print(f"  Original FPS: {video_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Target FPS: {fps}")
    
    # Calculate frame sampling interval
    frame_interval = max(1, int(video_fps / fps)) if video_fps > 0 else 1
    print(f"  Frame interval: {frame_interval} (every {frame_interval} frames)")
    
    frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames at target FPS
        if frame_count % frame_interval == 0:
            frames.append(frame.copy())
            extracted_count += 1
            
            if extracted_count <= 5:  # Debug first few frames
                print(f"[extract_frames_to_memory] Extracted frame {extracted_count}: {frame.shape}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"[extract_frames_to_memory] Extracted {len(frames)} frames from {frame_count} total frames")
    return frames


def estimate_2d_pose(frames: list):
    """
    MediaPipe Holistic으로 메모리의 프레임들에서 2D 관절(x,y,z)을 뽑아서 리스트로 반환.
    frames: List of numpy arrays (frames in memory)
    returns: List of pose data dictionaries with landmark coordinates
    """
    # Use MediaPipe Holistic instead of just Pose for better results
    mp_holistic = mp.solutions.holistic
    
    # MediaPipe pose landmarks (33 landmarks)
    pose_landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
        'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
        'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]
    
    poses_data = []
    
    if not frames:
        print(f"[estimate_2d_pose_from_frames] No frames provided")
        return poses_data
    
    print(f"[estimate_2d_pose_from_frames] Processing {len(frames)} frames")

    detection_stats = {'detected': 0, 'failed': 0, 'errors': 0}
    pose_variations = []  # Track pose variations to detect if poses are changing
    
    # Use streaming/continuous mode for improved temporal tracking across frames
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
        refine_face_landmarks=False) as holistic:
        
        for i, img in enumerate(frames):
            try:
                # Get image dimensions
                height, width = img.shape[:2]
                if i < 3:  # Debug first few frames
                    print(f"[estimate_2d_pose_from_frames] Processing frame {i+1}/{len(frames)}: ({width}x{height})")
                
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process with MediaPipe Holistic
                results = holistic.process(image_rgb)
                
                # Extract pose landmarks
                frame_data = {'frame': i}
                
                # --------------------------------------------------
                # Pose landmarks (normalized) and world landmarks (m)
                # --------------------------------------------------
                if results.pose_landmarks:
                    pose_landmarks_data = results.pose_landmarks.landmark
                    detection_stats['detected'] += 1
                    
                    # Extract all pose coordinates with proper naming
                    pose_coords = []
                    for j, landmark_name in enumerate(pose_landmark_names):
                        if j < len(pose_landmarks_data):
                            landmark = pose_landmarks_data[j]
                            # Store normalized coordinates directly (as in reference)
                            pose_coords.append([landmark.x, landmark.y, landmark.z])
                            
                            # Also store in frame_data for compatibility
                            frame_data[f'{landmark_name}_x'] = landmark.x
                            frame_data[f'{landmark_name}_y'] = landmark.y
                            frame_data[f'{landmark_name}_z'] = landmark.z
                            frame_data[f'{landmark_name}_visibility'] = landmark.visibility
                        else:
                            pose_coords.append([0.0, 0.0, 0.0])
                            frame_data[f'{landmark_name}_x'] = 0.0
                            frame_data[f'{landmark_name}_y'] = 0.0
                            frame_data[f'{landmark_name}_z'] = 0.0
                            frame_data[f'{landmark_name}_visibility'] = 0.0
                    
                    frame_data['pose_landmarks'] = np.array(pose_coords)
                    poses_data.append(frame_data)
                    
                    # Save *world* landmark coordinates when available.
                    # These are in meter-scale camera space and include
                    # the subject's global translation – perfect for BVH.
                    if results.pose_world_landmarks:
                        world_coords = []
                        for j, lm in enumerate(results.pose_world_landmarks.landmark):
                            world_coords.append([lm.x, lm.y, lm.z])
                        frame_data['pose_world_landmarks'] = np.array(world_coords)
                    
                    # Track pose variation to detect if poses are actually changing
                    if len(poses_data) > 1 and poses_data[-2] is not None and 'pose_landmarks' in poses_data[-2]:
                        # Calculate difference between consecutive poses
                        prev_coords = poses_data[-2]['pose_landmarks'][:, :2]  # Use only x,y for variation
                        curr_coords = np.array(pose_coords)[:, :2]
                        diff = np.mean(np.abs(curr_coords - prev_coords))
                        pose_variations.append(diff)
                        if i < 5:  # Print first few for debugging
                            print(f"[estimate_2d_pose_from_frames] Frame {i+1} pose variation from previous: {diff:.6f}")
                    
                    # Print some landmark coordinates for debugging (first few frames)
                    if i < 3:
                        nose = pose_coords[0]  # First landmark is nose
                        left_shoulder = pose_coords[11]  # Left shoulder
                        right_shoulder = pose_coords[12]  # Right shoulder
                        print(f"[estimate_2d_pose_from_frames] Frame {i+1} sample landmarks:")
                        print(f"  Nose: ({nose[0]:.4f}, {nose[1]:.4f}, {nose[2]:.4f})")
                        print(f"  Left shoulder: ({left_shoulder[0]:.4f}, {left_shoulder[1]:.4f}, {left_shoulder[2]:.4f})")
                        print(f"  Right shoulder: ({right_shoulder[0]:.4f}, {right_shoulder[1]:.4f}, {right_shoulder[2]:.4f})")
                else:
                    poses_data.append(None)
                    detection_stats['failed'] += 1
                    if i < 5:  # Only print for first few frames to avoid spam
                        print(f"[estimate_2d_pose_from_frames] No pose detected in frame {i+1}")
                        
            except Exception as e:
                print(f"[estimate_2d_pose_from_frames] Error processing frame {i+1}: {e}")
                poses_data.append(None)
                detection_stats['errors'] += 1
    
    # Print detailed statistics
    print(f"\n[estimate_2d_pose_from_frames] Processing completed:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Poses detected: {detection_stats['detected']}")
    print(f"  Detection failed: {detection_stats['failed']}")
    print(f"  Processing errors: {detection_stats['errors']}")
    print(f"  Detection rate: {detection_stats['detected']/len(frames)*100:.1f}%")
    
    # Analyze pose variations
    if pose_variations:
        avg_variation = np.mean(pose_variations)
        max_variation = np.max(pose_variations)
        min_variation = np.min(pose_variations)
        print(f"  Pose variation stats:")
        print(f"    Average: {avg_variation:.6f}")
        print(f"    Max: {max_variation:.6f}")
        print(f"    Min: {min_variation:.6f}")
        
        if avg_variation < 0.001:
            print(f"  ⚠️  WARNING: Very low pose variation detected! Poses might be too similar or static.")
        elif avg_variation > 0.1:
            print(f"  ✅ Good pose variation detected - motion should be captured correctly.")
        else:
            print(f"  ✅ Moderate pose variation detected.")
    
    return poses_data


def estimate_3d_pose(poses_data: list):
    """
    VideoPose3D 같은 모델로 2D → 3D 복원.
    poses_data: List of pose data dictionaries from estimate_2d_pose
    returns: List of 3D pose data
    """
    # >>> 여기에 VideoPose3D 모델 불러오기 및 추론 코드 삽입 <<<
    # 현재는 MediaPipe에서 이미 z 좌표를 제공하므로 그것을 사용
    poses3d = []
    
    print(f"[lift_to_3d] Processing {len(poses_data)} pose data entries")
    
    for i, pose_data in enumerate(poses_data):
        if pose_data is None:
            poses3d.append(None)
            continue
        
        # Prefer world-landmark coordinates (contain subject translation).
        if 'pose_world_landmarks' in pose_data:
            landmarks_3d = pose_data['pose_world_landmarks']  # meters
        elif 'pose_landmarks' in pose_data:
            landmarks_3d = pose_data['pose_landmarks']  # normalized
        else:
            # ------------------------------------------------------
            # Use **absolute** coordinates if we used world landmarks
            # (they are already metric). Otherwise scale normalized.
            # ------------------------------------------------------
            if np.max(np.abs(landmarks_3d)) < 3.0:  # Heuristic: world coords in metres ~ <3
                scale_factor = 100.0  # m→cm
                root_x = landmarks_3d[0][0] * scale_factor
                root_y = landmarks_3d[0][2] * scale_factor  # depth → Y
                root_z = landmarks_3d[0][1] * scale_factor  # vertical → Z
            else:
                scale_factor = 170.0
                root_x = (landmarks_3d[0][0] - 0.5) * scale_factor * 0.5
                root_y = landmarks_3d[0][1] * scale_factor * 0.3
                root_z = landmarks_3d[0][2] * scale_factor * 0.3
        
        poses3d.append(landmarks_3d)
        
        # Debug: Print sample 3D coordinates for first few frames
        if i < 3:
            nose = landmarks_3d[0]  # Nose landmark
            left_shoulder = landmarks_3d[11]  # Left shoulder
            right_shoulder = landmarks_3d[12]  # Right shoulder
            print(f"[lift_to_3d] Frame {i+1} 3D landmarks:")
            print(f"  Nose: ({nose[0]:.4f}, {nose[1]:.4f}, {nose[2]:.4f})")
            print(f"  Left shoulder: ({left_shoulder[0]:.4f}, {left_shoulder[1]:.4f}, {left_shoulder[2]:.4f})")
            print(f"  Right shoulder: ({right_shoulder[0]:.4f}, {right_shoulder[1]:.4f}, {right_shoulder[2]:.4f})")
    
    detected_3d = sum(1 for p in poses3d if p is not None)
    print(f"[lift_to_3d] Processed {len(poses3d)} frames, {detected_3d} with 3D data")
    
    return poses3d


def save_bvh(poses3d: list, output_bvh: str, fps: int = 20):
    os.makedirs(os.path.dirname(output_bvh), exist_ok=True)
    
    MEDIAPIPE_LANDMARKS = {
        'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3, 'right_eye_inner': 4,
        'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7, 'right_ear': 8, 'mouth_left': 9,
        'mouth_right': 10, 'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
        'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29,
        'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32
    }
    
    
    bvh_frames = []
    for frame_idx, landmarks in enumerate(poses3d):
        try:
            nose = landmarks[MEDIAPIPE_LANDMARKS['nose']]
            left_shoulder = landmarks[MEDIAPIPE_LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[MEDIAPIPE_LANDMARKS['right_shoulder']]
            left_hip = landmarks[MEDIAPIPE_LANDMARKS['left_hip']]
            right_hip = landmarks[MEDIAPIPE_LANDMARKS['right_hip']]
            left_elbow = landmarks[MEDIAPIPE_LANDMARKS['left_elbow']]
            right_elbow = landmarks[MEDIAPIPE_LANDMARKS['right_elbow']]
            left_wrist = landmarks[MEDIAPIPE_LANDMARKS['left_wrist']]
            right_wrist = landmarks[MEDIAPIPE_LANDMARKS['right_wrist']]
            left_knee = landmarks[MEDIAPIPE_LANDMARKS['left_knee']]
            right_knee = landmarks[MEDIAPIPE_LANDMARKS['right_knee']]
            left_ankle = landmarks[MEDIAPIPE_LANDMARKS['left_ankle']]
            right_ankle = landmarks[MEDIAPIPE_LANDMARKS['right_ankle']]
            
            hip_center = (left_hip + right_hip) / 2.0
            
            scale_factor = 170.0
            
            root_x = (hip_center[0] - 0.5) * scale_factor * 0.5
            root_y = hip_center[1] * scale_factor * 0.3
            root_z = hip_center[2] * scale_factor * 0.3
            
            spine_direction = nose - hip_center
            hip_orientation = np.arctan2(spine_direction[0], spine_direction[2]) * 180.0 / np.pi
            
            # Arm rotations (simplified)
            # Left shoulder
            left_arm_vec = left_elbow - left_shoulder
            left_shoulder_x = np.arctan2(left_arm_vec[1], left_arm_vec[2]) * 180.0 / np.pi
            left_shoulder_y = np.arctan2(left_arm_vec[0], left_arm_vec[2]) * 180.0 / np.pi
            left_shoulder_z = 0.0
            
            # Right shoulder
            right_arm_vec = right_elbow - right_shoulder
            right_shoulder_x = np.arctan2(right_arm_vec[1], right_arm_vec[2]) * 180.0 / np.pi
            right_shoulder_y = np.arctan2(right_arm_vec[0], right_arm_vec[2]) * 180.0 / np.pi
            right_shoulder_z = 0.0
            
            # Left elbow
            left_forearm_vec = left_wrist - left_elbow
            left_elbow_angle = calculate_joint_angle(left_shoulder, left_elbow, left_wrist)
            
            # Right elbow  
            right_forearm_vec = right_wrist - right_elbow
            right_elbow_angle = calculate_joint_angle(right_shoulder, right_elbow, right_wrist)
            
            # Leg rotations (simplified)
            # Left hip
            left_thigh_vec = left_knee - left_hip
            left_hip_x = np.arctan2(left_thigh_vec[1], left_thigh_vec[2]) * 180.0 / np.pi
            left_hip_y = np.arctan2(left_thigh_vec[0], left_thigh_vec[2]) * 180.0 / np.pi
            left_hip_z = 0.0
            
            # Right hip
            right_thigh_vec = right_knee - right_hip
            right_hip_x = np.arctan2(right_thigh_vec[1], right_thigh_vec[2]) * 180.0 / np.pi
            right_hip_y = np.arctan2(right_thigh_vec[0], right_thigh_vec[2]) * 180.0 / np.pi
            right_hip_z = 0.0
            
            # Knee angles
            left_knee_angle = calculate_joint_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_joint_angle(right_hip, right_knee, right_ankle)
            
            frame_data = [
                root_x, root_y, root_z,  # Root translation
                0.0, hip_orientation, 0.0,  # Root rotation
                0.0, 0.0, 0.0,  # Spine rotations
                0.0, 0.0, 0.0,  # Neck rotations  
                0.0, 0.0, 0.0,  # Head rotations
                left_shoulder_x, left_shoulder_y, left_shoulder_z,  # Left shoulder
                left_elbow_angle, 0.0, 0.0,  # Left elbow
                0.0, 0.0, 0.0,  # Left wrist
                right_shoulder_x, right_shoulder_y, right_shoulder_z,  # Right shoulder
                right_elbow_angle, 0.0, 0.0,  # Right elbow
                0.0, 0.0, 0.0,  # Right wrist
                left_hip_x, left_hip_y, left_hip_z,  # Left hip
                left_knee_angle, 0.0, 0.0,  # Left knee
                0.0, 0.0, 0.0,  # Left ankle
                right_hip_x, right_hip_y, right_hip_z,  # Right hip
                right_knee_angle, 0.0, 0.0,  # Right knee
                0.0, 0.0, 0.0,  # Right ankle
            ]
            
            target_channels = 54
            if len(frame_data) < target_channels:
                frame_data.extend([0.0] * (target_channels - len(frame_data)))
            elif len(frame_data) > target_channels:
                frame_data = frame_data[:target_channels]
            
            bvh_frames.append(frame_data)
                
        except Exception as e:
            print(f"[retarget_and_save_bvh] Error processing frame {frame_idx+1}: {e}")
            # Use previous frame or zeros as fallback
            if bvh_frames:
                bvh_frames.append(bvh_frames[-1].copy())
            else:
                bvh_frames.append([0.0] * 63)
    
    frame_time = 1.0 / fps

    bvh_header = f"""HIERARCHY
ROOT Hips
{{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Spine
    {{
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT Neck
        {{
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT Head
            {{
                OFFSET 0.0 5.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
                End Site
                {{
                    OFFSET 0.0 5.0 0.0
                }}
            }}
        }}
        JOINT LeftShoulder
        {{
            OFFSET -5.0 8.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftElbow
            {{
                OFFSET -10.0 0.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT LeftWrist
                {{
                    OFFSET -10.0 0.0 0.0
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site
                    {{
                        OFFSET -5.0 0.0 0.0
                    }}
                }}
            }}
        }}
        JOINT RightShoulder
        {{
            OFFSET 5.0 8.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightElbow
            {{
                OFFSET 10.0 0.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT RightWrist
                {{
                    OFFSET 10.0 0.0 0.0
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site
                    {{
                        OFFSET 5.0 0.0 0.0
                    }}
                }}
            }}
        }}
    }}
    JOINT LeftHip
    {{
        OFFSET -5.0 0.0 0.0
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT LeftKnee
        {{
            OFFSET 0.0 -15.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftAnkle
            {{
                OFFSET 0.0 -15.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
                End Site
                {{
                    OFFSET 0.0 -5.0 10.0
                }}
            }}
        }}
    }}
    JOINT RightHip
    {{
        OFFSET 5.0 0.0 0.0
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT RightKnee
        {{
            OFFSET 0.0 -15.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightAnkle
            {{
                OFFSET 0.0 -15.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
                End Site
                {{
                    OFFSET 0.0 -5.0 10.0
                }}
            }}
        }}
    }}
    JOINT Spare
    {{
        OFFSET 0.0 0.0 0.0
        CHANNELS 3 Zrotation Yrotation Xrotation
        End Site
        {{
            OFFSET 0.0 0.0 0.0
        }}
    }}
}}
MOTION
Frames: {len(bvh_frames)}
Frame Time: {frame_time:.6f}
"""

    # Assemble complete BVH content (header + motion frames)
    motion_str = "\n".join(" ".join(f"{v:.6f}" for v in frame) for frame in bvh_frames)
    bvh_content = bvh_header + "\n" + motion_str + "\n"

    with open(output_bvh, 'w') as f:
        f.write(bvh_content)
    

def calculate_joint_angle(joint1, joint2, joint3):
    """Calculate angle at joint2 between joint1-joint2 and joint2-joint3 vectors"""
    try:
        vec1 = joint1 - joint2
        vec2 = joint3 - joint2
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        
        # Convert to joint rotation (bend angle)
        bend_angle = 180.0 - angle
        
        return bend_angle
    except:
        return 0.0


def create_default_t_pose():
    """Create a default T-pose with 33 MediaPipe landmarks"""
    t_pose = np.zeros((33, 3))
    
    # Head/Face landmarks (0-10)
    t_pose[0] = [0.5, 0.2, 0.0]  # nose
    t_pose[1] = [0.48, 0.18, 0.0]  # left_eye_inner
    t_pose[2] = [0.46, 0.18, 0.0]  # left_eye
    t_pose[3] = [0.44, 0.18, 0.0]  # left_eye_outer
    t_pose[4] = [0.52, 0.18, 0.0]  # right_eye_inner
    t_pose[5] = [0.54, 0.18, 0.0]  # right_eye
    t_pose[6] = [0.56, 0.18, 0.0]  # right_eye_outer
    t_pose[7] = [0.42, 0.2, 0.0]   # left_ear
    t_pose[8] = [0.58, 0.2, 0.0]   # right_ear
    t_pose[9] = [0.48, 0.25, 0.0]  # mouth_left
    t_pose[10] = [0.52, 0.25, 0.0] # mouth_right
    
    # Upper body landmarks (11-22)
    t_pose[11] = [0.4, 0.4, 0.0]   # left_shoulder
    t_pose[12] = [0.6, 0.4, 0.0]   # right_shoulder
    t_pose[13] = [0.3, 0.5, 0.0]   # left_elbow
    t_pose[14] = [0.7, 0.5, 0.0]   # right_elbow
    t_pose[15] = [0.2, 0.6, 0.0]   # left_wrist
    t_pose[16] = [0.8, 0.6, 0.0]   # right_wrist
    t_pose[17] = [0.18, 0.62, 0.0] # left_pinky
    t_pose[18] = [0.82, 0.62, 0.0] # right_pinky
    t_pose[19] = [0.15, 0.6, 0.0]  # left_index
    t_pose[20] = [0.85, 0.6, 0.0]  # right_index
    t_pose[21] = [0.22, 0.58, 0.0] # left_thumb
    t_pose[22] = [0.78, 0.58, 0.0] # right_thumb
    
    # Lower body landmarks (23-32)
    t_pose[23] = [0.45, 0.7, 0.0]  # left_hip
    t_pose[24] = [0.55, 0.7, 0.0]  # right_hip
    t_pose[25] = [0.45, 0.85, 0.0] # left_knee
    t_pose[26] = [0.55, 0.85, 0.0] # right_knee
    t_pose[27] = [0.45, 1.0, 0.0]  # left_ankle
    t_pose[28] = [0.55, 1.0, 0.0]  # right_ankle
    t_pose[29] = [0.44, 1.02, 0.0] # left_heel
    t_pose[30] = [0.56, 1.02, 0.0] # right_heel
    t_pose[31] = [0.46, 0.98, 0.0] # left_foot_index
    t_pose[32] = [0.54, 0.98, 0.0] # right_foot_index
    
    return t_pose


def save_gif(poses3d: list, original_frames: list, output_gif: str, fps: int = 10):
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # Face outline
        (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Shoulder to hip
        (23, 24),  # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (27, 29), (27, 31),  # Left foot
        (28, 30), (28, 32),  # Right foot
    ]
    
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
        'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
        'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]
    
    pil_images = []
    created_frames = 0
    
    for i, (pose3d, original_frame) in enumerate(zip(poses3d, original_frames)):
        try:
            fig = plt.figure(figsize=(15, 6))
            ax1 = fig.add_subplot(121)
            original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            ax1.imshow(original_rgb)
            ax1.set_title(f'Original Frame {i+1}')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(122, projection='3d')
            
            if pose3d is not None and len(pose3d) > 0:
                if isinstance(pose3d, np.ndarray) and pose3d.shape[1] == 3:
                    landmarks_3d = pose3d.copy()
                    landmarks_3d_transformed = np.zeros_like(landmarks_3d)
                    landmarks_3d_transformed[:, 0] = landmarks_3d[:, 0]
                    landmarks_3d_transformed[:, 1] = landmarks_3d[:, 2]
                    landmarks_3d_transformed[:, 2] = 1 - landmarks_3d[:, 1]
                    
                    for connection in pose_connections:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks_3d_transformed) and end_idx < len(landmarks_3d_transformed):
                            start_point = landmarks_3d_transformed[start_idx]
                            end_point = landmarks_3d_transformed[end_idx]
                            
                            if not (np.allclose(landmarks_3d[start_idx], 0) or np.allclose(landmarks_3d[end_idx], 0)):
                                ax2.plot3D([start_point[0], end_point[0]], 
                                          [start_point[1], end_point[1]], 
                                          [start_point[2], end_point[2]], 'b-', linewidth=2)
                    
                    valid_landmarks = 0
                    for j, (landmark, original_landmark) in enumerate(zip(landmarks_3d_transformed, landmarks_3d)):
                        if not np.allclose(original_landmark, 0):
                            color = 'r' if j == 0 else 'g'
                            markersize = 50 if j == 0 else 30
                            ax2.scatter(landmark[0], landmark[1], landmark[2], 
                                      c=color, s=markersize, alpha=0.8)
                            valid_landmarks += 1
                            
                            if j in [0, 11, 12, 15, 16, 23, 24]:
                                ax2.text(landmark[0], landmark[1], landmark[2], 
                                        landmark_names[j].replace('_', '\n'), 
                                        fontsize=8, alpha=0.7)
                    
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([-0.5, 0.5])
                    ax2.set_zlim([0, 1])
                    ax2.set_xlabel('X (Left-Right)')
                    ax2.set_ylabel('Y (Depth)')
                    ax2.set_zlabel('Z (Height)')
                    ax2.set_title(f'3D Pose Frame {i+1}\n({valid_landmarks} landmarks)')
                    
                    ax2.view_init(elev=10, azim=-90 + i * 1)
                else:
                    ax2.text(0.5, 0.5, 0, 'Invalid 3D Data', fontsize=12, ha='center')
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([0, 1])
                    ax2.set_zlim([0, 1])
                    ax2.set_title(f'3D Pose Frame {i+1}\n(Invalid Data)')
            else:
                ax2.text(0.5, 0.5, 0, 'No Pose Detected', fontsize=12, ha='center')
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                ax2.set_zlim([0, 1])
                ax2.set_title(f'3D Pose Frame {i+1}\n(No Pose)')
            
            plt.tight_layout()
            fig.canvas.draw()
            
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]
            
            pil_img = Image.fromarray(buf)
            pil_images.append(pil_img)
            
            plt.close(fig)
            created_frames += 1
            
        except Exception as e:
            print(f"[create_3d_pose_gif_from_memory] Error processing frame {i+1}: {e}")
            plt.close('all')
    
    if not pil_images:
        print(f"[create_3d_pose_gif_from_memory] No images created for GIF")
        return False
    
    duration = int(1000 / fps)
    pil_images[0].save(
        output_gif,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    return True


def run(
    input_mp4: str = "https://i.pinimg.com/originals/58/f2/3c/58f23c62733b75711785a822cdb052c6.gif",
    output_dir: str = "temp_mocap",
    fps: int = 20,
    xml_path: str = "../asset/unitree_g1/scene_g1.xml",
    p_rate: float = 0.056444*15,
    do_gif: bool = True,
    do_bvh: bool = True,
    do_simulate: str = "bvh", # none, mocap, bvh
    z_offset: float = 0.7,
    repetition: int = 15,
    playback_speed: float = 0.5,
    radians: tuple = (90, 180, -90)
):
    bvh_dir = os.path.join(output_dir, "bvh")
    viz_dir = os.path.join(output_dir, "visualization")
    os.makedirs(bvh_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    frames = load_video_to_frames(input_mp4, fps=fps)
    if not frames:
        print("Frame extraction failed. Skipping processing.")
        return
    
    poses_data = estimate_2d_pose(frames)
    poses3d = estimate_3d_pose(poses_data)
    
    input_basename = os.path.basename(input_mp4).split('.')[0]
    if do_gif:
        pose_3d_gif = os.path.join(viz_dir, f"{input_basename}_3d_pose.gif")
        if save_gif(poses3d, frames, pose_3d_gif, fps=min(fps, 1)):
            print(f"✅ 3D pose GIF saved: {pose_3d_gif}")

    simulate_mode = (do_simulate or '').lower()

    # ---------------- BVH generation -----------------
    if do_bvh:
        bvh_out_path = os.path.join(bvh_dir, f"{input_basename}.bvh")
        save_bvh(poses3d, bvh_out_path, fps=fps)

    # ---------------- Simulation --------------------
    if simulate_mode == 'bvh':
        # Visualise the generated BVH file
        bvh_out_path = os.path.join(bvh_dir, f"{input_basename}.bvh")
        env = MuJoCoParserClass(name='BVHViewer', rel_xml_path=xml_path, verbose=False)
        secs, chains = get_chains_from_bvh_cmu_mocap(
            bvh_path=bvh_out_path,
            p_rate=p_rate,
            rpy_order=[0, 1, 2],
            zup=True,
            plot_chain_graph=False,
            verbose=False,
        )
        secs, chains = secs[1:], chains[1:]  # skip first T-pose frame

        for tick, chain in enumerate(chains):
            p_root, R_root = chain.get_root_joint_pR()
            T_root = pr2t(p_root, R_root)
            T_root_rot = pr2t(np.zeros(3), rpy2r(np.radians(radians))) @ T_root
            p_root_rot, R_root_rot = t2pr(T_root_rot)

            chain_rot = deepcopy(chain)
            chain_rot.set_root_joint_pR(p=p_root_rot + np.array([0.0, 0.0, z_offset]), R=R_root_rot)
            chain_rot.forward_kinematics()
            chains[tick] = chain_rot

        env.reset(step=True)
        env.init_viewer(title='BVH playback', transparent=False, backend='native')

        rep = tick = 0
        while env.is_viewer_alive() and rep < repetition:
            chain = chains[tick]
            env.remove_all_geoms()  # clear markers previously added
            env.plot_T()
            chain.plot_chain_mujoco(
                env=env,
                plot_joint=True,
                plot_joint_name=True,
                r_link=0.0025,
                rgba_link=(1, 0, 0, 0.6),
            )
            env.render()
            time.sleep(playback_speed)
            tick += 1
            if tick >= len(chains):
                tick = 0
                rep += 1
        env.close_viewer()
        print("Viewer closed.")

    elif simulate_mode == 'mocap':
        print("Simulating Mocap data...")
        pose_connections = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # face
            (9, 10),
            (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 23), (12, 24),
            (23, 24),
            (23, 25), (25, 27),
            (24, 26), (26, 28),
            (27, 29), (27, 31),
            (28, 30), (28, 32),
        ]

        env = MuJoCoParserClass(name='MocapViewer', rel_xml_path=xml_path, verbose=False)
        env.reset(step=True)
        env.init_viewer(title='Mocap playback', transparent=False, backend='native')

        rep = tick = 0
        while env.is_viewer_alive() and rep < repetition:
            landmarks = poses3d[tick]
            if landmarks is None:
                tick = (tick + 1) % len(poses3d)
                if tick == 0:
                    rep += 1
                continue

            # Root & transform (reuse same logic as BVH path)
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hip_center = (left_hip + right_hip) / 2.0

            p_root = hip_center * p_rate  # scale to match BVH visualisation
            R_root = np.eye(3)
            T_root = pr2t(p_root, R_root)
            T_root_rot = pr2t(np.zeros(3), rpy2r(np.radians(radians))) @ T_root
            p_root_rot, R_root_rot = t2pr(T_root_rot)

            env.remove_all_geoms()  # clear markers previously added
            env.plot_T()

            # Plot joints as spheres and bones as cylinders
            pts_global = []
            for lm in landmarks:
                p_local = (lm - hip_center) * p_rate
                p_glob = R_root_rot @ p_local + p_root_rot + np.array([0.0, 0.0, z_offset])
                pts_global.append(p_glob)
                env.plot_sphere(p=p_glob, r=0.01, rgba=[1, 0, 0, 0.8])

            for idx_fr, idx_to in pose_connections:
                if idx_fr < len(pts_global) and idx_to < len(pts_global):
                    env.plot_cylinder_fr2to(
                        p_fr=pts_global[idx_fr],
                        p_to=pts_global[idx_to],
                        r=0.01,
                        rgba=[0, 0, 1, 0.6],
                    )

            env.render()
            time.sleep(playback_speed)
            tick += 1
            if tick >= len(poses3d):
                tick = 0
                rep += 1

        env.close_viewer()
        print("Viewer closed.")


if __name__ == "__main__":
    import fire
    fire.Fire(run)

