import os
import cv2
import numpy as np
import requests
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from utils import download_video, extract_frames
import mediapipe as mp
import shutil
import pickle


def extract_frames_to_memory(video_path: str, fps: int = 20):
    """
    Extract video frames directly into memory as numpy arrays.
    Returns list of frames as numpy arrays.
    """
    print(f"[extract_frames_to_memory] Processing video: {video_path}")
    
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


def estimate_2d_pose_from_frames(frames: list, save_annotated: bool = False):
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
    
    # Use MediaPipe Holistic for comprehensive pose detection
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=2) as holistic:
        
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


def estimate_2d_pose(frames_dir: str, save_annotated: bool = True):
    """
    MediaPipe Holistic으로 각 프레임의 2D 관절(x,y,z)을 뽑아서 리스트로 반환.
    save_annotated=True이면 포즈가 표시된 프레임도 저장.
    returns: List of pose data dictionaries with landmark coordinates
    """
    # Use MediaPipe Holistic instead of just Pose for better results
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    
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
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    poses_data = []
    
    if not frame_files:
        print(f"[estimate_2d_pose] No image files found in {frames_dir}")
        return poses_data
    
    print(f"[estimate_2d_pose] Found {len(frame_files)} frames to process")
    
    # Create directory for annotated frames if saving
    if save_annotated:
        annotated_dir = os.path.join(os.path.dirname(frames_dir), "frames_with_pose")
        os.makedirs(annotated_dir, exist_ok=True)
        print(f"[estimate_2d_pose] Saving annotated frames to {annotated_dir}")

    detection_stats = {'detected': 0, 'failed': 0, 'errors': 0}
    pose_variations = []  # Track pose variations to detect if poses are changing
    
    # Use MediaPipe Holistic for comprehensive pose detection
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=2) as holistic:
        
        for i, fn in enumerate(frame_files):
            img_path = os.path.join(frames_dir, fn)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[estimate_2d_pose] Error: Could not read frame {fn}")
                    poses_data.append(None)
                    detection_stats['errors'] += 1
                    continue
                
                # Get image dimensions
                height, width = img.shape[:2]
                print(f"[estimate_2d_pose] Processing frame {i+1}/{len(frame_files)}: {fn} ({width}x{height})")
                
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process with MediaPipe Holistic
                results = holistic.process(image_rgb)
                
                # Extract pose landmarks
                frame_data = {'frame': i}
                
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
                    
                    # Track pose variation to detect if poses are actually changing
                    if len(poses_data) > 1 and poses_data[-2] is not None and 'pose_landmarks' in poses_data[-2]:
                        # Calculate difference between consecutive poses
                        prev_coords = poses_data[-2]['pose_landmarks'][:, :2]  # Use only x,y for variation
                        curr_coords = np.array(pose_coords)[:, :2]
                        diff = np.mean(np.abs(curr_coords - prev_coords))
                        pose_variations.append(diff)
                        if i < 5:  # Print first few for debugging
                            print(f"[estimate_2d_pose] Frame {i+1} pose variation from previous: {diff:.6f}")
                    
                    # Print some landmark coordinates for debugging (first few frames)
                    if i < 3:
                        nose = pose_coords[0]  # First landmark is nose
                        left_shoulder = pose_coords[11]  # Left shoulder
                        right_shoulder = pose_coords[12]  # Right shoulder
                        print(f"[estimate_2d_pose] Frame {i+1} sample landmarks:")
                        print(f"  Nose: ({nose[0]:.4f}, {nose[1]:.4f}, {nose[2]:.4f})")
                        print(f"  Left shoulder: ({left_shoulder[0]:.4f}, {left_shoulder[1]:.4f}, {left_shoulder[2]:.4f})")
                        print(f"  Right shoulder: ({right_shoulder[0]:.4f}, {right_shoulder[1]:.4f}, {right_shoulder[2]:.4f})")
                else:
                    poses_data.append(None)
                    detection_stats['failed'] += 1
                    print(f"[estimate_2d_pose] No pose detected in frame {fn}")
                
                # Save annotated frame if requested
                if save_annotated:
                    annotated_img = img.copy()
                    
                    if results.pose_landmarks:
                        # Draw pose landmarks and connections using MediaPipe's drawing utilities
                        mp_draw.draw_landmarks(
                            annotated_img,
                            results.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style()
                        )
                        
                        # Also draw hand landmarks if available
                        if results.left_hand_landmarks:
                            mp_draw.draw_landmarks(
                                annotated_img,
                                results.left_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_draw_styles.get_default_hand_landmarks_style()
                            )
                        
                        if results.right_hand_landmarks:
                            mp_draw.draw_landmarks(
                                annotated_img,
                                results.right_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_draw_styles.get_default_hand_landmarks_style()
                            )
                        
                        # Add frame info with more details
                        cv2.putText(annotated_img, f"Frame {i+1}/{len(frame_files)}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(annotated_img, f"Pose: DETECTED ({len(pose_landmarks_data)} landmarks)", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show visibility info
                        avg_visibility = np.mean([lm.visibility for lm in pose_landmarks_data])
                        cv2.putText(annotated_img, f"Avg Visibility: {avg_visibility:.3f}", 
                                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show sample coordinates
                        nose_coord = pose_landmarks_data[0]
                        cv2.putText(annotated_img, f"Nose: ({nose_coord.x:.3f}, {nose_coord.y:.3f})", 
                                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Show hand detection status
                        hand_status = []
                        if results.left_hand_landmarks:
                            hand_status.append("Left Hand")
                        if results.right_hand_landmarks:
                            hand_status.append("Right Hand")
                        if hand_status:
                            cv2.putText(annotated_img, f"Hands: {', '.join(hand_status)}", 
                                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Add "no pose detected" text
                        cv2.putText(annotated_img, f"Frame {i+1}/{len(frame_files)}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(annotated_img, f"Pose: NOT DETECTED", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Save annotated frame
                    annotated_path = os.path.join(annotated_dir, fn)
                    cv2.imwrite(annotated_path, annotated_img)
                    
            except Exception as e:
                print(f"[estimate_2d_pose] Error processing frame {fn}: {e}")
                poses_data.append(None)
                detection_stats['errors'] += 1
    
    # Print detailed statistics
    print(f"\n[estimate_2d_pose] Processing completed:")
    print(f"  Total frames: {len(frame_files)}")
    print(f"  Poses detected: {detection_stats['detected']}")
    print(f"  Detection failed: {detection_stats['failed']}")
    print(f"  Processing errors: {detection_stats['errors']}")
    print(f"  Detection rate: {detection_stats['detected']/len(frame_files)*100:.1f}%")
    
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
    
    if save_annotated:
        print(f"  Annotated frames saved to: {annotated_dir}")
    
    return poses_data


def lift_to_3d(poses_data: list):
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
        
        if 'pose_landmarks' in pose_data:
            # MediaPipe already provides x, y, z coordinates
            landmarks_3d = pose_data['pose_landmarks']  # Shape: (33, 3)
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
        else:
            # Fallback: create zero landmarks if no pose detected
            poses3d.append(None)
    
    detected_3d = sum(1 for p in poses3d if p is not None)
    print(f"[lift_to_3d] Processed {len(poses3d)} frames, {detected_3d} with 3D data")
    
    return poses3d


def retarget_and_save_bvh(poses3d: list, template_bvh: str, output_bvh: str, fps: int = 20):
    """
    Convert 3D pose landmarks to BVH joint rotations and save as BVH file.
    """
    print(f"[retarget_and_save_bvh] Converting {len(poses3d)} frames to BVH format")
    
    # Read template BVH to get structure
    with open(template_bvh, 'r') as f:
        template_content = f.read()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_bvh), exist_ok=True)
    
    # MediaPipe landmark indices (33 landmarks)
    MEDIAPIPE_LANDMARKS = {
        'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3, 'right_eye_inner': 4,
        'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7, 'right_ear': 8, 'mouth_left': 9,
        'mouth_right': 10, 'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
        'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29,
        'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32
    }
    
    # Filter out frames without valid pose data
    valid_poses = []
    for i, pose in enumerate(poses3d):
        if pose is not None and isinstance(pose, np.ndarray) and pose.shape[0] >= 33:
            valid_poses.append(pose)
        else:
            print(f"[retarget_and_save_bvh] Warning: Frame {i+1} has invalid pose data")
            # Create a T-pose as default
            if len(valid_poses) > 0:
                # Use previous valid pose
                valid_poses.append(valid_poses[-1].copy())
            else:
                # Create default T-pose
                t_pose = create_default_t_pose()
                valid_poses.append(t_pose)
    
    if not valid_poses:
        print("[retarget_and_save_bvh] Error: No valid poses found!")
        return
    
    print(f"[retarget_and_save_bvh] Processing {len(valid_poses)} valid poses")
    
    # Convert poses to BVH format
    bvh_frames = []
    
    for frame_idx, landmarks in enumerate(valid_poses):
        # Extract key landmark positions (normalized coordinates 0-1)
        try:
            # Get landmark positions
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
            
            # Calculate hip center (root position)
            hip_center = (left_hip + right_hip) / 2.0
            
            # Scale to real-world coordinates (approximate human size)
            # MediaPipe coordinates are normalized, so we scale to reasonable human proportions
            scale_factor = 170.0  # Approximate height in cm
            
            # Root translation (hip center position)
            root_x = (hip_center[0] - 0.5) * scale_factor * 0.5  # Center and scale
            root_y = hip_center[1] * scale_factor * 0.3  # Scale height
            root_z = hip_center[2] * scale_factor * 0.3  # Scale depth
            
            # Calculate joint rotations (simplified approach)
            # Root rotation (based on hip orientation)
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
            
            # Create BVH frame data (order matters - must match BVH hierarchy)
            # This is a simplified mapping - actual BVH files may have different joint orders
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
            
            # Ensure we have the right number of channels (pad or truncate as needed)
            target_channels = 63  # Common BVH channel count for humanoid
            if len(frame_data) < target_channels:
                frame_data.extend([0.0] * (target_channels - len(frame_data)))
            elif len(frame_data) > target_channels:
                frame_data = frame_data[:target_channels]
            
            bvh_frames.append(frame_data)
            
            # Debug first few frames
            if frame_idx < 3:
                print(f"[retarget_and_save_bvh] Frame {frame_idx+1}:")
                print(f"  Root pos: ({root_x:.2f}, {root_y:.2f}, {root_z:.2f})")
                print(f"  Hip orientation: {hip_orientation:.1f}°")
                print(f"  Left elbow: {left_elbow_angle:.1f}°")
                print(f"  Right elbow: {right_elbow_angle:.1f}°")
                print(f"  Left knee: {left_knee_angle:.1f}°")
                print(f"  Right knee: {right_knee_angle:.1f}°")
                
        except Exception as e:
            print(f"[retarget_and_save_bvh] Error processing frame {frame_idx+1}: {e}")
            # Use previous frame or zeros as fallback
            if bvh_frames:
                bvh_frames.append(bvh_frames[-1].copy())
            else:
                bvh_frames.append([0.0] * 63)
    
    # Create BVH content
    bvh_content = create_simple_bvh_content(bvh_frames, fps)
    
    # Save BVH file
    with open(output_bvh, 'w') as f:
        f.write(bvh_content)
    
    print(f"[retarget_and_save_bvh] Saved BVH with {len(bvh_frames)} frames to {output_bvh}")
    print(f"[retarget_and_save_bvh] Each frame has {len(bvh_frames[0]) if bvh_frames else 0} channels")


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


def save_3d_keypoints_raw(poses3d, output_path):
    """
    Save raw 3D keypoints to a pickle file for direct loading in simulator.
    poses3d: List of 3D pose data from MediaPipe
    output_path: Path to save the pickle file
    """
    print(f"[save_3d_keypoints_raw] Saving {len(poses3d)} frames of raw 3D keypoints")
    
    # Filter and prepare valid poses
    valid_poses = []
    for i, pose in enumerate(poses3d):
        if pose is not None and isinstance(pose, np.ndarray) and pose.shape[0] >= 33:
            valid_poses.append(pose)
        else:
            print(f"[save_3d_keypoints_raw] Warning: Frame {i+1} has invalid pose data")
            # Use previous valid pose or create default
            if len(valid_poses) > 0:
                valid_poses.append(valid_poses[-1].copy())
            else:
                # Create default T-pose
                t_pose = create_default_t_pose()
                valid_poses.append(t_pose)
    
    # Save raw keypoints
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(valid_poses, f)
    
    print(f"[save_3d_keypoints_raw] Saved {len(valid_poses)} valid poses to {output_path}")
    return output_path


def create_simple_bvh_content(frames_data, fps=20):
    """Create a simple BVH file content with proper hierarchy"""
    frame_time = 1.0 / fps
    
    bvh_header = f"""HIERARCHY
ROOT Hips
{{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {{
        OFFSET 0.00 5.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Spine1
        {{
            OFFSET 0.00 10.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Neck
            {{
                OFFSET 0.00 10.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Head
                {{
                    OFFSET 0.00 5.00 0.00
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {{
                        OFFSET 0.00 5.00 0.00
                    }}
                }}
            }}
            JOINT LeftShoulder
            {{
                OFFSET -10.00 8.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftArm
                {{
                    OFFSET -15.00 0.00 0.00
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT LeftForeArm
                    {{
                        OFFSET -15.00 0.00 0.00
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {{
                            OFFSET -10.00 0.00 0.00
                        }}
                    }}
                }}
            }}
            JOINT RightShoulder
            {{
                OFFSET 10.00 8.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightArm
                {{
                    OFFSET 15.00 0.00 0.00
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT RightForeArm
                    {{
                        OFFSET 15.00 0.00 0.00
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {{
                            OFFSET 10.00 0.00 0.00
                        }}
                    }}
                }}
            }}
        }}
    }}
    JOINT LeftUpLeg
    {{
        OFFSET -5.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {{
            OFFSET 0.00 -20.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {{
                OFFSET 0.00 -20.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.00 -5.00 10.00
                }}
            }}
        }}
    }}
    JOINT RightUpLeg
    {{
        OFFSET 5.00 0.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightLeg
        {{
            OFFSET 0.00 -20.00 0.00
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightFoot
            {{
                OFFSET 0.00 -20.00 0.00
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.00 -5.00 10.00
                }}
            }}
        }}
    }}
}}
MOTION
Frames: {len(frames_data)}
Frame Time: {frame_time:.6f}
"""
    
    # Add frame data
    for frame in frames_data:
        frame_line = ' '.join(f"{val:.6f}" for val in frame)
        bvh_header += frame_line + '\n'
    
    return bvh_header


def run(
    input_mp4: str = "https://i.pinimg.com/originals/58/f2/3c/58f23c62733b75711785a822cdb052c6.gif",
    output_dir: str = "temp_mocap",
    fps: int = 20,
    template_bvh: str = "/home/iyy1112/workspace/exercise-yet-another-mujoco-tutorial-v3/bvh/cmu_mocap/05_14.bvh",
    create_visualization: bool = True,  # New parameter to control GIF creation
):
    """
    1) Process video frames in memory (no frame saving)
    2) 2D 포즈 추정 → 3D 리프팅  
    3) BVH 리타게팅 & output_bvh 저장
    4) 시각화 GIF 생성 (선택사항)
    """
    bvh_dir = os.path.join(output_dir, "bvh")
    viz_dir = os.path.join(output_dir, "visualization")

    # Generate BVH output path based on input filename
    input_basename = os.path.basename(input_mp4)
    # Remove common video extensions and add .bvh
    for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm']:
        if input_basename.lower().endswith(ext):
            input_basename = input_basename[:-len(ext)]
            break
    bvh_path = os.path.join(bvh_dir, input_basename + ".bvh")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bvh_dir, exist_ok=True)
    if create_visualization:
        os.makedirs(viz_dir, exist_ok=True)

    # Check if input is URL or local file
    if input_mp4.startswith(('http://', 'https://')):
        # Download video/GIF from URL
        video_filename = os.path.basename(urlparse(input_mp4).path) or "video.mp4"
        
        # Preserve original extension if it's a supported format
        if not video_filename.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
            # If no extension or unsupported, try to detect from URL
            if '.gif' in input_mp4.lower():
                video_filename += '.gif'
            else:
                video_filename += '.mp4'
        
        video_path = download_video(input_mp4, os.path.join(output_dir, video_filename))
        if not video_path:
            print("Video/GIF download failed. Skipping processing.")
            return
    else:
        # Use local file path
        video_path = input_mp4
        if not os.path.exists(video_path):
            print(f"Local video/GIF file not found: {video_path}")
            return

    # Extract frames from video directly into memory
    print(f"🎬 Processing video frames in memory...")
    frames = extract_frames_to_memory(video_path, fps=fps)
    if not frames:
        print("Frame extraction failed. Cannot proceed.")
        return
    
    poses_data = estimate_2d_pose_from_frames(frames)
    poses3d = lift_to_3d(poses_data)
    
    retarget_and_save_bvh(poses3d, template_bvh, bvh_path, fps=fps)
    
    # Save raw keypoints for comparison/debugging
    keypoints_path = os.path.join(output_dir, "keypoints_raw.pkl")
    save_3d_keypoints_raw(poses3d, keypoints_path)
    print(f"✅ Raw 3D keypoints saved to: {keypoints_path}")
    print(f"   Use just_bvh() in visual_retargeting.py to visualize these keypoints!")
    
    # Create visualization GIF
    if create_visualization:
        print(f"\n🎬 Creating 3D pose visualization GIF...")
        pose_3d_gif = os.path.join(viz_dir, f"{input_basename}_3d_pose.gif")
        if create_3d_pose_gif_from_memory(poses3d, frames, pose_3d_gif, fps=min(fps, 10)):
            print(f"✅ 3D pose GIF saved: {pose_3d_gif}")
        
        print(f"\n📁 All outputs saved to: {output_dir}")
        print(f"   📄 BVH file: {bvh_path}")
        print(f"   🎥 3D Pose GIF: {pose_3d_gif}")
    else:
        print(f"\n📁 All outputs saved to: {output_dir}")
        print(f"   📄 BVH file: {bvh_path}")
    
    print(f"\n🎉 Motion capture pipeline completed!")
    return {
        "bvh_path": bvh_path,
        "visualization_gif": os.path.join(viz_dir, f"{input_basename}_3d_pose.gif") if create_visualization else None
    }


def debug_pose_estimation(frames_dir: str, max_frames: int = 5):
    """
    Debug function to test pose estimation on a few frames and print detailed results.
    """
    print(f"\n🔍 DEBUG: Testing pose estimation on first {max_frames} frames...")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not frame_files:
        print("No image files found!")
        return
    
    # Use MediaPipe Holistic for consistency
    mp_holistic = mp.solutions.holistic
    
    test_frames = frame_files[:min(max_frames, len(frame_files))]
    
    with mp_holistic.Holistic(
        static_image_mode=True, 
        min_detection_confidence=0.5, 
        model_complexity=2) as holistic:
        
        for i, fn in enumerate(test_frames):
            print(f"\n--- Testing frame {i+1}: {fn} ---")
            img_path = os.path.join(frames_dir, fn)
            
            # Load and analyze image
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Could not load image: {img_path}")
                continue
                
            height, width = img.shape[:2]
            print(f"📐 Image dimensions: {width}x{height}")
            
            # Check image content
            avg_brightness = np.mean(img)
            print(f"💡 Average brightness: {avg_brightness:.1f}")
            
            # Process with MediaPipe Holistic
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                print(f"✅ Pose detected! {len(landmarks)} landmarks found")
                
                # Print key landmark positions
                key_landmarks = {
                    'nose': 0,
                    'left_shoulder': 11,
                    'right_shoulder': 12,
                    'left_hip': 23,
                    'right_hip': 24
                }
                
                for name, idx in key_landmarks.items():
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        pixel_x = lm.x * width
                        pixel_y = lm.y * height
                        print(f"  {name}: ({lm.x:.4f}, {lm.y:.4f}, {lm.z:.4f}) = ({pixel_x:.1f}, {pixel_y:.1f}) pixels")
                        print(f"    visibility: {lm.visibility:.3f}")
                
                # Check for hand landmarks
                if results.left_hand_landmarks:
                    print(f"  ✅ Left hand detected ({len(results.left_hand_landmarks.landmark)} landmarks)")
                if results.right_hand_landmarks:
                    print(f"  ✅ Right hand detected ({len(results.right_hand_landmarks.landmark)} landmarks)")
                
                # Check for face landmarks
                if results.face_landmarks:
                    print(f"  ✅ Face detected ({len(results.face_landmarks.landmark)} landmarks)")
                    
            else:
                print("❌ No pose detected!")
                
                # Try to diagnose why no pose was detected
                print("🔍 Diagnosis:")
                if avg_brightness < 50:
                    print("  - Image might be too dark")
                elif avg_brightness > 200:
                    print("  - Image might be too bright/overexposed")
                
                # Check if image has reasonable contrast
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                contrast = np.std(gray)
                print(f"  - Contrast (std dev): {contrast:.1f}")
                if contrast < 20:
                    print("  - Very low contrast detected")
    
    print(f"\n🔍 Debug test completed!")


def save_3d_pose_frames(poses3d: list, frames_dir: str, output_dir: str = None):
    """
    Save 3D pose visualization as individual PNG frames for debugging.
    poses3d: List of 3D pose data from lift_to_3d
    frames_dir: Directory containing original frames
    output_dir: Directory to save 3D pose frames (default: frames_with_3d_pose)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(frames_dir), "frames_with_3d_pose")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[save_3d_pose_frames] Saving 3D pose frames to {output_dir}")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # MediaPipe pose connections for 3D visualization
    pose_connections = [
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
    
    # Landmark names for reference
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
        'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
        'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]
    
    created_frames = 0
    
    for i, (pose3d, frame_file) in enumerate(zip(poses3d, frame_files)):
        try:
            # Load original frame
            original_img = cv2.imread(os.path.join(frames_dir, frame_file))
            if original_img is None:
                continue
                
            # Create figure with 2 subplots: original + 3D pose
            fig = plt.figure(figsize=(15, 6))
            
            # Left subplot: Original frame
            ax1 = fig.add_subplot(121)
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            ax1.imshow(original_rgb)
            ax1.set_title(f'Original Frame {i+1}')
            ax1.axis('off')
            
            # Right subplot: 3D pose
            ax2 = fig.add_subplot(122, projection='3d')
            
            if pose3d is not None and len(pose3d) > 0:
                # Extract 3D coordinates
                if isinstance(pose3d, np.ndarray) and pose3d.shape[1] == 3:
                    # pose3d is shape (33, 3) with [x, y, z] coordinates
                    landmarks_3d = pose3d.copy()
                    
                    # Transform coordinates to make person stand upright
                    # MediaPipe: X=horizontal, Y=down, Z=depth
                    # We want: X=horizontal, Y=depth, Z=up (standing person)
                    landmarks_3d_transformed = np.zeros_like(landmarks_3d)
                    landmarks_3d_transformed[:, 0] = landmarks_3d[:, 0]  # X stays the same
                    landmarks_3d_transformed[:, 1] = landmarks_3d[:, 2]  # Y = original Z (depth)
                    landmarks_3d_transformed[:, 2] = 1 - landmarks_3d[:, 1]  # Z = 1 - original Y (flip Y to make head up)
                    
                    # Draw skeleton connections
                    for connection in pose_connections:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks_3d_transformed) and end_idx < len(landmarks_3d_transformed):
                            start_point = landmarks_3d_transformed[start_idx]
                            end_point = landmarks_3d_transformed[end_idx]
                            
                            # Check if both points are valid (not zeros in original)
                            if not (np.allclose(landmarks_3d[start_idx], 0) or np.allclose(landmarks_3d[end_idx], 0)):
                                ax2.plot3D([start_point[0], end_point[0]], 
                                          [start_point[1], end_point[1]], 
                                          [start_point[2], end_point[2]], 'b-', linewidth=2)
                    
                    # Draw landmark points
                    valid_landmarks = 0
                    for j, (landmark, original_landmark) in enumerate(zip(landmarks_3d_transformed, landmarks_3d)):
                        if not np.allclose(original_landmark, 0):  # Skip zero landmarks (check original)
                            color = 'r' if j == 0 else 'g'  # Red for nose, green for others
                            markersize = 50 if j == 0 else 30  # Larger nose
                            ax2.scatter(landmark[0], landmark[1], landmark[2], 
                                      c=color, s=markersize, alpha=0.8)
                            valid_landmarks += 1
                            
                            # Label key landmarks
                            if j in [0, 11, 12, 15, 16, 23, 24]:  # Key landmarks
                                ax2.text(landmark[0], landmark[1], landmark[2], 
                                        landmark_names[j].replace('_', '\n'), 
                                        fontsize=8, alpha=0.7)
                    
                    # Set 3D plot properties for upright person
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([-0.5, 0.5])  # Depth range
                    ax2.set_zlim([0, 1])       # Height range (head at top)
                    ax2.set_xlabel('X (Left-Right)')
                    ax2.set_ylabel('Y (Depth)')
                    ax2.set_zlabel('Z (Height)')
                    ax2.set_title(f'3D Pose Frame {i+1}\n({valid_landmarks} landmarks)')
                    
                    # Set view angle to show person from front/side, slightly elevated
                    ax2.view_init(elev=10, azim=-90 + i * 1)  # Rotate slowly around vertical axis
                    
                    # Print debug info for first few frames
                    if i < 3:
                        print(f"[save_3d_pose_frames] Frame {i+1} 3D pose stats (upright orientation):")
                        print(f"  Shape: {landmarks_3d_transformed.shape}")
                        print(f"  Valid landmarks: {valid_landmarks}/33")
                        print(f"  X range (Left-Right): [{landmarks_3d_transformed[:, 0].min():.4f}, {landmarks_3d_transformed[:, 0].max():.4f}]")
                        print(f"  Y range (Depth): [{landmarks_3d_transformed[:, 1].min():.4f}, {landmarks_3d_transformed[:, 1].max():.4f}]")
                        print(f"  Z range (Height): [{landmarks_3d_transformed[:, 2].min():.4f}, {landmarks_3d_transformed[:, 2].max():.4f}]")
                        
                        # Sample landmark coordinates (transformed)
                        if valid_landmarks > 0:
                            nose = landmarks_3d_transformed[0]
                            left_shoulder = landmarks_3d_transformed[11]
                            right_shoulder = landmarks_3d_transformed[12]
                            print(f"  Nose (upright): ({nose[0]:.4f}, {nose[1]:.4f}, {nose[2]:.4f})")
                            print(f"  L_shoulder (upright): ({left_shoulder[0]:.4f}, {left_shoulder[1]:.4f}, {left_shoulder[2]:.4f})")
                            print(f"  R_shoulder (upright): ({right_shoulder[0]:.4f}, {right_shoulder[1]:.4f}, {right_shoulder[2]:.4f})")
                else:
                    print(f"[save_3d_pose_frames] Frame {i+1}: Invalid pose3d format - {type(pose3d)}")
                    ax2.text(0.5, 0.5, 0, 'Invalid 3D Data', fontsize=12, ha='center')
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([0, 1])
                    ax2.set_zlim([0, 1])
                    ax2.set_title(f'3D Pose Frame {i+1}\n(Invalid Data)')
            else:
                # No pose detected
                ax2.text(0.5, 0.5, 0, 'No Pose Detected', fontsize=12, ha='center')
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                ax2.set_zlim([0, 1])
                ax2.set_title(f'3D Pose Frame {i+1}\n(No Pose)')
            
            # Save frame
            plt.tight_layout()
            output_path = os.path.join(output_dir, frame_file)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            created_frames += 1
            
        except Exception as e:
            print(f"[save_3d_pose_frames] Error processing frame {i+1}: {e}")
            plt.close()
    
    return output_dir


def create_3d_pose_gif_from_memory(poses3d: list, original_frames: list, output_gif: str, fps: int = 10):
    """
    Create 3D pose visualization GIF directly from memory without saving intermediate frames.
    poses3d: List of 3D pose data from lift_to_3d
    original_frames: List of original frame images (numpy arrays)
    output_gif: Path to save the output GIF
    fps: Frames per second for the GIF
    """
    print(f"[create_3d_pose_gif_from_memory] Creating 3D pose GIF with {len(poses3d)} frames")
    
    if not poses3d or not original_frames:
        print(f"[create_3d_pose_gif_from_memory] No pose data or frames provided")
        return False
    
    # MediaPipe pose connections for 3D visualization
    pose_connections = [
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
    
    # Landmark names for reference
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
            # Create figure with 2 subplots: original + 3D pose
            fig = plt.figure(figsize=(15, 6))
            
            # Left subplot: Original frame
            ax1 = fig.add_subplot(121)
            original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            ax1.imshow(original_rgb)
            ax1.set_title(f'Original Frame {i+1}')
            ax1.axis('off')
            
            # Right subplot: 3D pose
            ax2 = fig.add_subplot(122, projection='3d')
            
            if pose3d is not None and len(pose3d) > 0:
                # Extract 3D coordinates
                if isinstance(pose3d, np.ndarray) and pose3d.shape[1] == 3:
                    # pose3d is shape (33, 3) with [x, y, z] coordinates
                    landmarks_3d = pose3d.copy()
                    
                    # Transform coordinates to make person stand upright
                    # MediaPipe: X=horizontal, Y=down, Z=depth
                    # We want: X=horizontal, Y=depth, Z=up (standing person)
                    landmarks_3d_transformed = np.zeros_like(landmarks_3d)
                    landmarks_3d_transformed[:, 0] = landmarks_3d[:, 0]  # X stays the same
                    landmarks_3d_transformed[:, 1] = landmarks_3d[:, 2]  # Y = original Z (depth)
                    landmarks_3d_transformed[:, 2] = 1 - landmarks_3d[:, 1]  # Z = 1 - original Y (flip Y to make head up)
                    
                    # Draw skeleton connections
                    for connection in pose_connections:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks_3d_transformed) and end_idx < len(landmarks_3d_transformed):
                            start_point = landmarks_3d_transformed[start_idx]
                            end_point = landmarks_3d_transformed[end_idx]
                            
                            # Check if both points are valid (not zeros in original)
                            if not (np.allclose(landmarks_3d[start_idx], 0) or np.allclose(landmarks_3d[end_idx], 0)):
                                ax2.plot3D([start_point[0], end_point[0]], 
                                          [start_point[1], end_point[1]], 
                                          [start_point[2], end_point[2]], 'b-', linewidth=2)
                    
                    # Draw landmark points
                    valid_landmarks = 0
                    for j, (landmark, original_landmark) in enumerate(zip(landmarks_3d_transformed, landmarks_3d)):
                        if not np.allclose(original_landmark, 0):  # Skip zero landmarks (check original)
                            color = 'r' if j == 0 else 'g'  # Red for nose, green for others
                            markersize = 50 if j == 0 else 30  # Larger nose
                            ax2.scatter(landmark[0], landmark[1], landmark[2], 
                                      c=color, s=markersize, alpha=0.8)
                            valid_landmarks += 1
                            
                            # Label key landmarks
                            if j in [0, 11, 12, 15, 16, 23, 24]:  # Key landmarks
                                ax2.text(landmark[0], landmark[1], landmark[2], 
                                        landmark_names[j].replace('_', '\n'), 
                                        fontsize=8, alpha=0.7)
                    
                    # Set 3D plot properties for upright person
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([-0.5, 0.5])  # Depth range
                    ax2.set_zlim([0, 1])       # Height range (head at top)
                    ax2.set_xlabel('X (Left-Right)')
                    ax2.set_ylabel('Y (Depth)')
                    ax2.set_zlabel('Z (Height)')
                    ax2.set_title(f'3D Pose Frame {i+1}\n({valid_landmarks} landmarks)')
                    
                    # Set view angle to show person from front/side, slightly elevated
                    ax2.view_init(elev=10, azim=-90 + i * 1)  # Rotate slowly around vertical axis
                else:
                    ax2.text(0.5, 0.5, 0, 'Invalid 3D Data', fontsize=12, ha='center')
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([0, 1])
                    ax2.set_zlim([0, 1])
                    ax2.set_title(f'3D Pose Frame {i+1}\n(Invalid Data)')
            else:
                # No pose detected
                ax2.text(0.5, 0.5, 0, 'No Pose Detected', fontsize=12, ha='center')
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                ax2.set_zlim([0, 1])
                ax2.set_title(f'3D Pose Frame {i+1}\n(No Pose)')
            
            # Convert matplotlib figure to PIL Image
            plt.tight_layout()
            fig.canvas.draw()
            
            # Get the RGBA buffer from the figure
            # Use buffer_rgba() which works with all backends
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            buf = buf[:, :, :3]
            
            # Convert to PIL Image
            pil_img = Image.fromarray(buf)
            pil_images.append(pil_img)
            
            plt.close(fig)
            created_frames += 1
            
            if i < 3:  # Debug first few frames
                print(f"[create_3d_pose_gif_from_memory] Created frame {i+1}: {buf.shape}")
            
        except Exception as e:
            print(f"[create_3d_pose_gif_from_memory] Error processing frame {i+1}: {e}")
            plt.close('all')  # Clean up any open figures
    
    if not pil_images:
        print(f"[create_3d_pose_gif_from_memory] No images created for GIF")
        return False
    
    # Create GIF
    duration = int(1000 / fps)  # Convert to milliseconds
    pil_images[0].save(
        output_gif,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    print(f"[create_3d_pose_gif_from_memory] Created GIF with {len(pil_images)} frames: {output_gif}")
    return True


def create_3d_pose_gif(pose_frames_dir: str, output_gif: str, fps: int = 10):
    """
    Create a GIF from 3D pose visualization frames.
    """
    print(f"[create_3d_pose_gif] Creating GIF from {pose_frames_dir}")
    
    # Get all PNG files from the 3D pose frames directory
    frame_files = sorted([f for f in os.listdir(pose_frames_dir) if f.lower().endswith('.png')])
    
    if not frame_files:
        print(f"[create_3d_pose_gif] No PNG files found in {pose_frames_dir}")
        return False
    
    print(f"[create_3d_pose_gif] Found {len(frame_files)} frames to convert to GIF")
    
    # Load images
    images = []
    for frame_file in frame_files:
        img_path = os.path.join(pose_frames_dir, frame_file)
        img = Image.open(img_path)
        images.append(img)
    
    # Create GIF
    duration = int(1000 / fps)  # Convert to milliseconds
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"[create_3d_pose_gif] Created GIF with {len(images)} frames: {output_gif}")
    return True


if __name__ == "__main__":
    import fire

    fire.Fire(run)

