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
    미리 준비한 BVH 템플릿 스켈레톤에 3D 관절 위치를 매핑하여 .bvh 파일로 저장.
    """
    from bvh import Bvh

    # 템플릿 불러오기
    with open(template_bvh, 'r') as f:
        template_content = f.read()
        template = Bvh(template_content)

    # Get number of channels from template - count actual channels
    num_channels = 0
    lines = template_content.split('\n')
    for line in lines:
        if 'CHANNELS' in line:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1].isdigit():
                num_channels += int(parts[1])
    
    print(f"[retarget_and_save_bvh] Template has {num_channels} channels")
    
    # Frames마다 채널 데이터 생성 (실제로는 IK/회전 계산 필요)
    frames_channels = []
    for p in poses3d:
        if p is None:
            # 빈 프레임: 제로 채널
            channels = [0.0] * num_channels
        else:
            # TODO: 각 Joint별 World→Local 회전(채널)으로 변환
            # For now, fill with zeros to match expected channel count
            channels = [0.0] * num_channels
        frames_channels.append(channels)

    # BVH 헤더 + 프레임 정보 조합
    header_end = template_content.find('MOTION')
    if header_end == -1:
        print("Warning: Invalid BVH template - no MOTION section found")
        return
    
    header = template_content[:header_end]
    motion_header = (
        "MOTION\n"
        f"Frames: {len(frames_channels)}\n"
        f"Frame Time: {1.0/fps:.6f}\n"
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_bvh), exist_ok=True)
    
    with open(output_bvh, 'w') as fout:
        fout.write(header + motion_header)
        for ch in frames_channels:
            fout.write(' '.join(f"{v:.6f}" for v in ch) + '\n')
    
    print(f"[retarget_and_save_bvh] Saved BVH with {len(frames_channels)} frames, {num_channels} channels per frame to {output_bvh}")


def run(
    input_mp4: str = "https://www.icegif.com/wp-content/uploads/2022/04/icegif-434.gif",
    output_dir: str = "temp_mocap",
    fps: int = 20,
    template_bvh: str = "/home/iyy1112/workspace/exercise-yet-another-mujoco-tutorial-v3/bvh/cmu_mocap/05_14.bvh",
    create_visualization: bool = True,  # New parameter to control GIF creation
):
    """
    1) temp_mocap/frames 에 프레임 저장  
    2) 2D 포즈 추정 → 3D 리프팅  
    3) BVH 리타게팅 & output_bvh 저장
    4) 시각화 GIF 생성 (선택사항)
    """
    frames_dir = os.path.join(output_dir, "frames")
    annotated_frames_dir = os.path.join(output_dir, "frames_with_pose")
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
    if os.path.exists(frames_dir):
    for i in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, i))
    os.makedirs(frames_dir, exist_ok=True)
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

    # Extract frames from video
    if not extract_frames(video_path, frames_dir, fps=fps):
        print("Frame extraction failed. Cannot proceed.")
        return
    
    poses_data = estimate_2d_pose(frames_dir)
    poses3d = lift_to_3d(poses_data)
    
    # Save 3D pose visualization frames for debugging
    print(f"\n🎯 Creating 3D pose visualization frames...")
    pose_3d_frames_dir = save_3d_pose_frames(poses3d, frames_dir)
    
    retarget_and_save_bvh(poses3d, template_bvh, bvh_path, fps=fps)
    
    # Create visualization GIFs
    if create_visualization:
        print(f"\n🎬 Creating visualization GIFs...")
        
        # Create 3D pose GIF from actual pose data (most important!)
        pose_3d_gif = os.path.join(viz_dir, f"{input_basename}_3d_pose_real.gif")
        if create_3d_pose_gif(pose_3d_frames_dir, pose_3d_gif, fps=min(fps, 10)):
            print(f"✅ Real 3D pose GIF saved: {pose_3d_gif}")
        
        
        print(f"\n📁 All outputs saved to: {output_dir}")
        print(f"   🎯 Original frames: {frames_dir}")
        print(f"   🎯 Annotated frames (with 2D pose): {annotated_frames_dir}")
        print(f"   🎯 3D pose frames: {pose_3d_frames_dir}")
        print(f"   📄 BVH file: {bvh_path}")
        print(f"   🎥 REAL 3D Pose GIF: {pose_3d_gif}")
    else:
        print(f"\n📁 All outputs saved to: {output_dir}")
        print(f"   🎯 Original frames: {frames_dir}")
        print(f"   🎯 Annotated frames (with 2D pose): {annotated_frames_dir}")
        print(f"   🎯 3D pose frames: {pose_3d_frames_dir}")
        print(f"   📄 BVH file: {bvh_path}")
    
    print(f"\n🎉 Motion capture pipeline completed!")
    return {
        "frames_dir": frames_dir,
        "annotated_frames_dir": annotated_frames_dir,
        "bvh_path": bvh_path,
        "visualization_dir": viz_dir if create_visualization else None
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
                    landmarks_3d = pose3d
                    
                    # Draw skeleton connections
                    for connection in pose_connections:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
                            start_point = landmarks_3d[start_idx]
                            end_point = landmarks_3d[end_idx]
                            
                            # Check if both points are valid (not zeros)
                            if not (np.allclose(start_point, 0) or np.allclose(end_point, 0)):
                                ax2.plot3D([start_point[0], end_point[0]], 
                                          [start_point[1], end_point[1]], 
                                          [start_point[2], end_point[2]], 'b-', linewidth=2)
                    
                    # Draw landmark points
                    valid_landmarks = 0
                    for j, landmark in enumerate(landmarks_3d):
                        if not np.allclose(landmark, 0):  # Skip zero landmarks
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
                    
                    # Set 3D plot properties
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([0, 1])
                    ax2.set_zlim([-0.5, 0.5])
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    ax2.set_zlabel('Z (depth)')
                    ax2.set_title(f'3D Pose Frame {i+1}\n({valid_landmarks} landmarks)')
                    
                    # Rotate view for better visualization
                    ax2.view_init(elev=15, azim=45 + i * 2)
                    
                    # Print debug info for first few frames
                    if i < 3:
                        print(f"[save_3d_pose_frames] Frame {i+1} 3D pose stats:")
                        print(f"  Shape: {landmarks_3d.shape}")
                        print(f"  Valid landmarks: {valid_landmarks}/33")
                        print(f"  X range: [{landmarks_3d[:, 0].min():.4f}, {landmarks_3d[:, 0].max():.4f}]")
                        print(f"  Y range: [{landmarks_3d[:, 1].min():.4f}, {landmarks_3d[:, 1].max():.4f}]")
                        print(f"  Z range: [{landmarks_3d[:, 2].min():.4f}, {landmarks_3d[:, 2].max():.4f}]")
                        
                        # Sample landmark coordinates
                        if valid_landmarks > 0:
                            nose = landmarks_3d[0]
                            left_shoulder = landmarks_3d[11]
                            right_shoulder = landmarks_3d[12]
                            print(f"  Nose: ({nose[0]:.4f}, {nose[1]:.4f}, {nose[2]:.4f})")
                            print(f"  L_shoulder: ({left_shoulder[0]:.4f}, {left_shoulder[1]:.4f}, {left_shoulder[2]:.4f})")
                            print(f"  R_shoulder: ({right_shoulder[0]:.4f}, {right_shoulder[1]:.4f}, {right_shoulder[2]:.4f})")
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

