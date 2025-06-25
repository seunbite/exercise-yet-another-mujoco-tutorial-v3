import os
import cv2
import numpy as np
import requests
from urllib.parse import urlparse

def download_video(url: str, output_path: str):
    """
    URL에서 비디오 파일(MP4, GIF 등)을 다운로드하여 로컬에 저장.
    """
    try:
        print(f"[download_video] Downloading from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[download_video] Video/GIF saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[download_video] Error downloading video: {e}")
        return None

def extract_frames(video_path: str, frames_dir: str, fps: int = 20):
    """
    비디오 또는 GIF에서 fps만큼의 속도로 프레임을 PNG로 저장.
    """
    os.makedirs(frames_dir, exist_ok=True)
    
    # Check if it's a GIF file
    if video_path.lower().endswith('.gif'):
        return extract_frames_from_gif(video_path, frames_dir, fps)
    else:
        return extract_frames_from_video(video_path, frames_dir, fps)

def extract_frames_from_gif(gif_path: str, frames_dir: str, fps: int = 20):
    """
    GIF 파일에서 프레임을 추출하여 PNG로 저장.
    """
    # Ensure output directory exists
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        from PIL import Image
        
        gif = Image.open(gif_path)
        
        # Check if GIF is animated
        try:
            gif.seek(1)  # Try to seek to second frame
            gif.seek(0)  # Go back to first frame
            is_animated = True
        except EOFError:
            is_animated = False
            print(f"[extract_frames_from_gif] Warning: GIF appears to be static (single frame)")
        
        if not is_animated:
            # Handle single frame GIF
            frame = gif.convert('RGB')
            fname = os.path.join(frames_dir, f"frame_00000.png")
            frame.save(fname)
            print(f"[extract_frames_from_gif] saved 1 frame (static GIF) to {frames_dir}")
            gif.close()
            return True
        
        # Get GIF properties for animated GIF
        total_frames = getattr(gif, 'n_frames', 1)
        print(f"[extract_frames_from_gif] Processing animated GIF: {total_frames} frames")
        
        saved = 0
        frame_idx = 0
        
        try:
            while True:
                # Get current frame
                current_frame = gif.copy()
                
                # Convert to RGB (handle palette mode, transparency, etc.)
                if current_frame.mode == 'P':
                    # Handle palette mode
                    current_frame = current_frame.convert('RGBA')
                    # Create white background
                    background = Image.new('RGB', current_frame.size, (255, 255, 255))
                    background.paste(current_frame, mask=current_frame.split()[-1] if 'transparency' in current_frame.info else None)
                    current_frame = background
                else:
                    current_frame = current_frame.convert('RGB')
                
                # Save frame
                fname = os.path.join(frames_dir, f"frame_{saved:05d}.png")
                current_frame.save(fname)
                saved += 1
                
                # Move to next frame
                frame_idx += 1
                gif.seek(frame_idx)
                
        except EOFError:
            # Reached end of GIF
            pass
        
        gif.close()
        print(f"[extract_frames_from_gif] saved {saved} frames to {frames_dir}")
        return saved > 0
            
    except ImportError:
        print("[extract_frames_from_gif] PIL/Pillow not available, trying with OpenCV...")
        return extract_frames_from_video(gif_path, frames_dir, fps)
    except Exception as e:
        print(f"[extract_frames_from_gif] Error processing GIF with PIL: {e}")
        print("Trying with OpenCV as fallback...")
        return extract_frames_from_video(gif_path, frames_dir, fps)

def extract_frames_from_video(video_path: str, frames_dir: str, fps: int = 20):
    """
    비디오 파일에서 fps만큼의 속도로 프레임을 PNG로 저장.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"[extract_frames_from_video] Error: Could not open video file {video_path}")
        return False
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps <= 0 or total_frames <= 0:
        print(f"[extract_frames_from_video] Error: Invalid video properties. FPS: {video_fps}, Total frames: {total_frames}")
        cap.release()
        return False
    
    frame_interval = max(1, int(round(video_fps / fps)))
    count, saved = 0, 0

    print(f"[extract_frames_from_video] Processing video: FPS={video_fps:.2f}, Total frames={total_frames}, Extract every {frame_interval} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            fname = os.path.join(frames_dir, f"frame_{saved:05d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"[extract_frames_from_video] saved {saved} frames to {frames_dir}")
    return saved > 0


def estimate_2d_pose(frames_dir: str):
    """
    MediaPipe로 각 프레임의 2D 관절(x,y)을 뽑아서 리스트로 반환.
    returns: List of shape (N_frames, N_joints, 2)
    """
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    frame_files = sorted(os.listdir(frames_dir))
    poses2d = []

    for fn in frame_files:
        img = cv2.imread(os.path.join(frames_dir, fn))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            poses2d.append(None)
        else:
            lm = res.pose_landmarks.landmark
            coords = np.array([[p.x, p.y] for p in lm])
            poses2d.append(coords)
    pose.close()
    print(f"[estimate_2d_pose] processed {len(poses2d)} frames")
    return poses2d


def lift_to_3d(poses2d: list):
    """
    VideoPose3D 같은 모델로 2D → 3D 복원.
    returns: List of shape (N_frames, N_joints, 3)
    """
    # >>> 여기에 VideoPose3D 모델 불러오기 및 추론 코드 삽입 <<<
    # 예시로 입력 리스트를 그대로 3채널로 확장
    poses3d = []
    for p in poses2d:
        if p is None:
            poses3d.append(None)
        else:
            # 깊이(z)에 0을 채워 예시 생성
            z = np.zeros((p.shape[0], 1))
            poses3d.append(np.concatenate([p, z], axis=1))
    print(f"[lift_to_3d] lifted {len(poses3d)} frames")
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



# --- 5. 전체 파이프라인 실행 함수 ---
def run(
    input_mp4: str = "https://i.makeagif.com/media/3-10-2023/oHTd3S.gif",  # Example direct video URL
    output_dir: str = "temp_mocap",
    fps: int = 20,
    template_bvh: str = "/home/iyy1112/workspace/exercise-yet-another-mujoco-tutorial-v3/bvh/cmu_mocap/05_14.bvh",
):
    """
    1) temp_mocap/frames 에 프레임 저장  
    2) 2D 포즈 추정 → 3D 리프팅  
    3) BVH 리타게팅 & output_bvh 저장
    """
    frames_dir = os.path.join(output_dir, "frames")
    bvh_dir = os.path.join(output_dir, "bvh")
    # Generate BVH output path based on input filename
    input_basename = os.path.basename(input_mp4)
    # Remove common video extensions and add .bvh
    for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm']:
        if input_basename.lower().endswith(ext):
            input_basename = input_basename[:-len(ext)]
            break
    bvh_path = os.path.join(bvh_dir, input_basename + ".bvh")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(bvh_dir, exist_ok=True)

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
    
    poses2d = estimate_2d_pose(frames_dir)
    poses3d = lift_to_3d(poses2d)
    retarget_and_save_bvh(poses3d, template_bvh, bvh_path, fps=fps)


if __name__ == "__main__":
    import fire

    fire.Fire(run)

