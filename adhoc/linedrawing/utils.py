import os
import cv2
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt

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

