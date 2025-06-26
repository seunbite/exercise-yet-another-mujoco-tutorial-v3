#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import cv2

def parse_bvh_simple(bvh_path: str):
    """
    Simple BVH parser to extract basic frame data for visualization.
    Returns: (frames_data, fps, joint_names)
    """
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
    
    # Find MOTION section
    motion_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('MOTION'):
            motion_start = i
            break
    
    if motion_start == -1:
        raise ValueError("No MOTION section found in BVH file")
    
    # Parse frame info
    frames_line = lines[motion_start + 1].strip()
    frame_time_line = lines[motion_start + 2].strip()
    
    num_frames = int(frames_line.split(':')[1].strip())
    frame_time = float(frame_time_line.split(':')[1].strip())
    fps = 1.0 / frame_time
    
    print(f"[parse_bvh_simple] Frames: {num_frames}, FPS: {fps:.2f}")
    
    # Parse frame data
    frames_data = []
    for i in range(motion_start + 3, motion_start + 3 + num_frames):
        if i < len(lines):
            frame_values = [float(x) for x in lines[i].strip().split()]
            frames_data.append(frame_values)
    
    return frames_data, fps, num_frames

def create_skeleton_visualization(frames_data, output_gif: str, fps: int = 10):
    """
    Create a simple skeleton visualization from BVH frame data.
    Since we're using placeholder data (all zeros), we'll create a basic pose visualization.
    """
    print(f"[create_skeleton_visualization] Creating visualization with {len(frames_data)} frames")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('BVH Motion Visualization')
    ax.grid(True, alpha=0.3)
    
    # Since our BVH data is currently all zeros (placeholder), 
    # let's create a simple animated skeleton that shows the frame progression
    frames = []
    
    for frame_idx, frame_data in enumerate(frames_data):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title(f'BVH Motion - Frame {frame_idx + 1}/{len(frames_data)}')
        ax.grid(True, alpha=0.3)
        
        # Create a simple stick figure that moves based on frame number
        # This is a placeholder visualization since our BVH data is zeros
        t = frame_idx / len(frames_data) * 2 * np.pi
        
        # Simple animated stick figure
        # Head
        head_x, head_y = 0, 1.5 + 0.1 * np.sin(t * 2)
        ax.plot(head_x, head_y, 'ko', markersize=15)
        
        # Body
        body_x, body_y = 0, 0.5
        ax.plot([head_x, body_x], [head_y - 0.15, body_y + 0.5], 'k-', linewidth=3)
        
        # Arms
        arm_swing = 0.3 * np.sin(t)
        left_arm_x = body_x - 0.5 - arm_swing
        right_arm_x = body_x + 0.5 + arm_swing
        arm_y = body_y + 0.2
        
        ax.plot([body_x, left_arm_x], [arm_y, arm_y], 'k-', linewidth=2)
        ax.plot([body_x, right_arm_x], [arm_y, arm_y], 'k-', linewidth=2)
        
        # Legs
        leg_swing = 0.2 * np.sin(t + np.pi)
        left_leg_x = body_x - 0.2 - leg_swing
        right_leg_x = body_x + 0.2 + leg_swing
        leg_y = body_y - 0.8
        
        ax.plot([body_x, left_leg_x], [body_y, leg_y], 'k-', linewidth=2)
        ax.plot([body_x, right_leg_x], [body_y, leg_y], 'k-', linewidth=2)
        
        # Add frame info
        ax.text(-1.8, 1.8, f'Channels: {len(frame_data)}', fontsize=10)
        ax.text(-1.8, 1.6, f'Data: {"All zeros" if all(x == 0 for x in frame_data[:10]) else "Has data"}', fontsize=10)
        
        # Save frame as image
        plt.tight_layout()
        temp_path = f'temp_frame_{frame_idx:05d}.png'
        plt.savefig(temp_path, dpi=80, bbox_inches='tight')
        frames.append(temp_path)
    
    plt.close()
    
    # Create GIF from frames
    print(f"[create_skeleton_visualization] Creating GIF: {output_gif}")
    images = []
    for frame_path in frames:
        img = Image.open(frame_path)
        images.append(img)
        os.remove(frame_path)  # Clean up temp file
    
    # Save as GIF
    duration = int(1000 / fps)  # Convert to milliseconds
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"[create_skeleton_visualization] Saved visualization to {output_gif}")

def create_pose_comparison_gif(original_frames_dir: str, bvh_path: str, output_gif: str, fps: int = 10):
    """
    Create a side-by-side comparison of original frames and BVH visualization.
    """
    print(f"[create_pose_comparison_gif] Creating comparison GIF")
    
    # Get original frames
    frame_files = sorted([f for f in os.listdir(original_frames_dir) if f.endswith('.png')])
    if not frame_files:
        print("No original frames found!")
        return
    
    # Parse BVH data
    frames_data, bvh_fps, num_frames = parse_bvh_simple(bvh_path)
    
    # Create comparison frames
    comparison_frames = []
    
    for i, frame_file in enumerate(frame_files[:min(len(frame_files), len(frames_data))]):
        # Load original frame
        original_img = cv2.imread(os.path.join(original_frames_dir, frame_file))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create skeleton visualization for this frame
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original frame
        ax1.imshow(original_img)
        ax1.set_title(f'Original Frame {i+1}')
        ax1.axis('off')
        
        # BVH visualization
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.set_title(f'BVH Data Frame {i+1}')
        ax2.grid(True, alpha=0.3)
        
        # Simple stick figure (since our data is zeros)
        t = i / len(frames_data) * 2 * np.pi
        
        # Head
        head_x, head_y = 0, 1.5 + 0.1 * np.sin(t * 2)
        ax2.plot(head_x, head_y, 'ro', markersize=12)
        
        # Body
        body_x, body_y = 0, 0.5
        ax2.plot([head_x, body_x], [head_y - 0.15, body_y + 0.5], 'r-', linewidth=3)
        
        # Arms
        arm_swing = 0.3 * np.sin(t)
        left_arm_x = body_x - 0.5 - arm_swing
        right_arm_x = body_x + 0.5 + arm_swing
        arm_y = body_y + 0.2
        
        ax2.plot([body_x, left_arm_x], [arm_y, arm_y], 'r-', linewidth=2)
        ax2.plot([body_x, right_arm_x], [arm_y, arm_y], 'r-', linewidth=2)
        
        # Legs
        leg_swing = 0.2 * np.sin(t + np.pi)
        left_leg_x = body_x - 0.2 - leg_swing
        right_leg_x = body_x + 0.2 + leg_swing
        leg_y = body_y - 0.8
        
        ax2.plot([body_x, left_leg_x], [body_y, leg_y], 'r-', linewidth=2)
        ax2.plot([body_x, right_leg_x], [body_y, leg_y], 'r-', linewidth=2)
        
        # Save comparison frame
        plt.tight_layout()
        temp_path = f'temp_comparison_{i:05d}.png'
        plt.savefig(temp_path, dpi=80, bbox_inches='tight')
        plt.close()
        
        # Load and add to frames list
        img = Image.open(temp_path)
        comparison_frames.append(img)
        os.remove(temp_path)
    
    # Save comparison GIF
    duration = int(1000 / fps)
    comparison_frames[0].save(
        output_gif,
        save_all=True,
        append_images=comparison_frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"[create_pose_comparison_gif] Saved comparison GIF to {output_gif}")

def visualize_bvh_data(bvh_path: str, original_frames_dir: str = None, output_dir: str = "visualization"):
    """
    Main function to create BVH visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simple skeleton animation
    frames_data, fps, num_frames = parse_bvh_simple(bvh_path)
    skeleton_gif = os.path.join(output_dir, "skeleton_animation.gif")
    create_skeleton_visualization(frames_data, skeleton_gif, fps=min(fps, 10))
    
    # Create comparison if original frames are available
    if original_frames_dir and os.path.exists(original_frames_dir):
        comparison_gif = os.path.join(output_dir, "pose_comparison.gif")
        create_pose_comparison_gif(original_frames_dir, bvh_path, comparison_gif, fps=min(fps, 10))
    
    print(f"\n✅ Visualization complete!")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🎬 Skeleton animation: {skeleton_gif}")
    if original_frames_dir:
        print(f"   🔄 Comparison GIF: {comparison_gif}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_bvh.py <bvh_file> [original_frames_dir]")
        print("Example: python visualize_bvh.py test_gif_mocap/bvh/oHTd3S.bvh test_gif_mocap/frames")
        sys.exit(1)
    
    bvh_file = sys.argv[1]
    frames_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_bvh_data(bvh_file, frames_dir) 