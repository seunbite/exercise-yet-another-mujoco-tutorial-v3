#!/usr/bin/env python3
"""
Compare BVH retargeting vs Raw 3D keypoints visualization

This script demonstrates the difference between:
1. BVH retargeting (processed joint rotations)
2. Raw 3D keypoints (direct MediaPipe output)

Usage:
    python compare_bvh_vs_raw.py
"""

import fire
import os
import sys

def compare_demo(
    input_url: str = "https://i.pinimg.com/originals/58/f2/3c/58f23c62733b75711785a822cdb052c6.gif",
    output_dir: str = "temp_mocap_comparison",
    fps: int = 20
):
    """
    Run comparison between BVH retargeting and raw keypoints visualization.
    
    Args:
        input_url: URL or path to input video/GIF
        output_dir: Directory to save outputs
        fps: Frames per second for processing
    """
    
    print("=" * 60)
    print("🎬 MOTION CAPTURE COMPARISON DEMO")
    print("=" * 60)
    print(f"Input: {input_url}")
    print(f"Output: {output_dir}")
    print(f"FPS: {fps}")
    print()
    
    # Step 1: Process video and create both BVH and raw keypoints
    print("📊 Step 1: Processing video and extracting motion data...")
    print("-" * 50)
    
    # Import and run retarget_data
    from retarget_data import run
    
    results = run(
        input_mp4=input_url,
        output_dir=output_dir,
        fps=fps,
        create_visualization=True
    )
    
    if not results:
        print("❌ Failed to process video!")
        return
    
    bvh_path = results["bvh_path"]
    keypoints_path = os.path.join(output_dir, "keypoints_raw.pkl")
    
    print(f"✅ BVH file created: {bvh_path}")
    print(f"✅ Raw keypoints saved: {keypoints_path}")
    print()
    
    # Step 2: Show the difference
    print("🔍 Step 2: Understanding the difference...")
    print("-" * 50)
    print("📋 BVH FILE:")
    print("   - Contains joint rotations and transformations")
    print("   - Processed for humanoid skeleton structure")
    print("   - Used for robot retargeting in simulator")
    print("   - May have coordinate system transformations")
    print()
    print("📋 RAW KEYPOINTS:")
    print("   - Direct MediaPipe 3D landmark coordinates")
    print("   - Same data shown in 3D visualization GIF")
    print("   - No processing or transformations applied")
    print("   - Pure keypoint positions in space")
    print()
    
    # Step 3: Instructions for visualization
    print("🎮 Step 3: Visualization options...")
    print("-" * 50)
    print("To visualize BVH retargeting:")
    print(f"   python visual_retargeting.py --bvh_path='{bvh_path}'")
    print()
    print("To visualize raw keypoints:")
    print(f"   python visual_retargeting.py --keypoints_path='{keypoints_path}' just_bvh")
    print()
    
    # Step 4: Analysis
    print("🔬 Step 4: What to look for...")
    print("-" * 50)
    print("🔍 EXPECTED DIFFERENCES:")
    print("   1. Scale: BVH may be scaled differently")
    print("   2. Orientation: BVH may have different coordinate system")
    print("   3. Movement: BVH shows joint rotations, raw shows positions")
    print("   4. Timing: May have different frame rates or timing")
    print()
    print("🎯 DEBUGGING TIPS:")
    print("   - Compare the 3D visualization GIF with simulator output")
    print("   - Raw keypoints should match the GIF exactly")
    print("   - BVH retargeting may show different movements")
    print("   - Use just_bvh() to see raw keypoints in simulator")
    print()
    
    print("✅ Comparison setup complete!")
    print("=" * 60)
    
    return {
        "bvh_path": bvh_path,
        "keypoints_path": keypoints_path,
        "visualization_gif": results.get("visualization_gif")
    }

def demo_raw_keypoints(
    keypoints_path: str = "temp_mocap_comparison/keypoints_raw.pkl",
    max_frames: int = 30,
    playback_speed: float = 0.5
):
    """
    Demo function to show raw keypoints in simulator.
    
    Args:
        keypoints_path: Path to the raw keypoints pickle file
        max_frames: Maximum number of frames to display
        playback_speed: Speed of playback (seconds per frame)
    """
    
    print("🎮 Starting raw keypoints demonstration...")
    print(f"Loading from: {keypoints_path}")
    
    # Import and run just_bvh
    from visual_retargeting import just_bvh
    
    just_bvh(
        keypoints_path=keypoints_path,
        max_frames=max_frames,
        playback_speed=playback_speed,
        scale_factor=2.0,
        show_keypoints=True,
        show_connections=True
    )

def demo_bvh_retargeting(
    bvh_path: str = "temp_mocap_comparison/bvh/58f23c62733b75711785a822cdb052c6.bvh",
    max_frames: int = 30,
    playback_speed: float = 0.5
):
    """
    Demo function to show BVH retargeting in simulator.
    
    Args:
        bvh_path: Path to the BVH file
        max_frames: Maximum number of frames to display
        playback_speed: Speed of playback (seconds per frame)
    """
    
    print("🤖 Starting BVH retargeting demonstration...")
    print(f"Loading from: {bvh_path}")
    
    # Import and run main
    from visual_retargeting import main
    
    main(
        bvh_path=bvh_path,
        max_frames=max_frames,
        playback_speed=playback_speed,
        show_targets=True
    )

if __name__ == "__main__":
    fire.Fire() 