#!/usr/bin/env python3

import os
from retarget_data import run

def test_gif_processing():
    """
    Test the retargeting pipeline with GIF input
    """
    print("Testing GIF processing...")
    
    # Test with the GIF URL you provided
    gif_url = "https://i.makeagif.com/media/3-10-2023/oHTd3S.gif"
    
    # You can also test with a local GIF file:
    # gif_url = "/path/to/your/local/animation.gif"
    
    run(
        input_mp4=gif_url,  # Despite the parameter name, it now accepts GIFs too
        output_dir="test_gif_mocap",
        fps=10,  # Lower FPS for faster testing
        template_bvh="/home/iyy1112/workspace/exercise-yet-another-mujoco-tutorial-v3/bvh/cmu_mocap/05_14.bvh",
    )
    
    print("GIF processing test completed!")
    
    # Check if output files were created
    output_files = [
        "test_gif_mocap/frames",
        "test_gif_mocap/bvh",
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} created successfully")
            if os.path.isdir(file_path):
                file_count = len(os.listdir(file_path))
                print(f"  Contains {file_count} files")
        else:
            print(f"✗ {file_path} not found")

if __name__ == "__main__":
    test_gif_processing() 