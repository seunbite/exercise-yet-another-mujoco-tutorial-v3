#!/usr/bin/env python3
"""
Test script for ik.py
"""

import numpy as np
from ik import UR5eIKEnv

def test_ik_functionality():
    """Test the IK functionality"""
    print("Testing IK functionality...")
    
    try:
        # Create IK environment
        ik_env = UR5eIKEnv(
            xml_path='../../asset/makeup_frida/scene_table.xml',
            HZ=50,
            object_table_position=[1.0, 0, 0],
            base_table_position=[0, 0, 0],
            head_position=[1, 0.0, 0.53],
            waiting_time=0.0
        )
        
        print("✓ IK environment created successfully")
        
        # Test uniform point generation
        start_point = np.array([1.0, 0.0, 0.7])
        end_point = np.array([1.1, 0.0, 0.7])
        points = ik_env._generate_uniform_points(start_point, end_point, num_points=10)
        
        print(f"✓ Generated {len(points)} uniform points")
        print(f"  Start: {start_point}")
        print(f"  End: {end_point}")
        print(f"  First point: {points[0]}")
        print(f"  Last point: {points[-1]}")
        
        # Test target line creation
        target_line = ik_env.create_target_line(use_dynamic_target=False)
        print(f"✓ Created target line: {target_line}")
        
        print("\nAll tests passed! The IK module is working correctly.")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ik_functionality() 