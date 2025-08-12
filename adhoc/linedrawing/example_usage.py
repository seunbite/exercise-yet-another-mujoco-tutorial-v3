#!/usr/bin/env python3
"""
Example usage of the Face Drawing Robot system

This script demonstrates how to use the modular face drawing system
to draw faces from images using facial landmark detection and robot IK.
"""

from face_drawing import FaceLandmarkDetector, FaceDrawingRobot, download_image_from_url
import numpy as np
import cv2


def example_local_image():
    """Example using a local image file"""
    print("=== Example: Local Image ===")
    
    # You can replace this with your own image path
    image_path = "path/to/your/face_image.jpg"
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image from {image_path}")
            return
        
        # Detect face landmarks
        detector = FaceLandmarkDetector(method="mediapipe")
        face_data = detector.detect_landmarks(image)
        
        if face_data:
            print("✓ Face detected successfully!")
            print(f"Face bounding box: {face_data['face_bbox']}")
            
            # You can now use this face_data with the robot
            # robot = FaceDrawingRobot()
            # robot.draw_face(face_data)
        else:
            print("No face detected in the image")
            
    except Exception as e:
        print(f"Error in local image example: {e}")


def example_url_image():
    """Example using an image URL"""
    print("\n=== Example: Image URL ===")
    
    # Example image URL (replace with your own)
    image_url = "https://example.com/face_image.jpg"
    
    try:
        # Download and process image
        image = download_image_from_url(image_url)
        if image is None:
            print("Failed to download image")
            return
        
        # Detect face landmarks
        detector = FaceLandmarkDetector(method="opencv_dnn")  # Using OpenCV as fallback
        face_data = detector.detect_landmarks(image)
        
        if face_data:
            print("✓ Face detected successfully!")
            print(f"Detection method: {face_data['method']}")
            print(f"Face bounding box: {face_data['face_bbox']}")
            
            # Show detected features
            for feature_name, points in face_data['feature_points'].items():
                if points:
                    print(f"  {feature_name}: {len(points)} points")
        else:
            print("No face detected in the image")
            
    except Exception as e:
        print(f"Error in URL image example: {e}")


def example_robot_setup():
    """Example of setting up the drawing robot"""
    print("\n=== Example: Robot Setup ===")
    
    try:
        # Initialize robot with custom parameters
        robot = FaceDrawingRobot(
            xml_path='asset/makeup_frida/scene_table.xml',
            drawing_surface_height=0.53,
            drawing_area_size=0.3,
            pen_tip_offset=0.05
        )
        
        print("✓ Robot initialized successfully")
        print(f"Drawing surface height: {robot.drawing_surface_height}")
        print(f"Drawing area size: {robot.drawing_area_size}")
        print(f"Pen tip offset: {robot.pen_tip_offset}")
        
        # Clean up
        robot.close()
        print("✓ Robot resources cleaned up")
        
    except Exception as e:
        print(f"Error in robot setup example: {e}")


def example_custom_drawing_paths():
    """Example of creating custom drawing paths"""
    print("\n=== Example: Custom Drawing Paths ===")
    
    try:
        # Create a simple robot instance
        robot = FaceDrawingRobot()
        
        # Create a custom path (simple square)
        custom_path = [
            np.array([0.1, 0.1, 0.53]),
            np.array([0.1, -0.1, 0.53]),
            np.array([-0.1, -0.1, 0.53]),
            np.array([-0.1, 0.1, 0.53]),
            np.array([0.1, 0.1, 0.53])
        ]
        
        print("Custom drawing path created:")
        for i, point in enumerate(custom_path):
            print(f"  Point {i}: {point}")
        
        # You could execute this path with:
        # success = robot._execute_drawing_path(custom_path, animate=False)
        
        robot.close()
        
    except Exception as e:
        print(f"Error in custom drawing paths example: {e}")


def main():
    """Run all examples"""
    print("Face Drawing Robot - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_local_image()
    example_url_image()
    example_robot_setup()
    example_custom_drawing_paths()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use the system:")
    print("1. python face_drawing.py --image_url='YOUR_IMAGE_URL'")
    print("2. Or import the classes in your own code:")
    print("   from face_drawing import FaceLandmarkDetector, FaceDrawingRobot")


if __name__ == "__main__":
    main() 