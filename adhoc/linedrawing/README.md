# Face Drawing Robot System

A modular system for drawing faces using facial landmark detection and robot inverse kinematics (IK) in MuJoCo.

## Overview

This system combines:
- **Face Landmark Detection**: Uses MediaPipe or OpenCV DNN for facial feature detection
- **Robot Control**: Leverages existing IK functionality from the codebase
- **Drawing Path Generation**: Converts facial landmarks to robot drawing trajectories
- **Modular Design**: Easy to import and extend for different use cases

## Features

- **Multiple Detection Methods**: MediaPipe (preferred) and OpenCV DNN fallback
- **Facial Feature Recognition**: Eyes, eyebrows, nose, lips, face oval, and contour detection
- **Robot IK Integration**: Uses existing `UR5eIKEnv` class for motion planning
- **Customizable Drawing**: Configurable drawing parameters and paths
- **Error Handling**: Robust error handling and fallback mechanisms

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_face_drawing.txt
```

2. Ensure MuJoCo and the existing IK system are properly set up in your environment.

## Quick Start

### Basic Usage

```python
from face_drawing import FaceLandmarkDetector, FaceDrawingRobot, download_image_from_url

# Download and detect face
image = download_image_from_url("https://example.com/face.jpg")
detector = FaceLandmarkDetector(method="mediapipe")
face_data = detector.detect_landmarks(image)

# Initialize robot and draw
robot = FaceDrawingRobot()
success = robot.draw_face(face_data, animate=True)
robot.close()
```

### Command Line Usage

```bash
python face_drawing.py --image_url="https://example.com/face.jpg" --method="mediapipe"
```

## Architecture

### Core Classes

#### `FaceLandmarkDetector`
Handles face detection and landmark extraction:
- **MediaPipe**: High-quality 468-point face mesh + face detection
- **OpenCV DNN**: Fallback with simplified landmark detection
- **Feature Extraction**: Eyes, eyebrows, nose, lips, face oval, and contour
- **Hybrid Approach**: Combines detection and mesh for better accuracy

#### `FaceDrawingRobot`
Manages robot control and drawing execution:
- **IK Integration**: Uses existing `UR5eIKEnv` for motion planning
- **Path Generation**: Converts landmarks to drawing trajectories
- **Drawing Execution**: Controls pen movement and drawing speed

#### Utility Functions
- **`download_image_from_url`**: Downloads images from URLs
- **Coordinate Conversion**: Pixel to world coordinate mapping

### Drawing Paths

The system generates different drawing paths for facial features:

- **Eyes**: Elliptical paths with configurable size
- **Eyebrows**: Curved eyebrow shapes
- **Nose**: Triangular shape with bridge and nostrils
- **Lips**: Detailed upper and lower lip curves
- **Face Oval**: Smooth oval face outline
- **Face Contour**: Detailed contour with interpolation

## Configuration

### Robot Parameters

```python
robot = FaceDrawingRobot(
    xml_path='asset/makeup_frida/scene_table.xml',
    drawing_surface_height=0.53,    # Height of drawing surface
    drawing_area_size=0.3,          # Size of drawing area
    pen_tip_offset=0.05             # Pen tip offset from surface
)
```

### Detection Parameters

```python
detector = FaceLandmarkDetector(
    method="mediapipe"               # "mediapipe" or "opencv_dnn"
)
```

## Examples

See `example_usage.py` for comprehensive examples:

1. **Local Image Processing**: Load and process local image files
2. **URL Image Download**: Download and process images from URLs
3. **Robot Setup**: Customize robot parameters
4. **Custom Paths**: Create and execute custom drawing trajectories

## Integration with Existing Code

The system is designed to work seamlessly with the existing codebase:

- **IK System**: Uses `UR5eIKEnv` from `ik.py`
- **MuJoCo Environment**: Compatible with existing scene files
- **Package Structure**: Follows the established package organization

## Customization

### Adding New Facial Features

```python
class CustomFaceDrawingRobot(FaceDrawingRobot):
    def _create_custom_feature_path(self, points):
        # Implement custom drawing path
        return custom_path
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom drawing paths
        self.drawing_paths['custom_feature'] = self._create_custom_feature_path
```

### Custom Detection Methods

```python
class CustomFaceDetector(FaceLandmarkDetector):
    def _detect_custom(self, image):
        # Implement custom detection logic
        return custom_face_data
```

## Troubleshooting

### Common Issues

1. **MediaPipe Import Error**: Falls back to OpenCV DNN automatically
2. **IK Solver Failures**: Check robot workspace and target positions
3. **Face Detection Failures**: Ensure good image quality and face visibility

### Debug Mode

Enable verbose logging by setting environment variables:
```bash
export FACE_DRAWING_DEBUG=1
```

## Performance Considerations

- **Detection Speed**: MediaPipe is faster than OpenCV DNN
- **Drawing Speed**: Configurable via `drawing_speed` parameter
- **Memory Usage**: Efficient landmark storage and processing

## Future Enhancements

- **Real-time Processing**: Video stream support
- **Multiple Faces**: Support for multiple face detection
- **Advanced Paths**: Bezier curves and smooth interpolation
- **Learning**: Adaptive drawing based on user preferences

## Contributing

1. Follow the existing code style and structure
2. Add comprehensive docstrings and type hints
3. Include example usage in `example_usage.py`
4. Update requirements and documentation as needed

## License

This system is part of the larger MuJoCo tutorial project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the existing IK system documentation
2. Review the example usage scripts
3. Examine the error messages and debug output
4. Ensure all dependencies are properly installed 