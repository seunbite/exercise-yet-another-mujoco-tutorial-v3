import numpy as np
import cv2
import requests
from PIL import Image
import io
import os
import time
from typing import List, Tuple, Optional, Dict, Any
import json
import mediapipe as mp
from mjct.helper.transformation import rpy2r
# Import the existing IK functionality
from ik import UR5eIKEnv

def download_image_from_url(image_url: str) -> Optional[np.ndarray]:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Convert to PIL Image then to numpy array
        image = Image.open(io.BytesIO(response.content))
        image_array = np.array(image)
        
        # Convert RGB to BGR if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            if image_array.dtype == np.uint8:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        print(f"✓ Image downloaded successfully: {image_array.shape}")
        return image_array
        
    except Exception as e:
        print(f"Failed to download image: {e}")
        return None


class FaceLandmarkDetector:
    """Face landmark detection using MediaPipe or OpenCV DNN"""
    
    def __init__(self, method: str = "mediapipe"):
        """
        Initialize face landmark detector
        
        Args:
            method: "mediapipe" or "opencv_dnn"
        """
        self.method = method
        self.face_mesh = None
        self.face_detector = None
        self.mp = None
        
        if method == "mediapipe":
            self.mp = mp
            
            # Initialize face mesh for detailed landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            # Initialize face detection for bounding box and key points
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.6
            )
            
            print("✓ MediaPipe face mesh and detection initialized")
        
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect face landmarks in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing landmarks and face information
        """
        if self.method == "mediapipe":
            return self._detect_mediapipe(image)
        else:
            return self._detect_opencv_dnn(image)
    
    def _detect_mediapipe(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect landmarks using MediaPipe with improved face mesh and detection"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # First, use face detection to get bounding box and key points
        detection_results = self.face_detector.process(rgb_image)
        face_bbox = None
        key_points = {}
        
        if detection_results.detections:
            face = detection_results.detections[0]
            confidence = face.score[0]
            
            # Get bounding box
            bounding_box = face.location_data.relative_bounding_box
            x = int(bounding_box.xmin * w)
            y = int(bounding_box.ymin * h)
            width = int(bounding_box.width * w)
            height = int(bounding_box.height * h)
            face_bbox = [x, y, width, height]
            
            # Get key points from face detection
            landmarks = face.location_data.relative_keypoints
            key_points = {
                'right_eye': (int(landmarks[0].x * w), int(landmarks[0].y * h)),
                'left_eye': (int(landmarks[1].x * w), int(landmarks[1].y * h)),
                'nose': (int(landmarks[2].x * w), int(landmarks[2].y * h)),
                'mouth': (int(landmarks[3].x * w), int(landmarks[3].y * h)),
                'right_ear': (int(landmarks[4].x * w), int(landmarks[4].y * h)),
                'left_ear': (int(landmarks[5].x * w), int(landmarks[5].y * h))
            }
        
        # Then, use face mesh for detailed landmarks
        mesh_results = self.face_mesh.process(rgb_image)
        
        if not mesh_results.multi_face_landmarks:
            # Fallback to detection-only results if mesh fails
            if face_bbox:
                return self._create_fallback_features(key_points, face_bbox, w, h)
            return None
        
        landmarks = mesh_results.multi_face_landmarks[0]
        
        # Extract all landmark points
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmark_points.append([x, y])
        
        # Use MediaPipe's predefined facial areas for better feature extraction
        facial_areas = {
            'left_eye': self.mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
            'right_eye': self.mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
            'left_eyebrow': self.mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
            'right_eyebrow': self.mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
            'lips': self.mp.solutions.face_mesh.FACEMESH_LIPS,
            'face_oval': self.mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
        }
        
        # Extract feature points using the predefined areas
        feature_points = {}
        for feature_name, area_indices in facial_areas.items():
            if feature_name == 'face_contour':
                # For contour, use all points to create a smooth outline
                feature_points[feature_name] = self._extract_contour_points(landmarks, w, h)
            else:
                # For other features, extract points from the mesh
                feature_points[feature_name] = self._extract_feature_points(landmarks, area_indices, w, h)
        
        # Add nose using key points from detection for better accuracy
        if 'nose' in key_points:
            feature_points['nose'] = [list(key_points['nose'])]
        
        # Use detection bounding box if mesh bbox is not available
        if not face_bbox:
            face_bbox = self._get_face_bbox(landmark_points)
        
        return {
            'landmarks': landmark_points,
            'feature_points': feature_points,
            'face_bbox': face_bbox,
            'key_points': key_points,
            'method': 'mediapipe',
            'confidence': confidence if 'confidence' in locals() else 0.0
        }
    
    def _extract_feature_points(self, landmarks, area_indices, w, h):
        """Extract points for a specific facial feature area"""
        points = []
        for source_idx, target_idx in area_indices:
            if source_idx < len(landmarks.landmark):
                source = landmarks.landmark[source_idx]
                x = int(source.x * w)
                y = int(source.y * h)
                points.append([x, y])
        return points
    
    def _extract_contour_points(self, landmarks, w, h):
        """Extract contour points for face outline"""
        # Use a subset of contour points for smoother drawing
        contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        points = []
        for idx in contour_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
        return points
    
    def _create_fallback_features(self, key_points, face_bbox, w, h):
        """Create feature points when face mesh fails but detection succeeds"""
        feature_points = {}
        
        # Create basic features from key points
        if 'left_eye' in key_points:
            x, y = key_points['left_eye']
            feature_points['left_eye'] = [[x, y]]
        
        if 'right_eye' in key_points:
            x, y = key_points['right_eye']
            feature_points['right_eye'] = [[x, y]]
        
        if 'nose' in key_points:
            x, y = key_points['nose']
            feature_points['nose'] = [[x, y]]
        
        if 'mouth' in key_points:
            x, y = key_points['mouth']
            feature_points['mouth'] = [[x, y]]
        
        # Create face contour from bounding box
        x, y, width, height = face_bbox
        feature_points['face_contour'] = [
            [x, y], [x + width, y], [x + width, y + height], [x, y + height]
        ]
        
        return {
            'landmarks': [],
            'feature_points': feature_points,
            'face_bbox': face_bbox,
            'key_points': key_points,
            'method': 'mediapipe_fallback',
            'confidence': 0.0
        }
    
    def _get_face_bbox(self, landmarks: List[List[int]]) -> List[int]:
        """Get bounding box from landmarks"""
        if not landmarks:
            return [0, 0, 0, 0]
        
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    


    def save_landmark_visualization(self, image: np.ndarray, face_data: Dict[str, Any], output_path: str):
        """Save image with landmark visualization"""
        # Create a copy of the image for drawing
        vis_image = image.copy()
        
        # Draw face bounding box
        if 'face_bbox' in face_data and face_data['face_bbox']:
            x, y, w, h = face_data['face_bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw key points
        if 'key_points' in face_data:
            for point_name, (px, py) in face_data['key_points'].items():
                cv2.circle(vis_image, (px, py), 5, (255, 0, 0), -1)
                cv2.putText(vis_image, point_name, (px + 10, py - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw feature points
        if 'feature_points' in face_data:
            colors = {
                'left_eye': (0, 255, 255),      # Yellow
                'right_eye': (0, 255, 255),     # Yellow
                'left_eyebrow': (255, 0, 255),  # Magenta
                'right_eyebrow': (255, 0, 255), # Magenta
                'nose': (255, 255, 0),          # Cyan
                'lips': (0, 0, 255),            # Red
                'mouth': (0, 0, 255),           # Red
                'face_oval': (128, 0, 128),     # Purple
                'face_contour': (0, 128, 128)   # Teal
            }
            
            for feature_name, points in face_data['feature_points'].items():
                if not points:
                    continue
                
                color = colors.get(feature_name, (128, 128, 128))
                
                # Draw points for this feature
                for point in points:
                    if len(point) >= 2:
                        px, py = int(point[0]), int(point[1])
                        cv2.circle(vis_image, (px, py), 2, color, -1)
                
                # Draw feature name
                if points and len(points[0]) >= 2:
                    center_x = int(np.mean([p[0] for p in points if len(p) >= 2]))
                    center_y = int(np.mean([p[1] for p in points if len(p) >= 2]))
                    cv2.putText(vis_image, feature_name, (center_x, center_y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save the visualization
        cv2.imwrite(output_path, vis_image)
        print(f"✓ Landmark visualization saved to: {output_path}")


class FaceDrawingRobot:
    def __init__(
        self,
        xml_path: str = 'asset/makeup_frida/scene_table.xml',
        drawing_surface_height: float = 0.53,
        drawing_area_size: float = 0.3,
        pen_tip_offset: float = 0.05
    ):
        self.drawing_surface_height = drawing_surface_height
        self.drawing_area_size = drawing_area_size
        self.pen_tip_offset = pen_tip_offset
        
        self.ik_env = UR5eIKEnv(xml_path=xml_path)
        
        if hasattr(self.ik_env, 'env') and hasattr(self.ik_env.env, 'viewer'):
            if self.ik_env.env.viewer is not None:
                print("✓ MuJoCo viewer initialized successfully")
                if hasattr(self.ik_env.env.viewer, 'cam'):
                    self.ik_env.env.viewer.cam.distance = 2.0
                    self.ik_env.env.viewer.cam.azimuth = 45
                    self.ik_env.env.viewer.cam.elevation = -30
                    self.ik_env.env.viewer.cam.lookat = np.array([0.5, 0, 0.5])
                self.ik_env.env.render()
            else:
                print("Warning: MuJoCo viewer not available")
        
        self.drawing_paths = {}
        self.face_visualization = None
        self.ik_env.reset()
        
        # Drawing parameters
        self.dot_n = 5  # Number of interpolation points between two positions
        self.lift_height = 0.02  # Height to lift pen between strokes
    
    def _sort_points_by_proximity(self, points: List[List[float]]) -> List[List[float]]:
        """
        점들을 인접한 순서로 정렬합니다 (Nearest Neighbor 알고리즘 사용)
        
        Args:
            points: 3D 점들의 리스트 [[x, y, z], ...]
            
        Returns:
            정렬된 점들의 리스트
        """
        if len(points) <= 2:
            return points
        
        points = np.array(points)
        sorted_points = []
        remaining_points = points.copy()
        
        # 시작점: 첫 번째 점
        current_point = remaining_points[0]
        sorted_points.append(current_point.tolist())
        remaining_points = np.delete(remaining_points, 0, axis=0)
        
        # Nearest Neighbor로 다음 점 찾기
        while len(remaining_points) > 0:
            # 현재 점에서 가장 가까운 점 찾기
            distances = np.linalg.norm(remaining_points - current_point, axis=1)
            nearest_idx = np.argmin(distances)
            
            # 가장 가까운 점을 다음 점으로 선택
            next_point = remaining_points[nearest_idx]
            sorted_points.append(next_point.tolist())
            
            # 현재 점 업데이트 및 사용된 점 제거
            current_point = next_point
            remaining_points = np.delete(remaining_points, nearest_idx, axis=0)
        
        return sorted_points
    
    def _sort_contour_points_clockwise(self, points: List[List[float]]) -> List[List[float]]:
        """
        윤곽선 점들을 시계방향으로 정렬합니다
        
        Args:
            points: 2D 또는 3D 점들의 리스트
            
        Returns:
            시계방향으로 정렬된 점들의 리스트
        """
        if len(points) <= 2:
            return points
        
        points = np.array(points)
        
        # 2D 점들로 변환 (z 좌표가 있다면 무시)
        if points.shape[1] == 3:
            points_2d = points[:, :2]
        else:
            points_2d = points
        
        # 중심점 계산
        center = np.mean(points_2d, axis=0)
        
        # 각 점의 중심점으로부터의 각도 계산 (시계방향: -π에서 π로)
        angles = []
        for point in points_2d:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            # atan2는 -π에서 π 범위의 각도를 반환
            # 시계방향으로 정렬하려면 -y, x 순서로 계산
            angle = np.arctan2(-dy, dx)
            angles.append(angle)
        
        # 각도에 따라 정렬 (시계방향)
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        return sorted_points.tolist()

    def draw_face(self, face_data: Dict[str, Any], rotation: Tuple[float, float, float] = (0, 90, 0), scale: float = 1.0, position: Tuple[float, float, float] = (0.8, 0, 0.53)) -> bool:
        """얼굴을 그립니다. 각 특징점을 순차적으로 그리고, 한 특징점이 완료되면 다음으로 넘어갑니다."""
        if not face_data or 'feature_points' not in face_data:
            print("Error: Invalid face data structure")
            return False
        
        # 1. 얼굴 데이터를 시뮬레이션 좌표로 변환
        print(f"\nTransforming face data with rotation {rotation}° and scale {scale} (position: {position})...")
        self.face_data_in_simul = self.face_data_to_target(face_data, rotation=rotation, scale=scale, position=position)
        
        total_points = sum(len(points) for points in self.face_data_in_simul['feature_points'].values() if points)
        completed_points = 0
        
        for feature_name, points in self.face_data_in_simul['feature_points'].items():
            if not points:
                continue
                
            print(f"\n--- Drawing {feature_name} ({len(points)} points) ---")
            
            # 현재 특징점의 모든 점들을 순차적으로 그리기
            feature_success = self._draw_feature_contour(feature_name, points)
            
            if feature_success:
                completed_points += len(points)
                print(f"✓ {feature_name} completed ({completed_points}/{total_points} points)")
                
            else:
                print(f"✗ {feature_name} failed")
        
        return completed_points > 0
    
    def _draw_feature_contour(self, feature_name: str, points: List[List[float]]) -> bool:
        if not points:
            return False
        
        print(f"  Drawing {len(points)} points for {feature_name}...")
        
        successful_points = 0
        
        # 각 점을 순차적으로 그리기
        for i, point in enumerate(points):
            current_point = self.ik_env.env.get_p_body(body_name='applicator_tip')
            intermediate_points = self._plan_movement(current_point, np.array(point), dot_n=self.dot_n)
            point_success = False

            # 중간점들을 따라 이동
            for j, intermediate_point in enumerate(intermediate_points):
                joint_config = self.ik_env.solve_ik_for_point(intermediate_point)
                
                if joint_config is not None:
                    # 조인트 설정 및 시뮬레이션 업데이트
                    joint_idxs = self.ik_env.env.get_idxs_fwd(joint_names=self.ik_env.joint_names)
                    self.ik_env.env.set_qpos_joints(self.ik_env.joint_names, joint_config)
                    self.ik_env.env.forward()
                    
                    self.ik_env.env.render()
                    self._add_target_visualization(point, color="green", size=0.005)
                    self._add_target_visualization(intermediate_points, color="blue", size=0.005)
                    self._add_target_visualization(points, color="black", size=0.005)
                    
                else:
                    print(f"    Warning: IK failed for intermediate point {j+1}/{len(intermediate_points)}")
                    continue
            
            # 마지막 중간점에 도달한 후, 목표 지점에 충분히 가까워질 때까지 IK 반복
            if len(intermediate_points) > 0:
                max_ik_attempts = 50  # 최대 IK 시도 횟수
                ik_attempts = 0
                
                while ik_attempts < max_ik_attempts:
                    current_tip_pos = self.ik_env.env.get_p_body(body_name='applicator_tip')
                    distance_to_target = np.linalg.norm(current_tip_pos - np.array(point))
                    
                    if distance_to_target < 0.01:  # 1mm 이내에 도달
                        point_success = True
                        successful_points += 1
                        print(f"    ✓ Completed point {i+1}/{len(points)} (distance: {distance_to_target:.4f}m, IK attempts: {ik_attempts})")
                        break
                    
                    # 목표 지점으로 IK 해결
                    joint_config = self.ik_env.solve_ik_for_point(np.array(point))
                    
                    if joint_config is not None:
                        # 조인트 설정 및 시뮬레이션 업데이트
                        joint_idxs = self.ik_env.env.get_idxs_fwd(joint_names=self.ik_env.joint_names)
                        self.ik_env.env.set_qpos_joints(self.ik_env.joint_names, joint_config)
                        self.ik_env.env.forward()
                        
                        # 시각화 업데이트
                        self.ik_env.env.render()
                        self._add_target_visualization(point, color="green", size=0.005)
                        self._add_target_visualization(intermediate_points, color="blue", size=0.005)
                        self._add_target_visualization(points, color="black", size=0.005)
                        
                        ik_attempts += 1
                    else:
                        print(f"    Warning: IK failed for final positioning of point {i+1}/{len(points)}")
                        break
                
                if not point_success:
                    print(f"    ✗ Point {i+1}/{len(points)} not reached after {ik_attempts} IK attempts (final distance: {distance_to_target:.4f}m)")
            else:
                print(f"    ✗ No intermediate points generated for point {i+1}/{len(points)}")
        
        print(f"  {feature_name}: {successful_points}/{len(points)} points drawn successfully")
        return successful_points > 0
    

    def _plan_movement(self, start_point: np.ndarray, end_point: np.ndarray, dot_n: int = 5) -> List[np.ndarray]:
        """두 지점 사이의 중간점들을 생성합니다."""
        intermediate_points = []
        
        # 시작점과 끝점이 너무 가까우면 중간점을 줄임
        distance = np.linalg.norm(end_point - start_point)
        if distance < 0.01:  # 1cm 이하면 중간점 없이
            return [start_point, end_point]
        
        for i in range(dot_n + 1):
            t = i / dot_n
            # 부드러운 보간을 위해 ease-in-out 적용
            if t <= 0.5:
                t_smooth = 2 * t * t
            else:
                t_smooth = 1 - 2 * (1 - t) * (1 - t)
            
            interp_point = start_point * (1.0 - t_smooth) + end_point * t_smooth
            intermediate_points.append(interp_point)
        
        return intermediate_points


    def _add_target_visualization(self, target_point: np.ndarray | List[np.ndarray], color: str = "red", size: float = 0.01):
        if color == "red":
            color_rgba = [1, 0, 0, 1]
        elif color == "green":
            color_rgba = [0, 1, 0, 1]
        elif color == "blue":
            color_rgba = [0, 0, 1, 1]
        elif color == "black":
            color_rgba = [0, 0, 0, 1]
            
        target_points = np.array(target_point)
        if target_points.shape == (3,):
            target_points = [target_points]
        elif target_points.shape == (2, 3):
            target_points = target_points
            
        for point in target_points:
            self.ik_env.env.plot_sphere(p=point, r=size, rgba=color_rgba)
    
    def face_data_to_target(self, face_data: Dict[str, Any], rotation=(0, 0, 90), scale=0.1, position=(0.8, 0, 0.53)) -> Dict[str, Any]:
        if not isinstance(position, (list, tuple, np.ndarray)) or len(position) < 3:
            assert False, "position must be a list, tuple, or numpy array with at least 3 elements"
        if not isinstance(rotation, (list, tuple, np.ndarray)) or len(rotation) < 3:
            assert False, "rotation must be a list, tuple, or numpy array with at least 3 elements"

        # sort
        new_feature_points = {}
        for feature_name, points in face_data['feature_points'].items():
            if feature_name in ['face_contour', 'face_oval']:
                points = self._sort_contour_points_clockwise(points)
            else:
                points = self._sort_points_by_proximity(points)
            new_feature_points[feature_name] = points

        # transform
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, scale]
        ])
        
        rotation_matrix = rpy2r(np.radians(rotation))
        
        x, y, width, height = face_data['face_bbox']
        face_center = np.array([x + width/2, y + height/2, 0])
        
        for feature_name, points in new_feature_points.items():
            if not points:
                continue
                
            points_array = np.array(points)
            if points_array.shape[1] == 2:
                points_array = np.hstack([points_array, np.zeros((len(points_array), 1))])
            
            centered_points = points_array - face_center
            scaled_points = centered_points @ scale_matrix.T
            rotated_points = scaled_points @ rotation_matrix.T
            final_points = rotated_points + np.array(position)
            new_feature_points[feature_name] = final_points.tolist()
        
        transformed_face_data = {
            'feature_points': new_feature_points,
        }
        
        return transformed_face_data

    def close(self):
        if hasattr(self, 'ik_env'):
            self.ik_env.close()


def main(
    image_url: str = 'https://encrypted-tbn1.gstatic.com/licensed-image?q=tbn:ANd9GcTu-GSY_bXjggu92Go8I0Od4bEoIE-RnSuaCRmN5xcL4lfSDQI169Wyg5hK0VegSLUJjpqlG47veDZ33C0',
    xml_path: str = 'asset/makeup_frida/scene_table.xml',
    method: str = "mediapipe",
    rotation: Tuple[float, float, float] = (90, 180, -90),
    position: Tuple[float, float, float] = (0.8, 0, 0.7),
    scale: float = 0.0005,
    tmp_folder: str = "tmp"
):
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. Download image
    image = download_image_from_url(image_url)
    if image is None:
        print("Failed to download image. Exiting.")
        return
    
    original_image_path = os.path.join(tmp_folder, f"original_image.png")
    cv2.imwrite(original_image_path, image)
    print(f"✓ Original image saved to: {original_image_path}")
    
    # 2. Detect face landmarks
    detector = FaceLandmarkDetector(method=method)
    face_data = detector.detect_landmarks(image)
    
    if face_data is None:
        print("No face detected in the image. Exiting.")
        return
    
    print("Detected facial features:")
    for feature_name, points in face_data['feature_points'].items():
        if points:
            print(f"  {feature_name}: {len(points)} points")
    
    landmark_vis_path = os.path.join(tmp_folder, f"landmark_visualization.png")
    detector.save_landmark_visualization(image, face_data, landmark_vis_path)
    
    # 3. Robot drawing
    robot = FaceDrawingRobot(xml_path=xml_path)
    success = robot.draw_face(face_data, rotation=rotation, scale=scale, position=position)
    if success:
        print("✓ Drawing Completed Successfully!")
    else:
        print("✗ Drawing Failed!")
    robot.close()



if __name__ == "__main__":
    import fire
    fire.Fire(main)