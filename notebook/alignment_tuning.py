import os
import json

import pickle
import cv2
import numpy as np
from datetime import datetime
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import trange

from ik_retarget import ik_retargeting, convert_landmarks_to_world, LMK2BODY_DEFAULT, JOINT_NAMES_FOR_IK
from mylmeval.vlm import MyVLMEval
from mujoco_parser import MuJoCoParserClass, solve_ik


class IKRefinementSystem:
    def __init__(self, 
                 vlm_model: str = "gpt-4o",
                 xml_path: str = '../asset/unitree_g1/scene_g1.xml',
                 max_iterations: int = 5,
                 frame_cache_dir: str = "/tmp/ik_refinement_cache"):
        
        home_cache_dir = os.path.expanduser('~/.cache')
        os.environ['HF_HOME'] = home_cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = home_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = home_cache_dir
        
        self.vlm = MyVLMEval(vlm_model, max_frames=32, target_size=224, download_dir=home_cache_dir)
        self.xml_path = xml_path
        self.max_iterations = max_iterations
        self.frame_cache_dir = frame_cache_dir
        os.makedirs(frame_cache_dir, exist_ok=True)
        
        self.ik_params = {
            'p_rate': 0.056444 * 15,
            'z_offset': 0.7,
            'radians': (90, 180, -90),
            'ik_steps': 50,
            'ik_err_th': 1e-3,
            'use_ankle_root': True,
            'smoothing_sigma': 2.0,
            'interpolation_factor': 2,
        }
        
    def interpolate_keypoints(self, keypoints_3d, factor=2):
        """Interpolate between keypoints to create smoother motion"""
        if len(keypoints_3d) < 2:
            return keypoints_3d
            
        valid_frames = []
        valid_indices = []
        for i, frame in enumerate(keypoints_3d):
            if frame is not None:
                valid_frames.append(frame)
                valid_indices.append(i)
        
        if len(valid_frames) < 2:
            return keypoints_3d
            
        valid_frames = np.array(valid_frames)  # (n_frames, n_landmarks, 3)
        
        old_timeline = np.array(valid_indices)
        new_timeline = np.linspace(0, len(keypoints_3d)-1, len(keypoints_3d) * factor)
        
        interpolated_frames = []
        for t in new_timeline:
            if t in valid_indices:
                frame_idx = valid_indices.index(int(t))
                interpolated_frames.append(valid_frames[frame_idx])
            else:
                interp_frame = np.zeros_like(valid_frames[0])
                for lm_idx in range(valid_frames.shape[1]):
                    for coord_idx in range(3):
                        values = valid_frames[:, lm_idx, coord_idx]
                        f = interp1d(old_timeline, values, kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
                        interp_frame[lm_idx, coord_idx] = f(t)
                interpolated_frames.append(interp_frame)
        
        return interpolated_frames
    
    def smooth_keypoints(self, keypoints_3d, sigma=2.0):
        if len(keypoints_3d) < 3:
            return keypoints_3d
            
        valid_frames = []
        valid_indices = []
        for i, frame in enumerate(keypoints_3d):
            if frame is not None:
                valid_frames.append(frame)
                valid_indices.append(i)
        
        if len(valid_frames) < 3:
            return keypoints_3d
            
        valid_frames = np.array(valid_frames)  # (n_frames, n_landmarks, 3)
        
        smoothed_frames = np.zeros_like(valid_frames)
        for lm_idx in range(valid_frames.shape[1]):
            for coord_idx in range(3):
                smoothed_frames[:, lm_idx, coord_idx] = gaussian_filter1d(
                    valid_frames[:, lm_idx, coord_idx], sigma=sigma)
        
        smoothed_keypoints = [None] * len(keypoints_3d)
        for i, frame_idx in enumerate(valid_indices):
            smoothed_keypoints[frame_idx] = smoothed_frames[i]
            
        return smoothed_keypoints
    
    def render_ik_to_gif(self, keypoints_pkl, output_gif_path, **ik_params):
        """Render IK retargeting result to GIF"""
        with open(keypoints_pkl, 'rb') as f:
            keypoints_3d = pickle.load(f)
        
        if ik_params.get('smoothing_sigma', 0) > 0:
            keypoints_3d = self.smooth_keypoints(keypoints_3d, ik_params['smoothing_sigma'])
        
        if ik_params.get('interpolation_factor', 1) > 1:
            keypoints_3d = self.interpolate_keypoints(keypoints_3d, ik_params['interpolation_factor'])
        
        # Setup MuJoCo environment  
        env = MuJoCoParserClass(name='IKRefine', rel_xml_path=self.xml_path, verbose=False)
        env.reset(step=True)
        
        # Initialize viewer for rendering
        try:
            env.init_viewer(transparent=True, backend='minimal')
        except Exception as e:
            print(f"Warning: Could not initialize viewer: {e}")
            print("Attempting to use minimal backend...")
            try:
                env.init_viewer(transparent=True, backend='native')
            except Exception as e2:
                print(f"Error: Could not initialize any viewer: {e2}")
                raise RuntimeError("Cannot initialize MuJoCo viewer for rendering")
        
        frames = []
        for i, landmarks in enumerate(keypoints_3d):
            if landmarks is None:
                continue
                
            # Convert landmarks to world coordinates
            pts_world = convert_landmarks_to_world(
                landmarks, 
                p_rate=ik_params.get('p_rate', 0.056444 * 15),
                z_offset=ik_params.get('z_offset', 0.7),
                radians=ik_params.get('radians', (90, 180, -90)),
                use_ankle_root=ik_params.get('use_ankle_root', True)
            )
            
            # Solve IK for each mapping
            for lm_idx, body_name in LMK2BODY_DEFAULT.items():
                p_trgt = pts_world[lm_idx]
                try:
                    q_init, ik_err_stack, ik_info = solve_ik(
                        env=env,
                        do='forward',
                        joint_names_for_ik=JOINT_NAMES_FOR_IK,
                        body_name_trgt=body_name,
                        p_trgt=p_trgt,
                        R_trgt=None,
                        max_ik_tick=ik_params.get('ik_steps', 50),
                        ik_err_th=ik_params.get('ik_err_th', 1e-3),
                        restore_state=False,
                        verbose=False,
                        verbose_warning=False,
                    )
                except Exception as e:
                    print(f'[IK] Warning: IK failed for {body_name}: {e}')
            
            # Capture frame
            env.step(ctrl=None)
            env.render()
            rgb_array = env.grab_image(rsz_rate=None)
            frames.append(rgb_array)
        
        # Clean up viewer
        if hasattr(env, 'viewer') and env.use_mujoco_viewer:
            env.close_viewer()
        
        # Save as GIF
        if frames:
            frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # First save as video
            temp_video = output_gif_path.replace('.gif', '_temp.mp4')
            out = cv2.VideoWriter(temp_video, fourcc, 10.0, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Convert to GIF using ffmpeg
            os.system(f'ffmpeg -i {temp_video} -vf "fps=10,scale=320:-1:flags=lanczos" -y {output_gif_path}')
            os.remove(temp_video)
        
        return output_gif_path
    
    def analyze_with_vlm(self, original_gif, generated_gif, iteration):
        """Use VLM to analyze differences and suggest improvements"""
        
        analysis_prompt = f"""
        Iteration {iteration}: Compare the original human motion (first GIF) with the robot's imitation (second GIF).

        Please analyze:
        1. **Motion Quality**: How well does the robot match the human motion?
        2. **Smoothness**: Are there any jerky or unnatural movements?
        3. **Timing**: Does the robot motion timing match the human?
        4. **Key Poses**: Are important poses (like peak positions) captured correctly?
        5. **Transitions**: Are the transitions between poses smooth and natural?

        Focus on specific improvements needed:
        - Should we increase/decrease smoothing?
        - Should we add more interpolated frames?
        - Should we adjust the root reference point (ankle vs hip)?
        - Should we modify the scale or offset parameters?
        - Are there specific joints that need attention?

        Please provide specific, actionable feedback for the next iteration.
        Rate the overall quality from 1-10 and suggest if we should continue iterating.
        """
        
        data = [
            {
                'inputs': [f"This is the robot's current attempt. " + analysis_prompt],
                'video_path': [original_gif, generated_gif],
                'id': f'iteration_{iteration}'
            },
        ]
        
        results = self.vlm.inference(
            prompt="{}",
            data=data,
            save_path=f"{self.frame_cache_dir}/vlm_analysis_{iteration}.json",
        )
        
        print(f"[VLM] Raw inference results (iteration {iteration}): {results}")
        
        return results[0] if results else "No analysis available"
    
    def parse_vlm_feedback(self, feedback_json: str):
        try:
            report = json.loads(feedback_json)
        except Exception as e:
            print("[VLM] Could not parse JSON:", e)
            return {}, 0  # no adj, neutral score
        
        adjustments = {}
        for body, info in report.items():
            if body in LMK2BODY_DEFAULT and "delta_deg" in info:
                joint_name = LMK2BODY_DEFAULT[body]
                delta_rad  = math.radians(info["delta_deg"])
                # Accumulate delta (add to current bias)
                adjustments.setdefault("joint_offsets", {})[joint_name] = \
                    adjustments.get("joint_offsets", {}).get(joint_name, 0.0) + delta_rad
        
        # Optional: pull a quality score if you also ask for it
        quality = report.get("_score", 5)
        return adjustments, quality
    
    def run(self, 
            original_gif_path: str,
            keypoints_pkl: str,
            output_dir: str = "refinement_results",
            max_iterations: int = 5,
            gif: str | list[int] = 'every'
            ):
        
        os.makedirs(output_dir, exist_ok=True)
        scores = []
        
        for iteration in trange(max_iterations):
            if gif == 'every' or int(iteration) in gif:
                generated_gif = os.path.join(output_dir, f"iteration_{iteration + 1}.gif")
                self.render_ik_to_gif(keypoints_pkl, generated_gif, **self.ik_params)
            
            feedback = self.analyze_with_vlm(original_gif_path, generated_gif, iteration + 1)
            adjustments, quality_score = self.parse_vlm_feedback(feedback)
            
            results = {
                'iteration': iteration + 1,
                'parameters': deepcopy(self.ik_params),
                'feedback': feedback,
                'quality_score': quality_score,
                'adjustments': adjustments,
                'generated_gif': generated_gif
            }
            
            with open(os.path.join(output_dir, "results.json"), 'a') as f:
                json.dump(results, f, indent=2)
            
            scores.append(quality_score)
            
            if adjustments and iteration < self.max_iterations - 1:
                self.ik_params.update(adjustments)

            
        return scores


def main(
    xml_path: str = '../asset/unitree_g1/scene_g1.xml',
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    original_gif: str = "temp_mocap/88848d2067d622de8e4f314e28dc431a.gif",
    keypoints_pkl: str = "temp_mocap/88848d2067d622de8e4f314e28dc431a.pkl",
    max_iterations: int = 5,
    output_dir: str = "refinement_results"
):
    now_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    refiner = IKRefinementSystem(
        vlm_model=vlm_model,
        xml_path=xml_path,
    )
    
    scores = refiner.run(
        original_gif_path=original_gif,
        keypoints_pkl=keypoints_pkl,
        max_iterations=max_iterations,
        output_dir=os.path.join(output_dir, now_date),
        gif='every',
    )

    print(f"Max Iteration: {len(scores)} / Best score: {max(scores)} / Mean score: {np.mean(scores)}")
    print(f"Final result: {os.path.join(output_dir, now_date)}/iteration_5.gif")
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)