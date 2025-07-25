import os
import cv2
import ssl
import random
import shutil
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Any
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

# --- ABLATION STUDY VERSION ---
# This script is modified for an ablation study.
# - Visual RAG components have been REMOVED.
# - Perturbation is applied to a RANDOM area, not a saliency-guided mask.
# - Results are saved to a separate 'results_ablation_study' directory.

# Classification classes for car view detection
CLASSES = [  "back" ,  "front-left" ,  "back-right" ,  "back-left" ,  "front-right" ,  "front"  ]

# --- Setup and Configuration ---

ssl._create_default_https_context = ssl._create_unverified_context

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# --- Core Video Processing Class (Ablation Version) ---

class AblationStudyVideoAnalyzer:
    """
    Ablation study version of the video analyzer.
    Applies diffusion perturbation to RANDOM areas instead of using RAG-guided masks.
    """
    
    def __init__(self):
        # Cache for repeated computations
        self.detection_cache = {}
        
        self.yolo = YOLO('yolov8n.pt')
        # Optimize YOLO for speed
        self.yolo.overrides['verbose'] = False
        self.yolo.overrides['device'] = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print("✓ YOLO model loaded successfully")
        
        # Load TFLite classifier for car view classification
        try:
            self.tflite_model = tf.lite.Interpreter(model_path='model.tflite')
            self.tflite_model.allocate_tensors()
            self.input_details = self.tflite_model.get_input_details()
            self.output_details = self.tflite_model.get_output_details()
            print("✓ TFLite classifier loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load TFLite model: {e}")
            print("Will use default 'front' category")
            self.tflite_model = None
        
        # Initialize Kandinsky 2.2 Diffusion Model
        self.initialize_kandinsky()
        print("✓ Ablation study analyzer initialized (No RAG)")

    def initialize_kandinsky(self):
        """Initialize Kandinsky 2.2 for inpainting."""
        from diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
        self.prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ).to("mps")
        self.diffusion_pipe = KandinskyV22InpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        ).to("mps")
        print("✓ Kandinsky 2.2 model loaded on Apple M2 GPU")

    def _get_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extracts and validates the region of interest (ROI)."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def get_all_detections(self, frame: np.ndarray) -> List[Dict]:
        """Gets all YOLO detections (box, score, class) from a frame with caching."""
        frame_hash = hash(frame.tobytes())
        if frame_hash in self.detection_cache:
            return self.detection_cache[frame_hash]
        detections = []
        results = self.yolo(frame, verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                detections.append({
                    'box': tuple(map(int, box.xyxy[0].cpu().numpy())),
                    'score': box.conf[0].item(),
                    'class': int(box.cls[0].item())
                })
        self.detection_cache[frame_hash] = detections
        return detections

    def get_confidence(self, frame: np.ndarray) -> float:
        """Gets the highest YOLO confidence for a frame."""
        detections = self.get_all_detections(frame)
        if not detections: return 0.0
        return max(d['score'] for d in detections)

    def detect_bbox(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Detects the bounding box of the highest-confidence object."""
        detections = self.get_all_detections(frame)
        if not detections: raise ValueError("No detections found in frame")
        return max(detections, key=lambda d: d['score'])['box']

    def classify_car(self, image):
        """Classify a car image using TFLite model."""
        if self.tflite_model is None: return 'front', 0.5
        try:
            if isinstance(image, np.ndarray): image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = image.convert('RGB').resize((224, 224))
            img_array = np.array(image)[None].astype('float32')
            self.tflite_model.resize_tensor_input(self.input_details[0]['index'], (1, 224, 224, 3))
            self.tflite_model.allocate_tensors()
            self.tflite_model.set_tensor(self.input_details[0]['index'], img_array)
            self.tflite_model.invoke()
            scores = self.tflite_model.get_tensor(self.output_details[0]['index'])
            return CLASSES[scores.argmax()], float(scores.max())
        except Exception as e:
            print(f"Error in classification: {e}")
            return 'front', 0.5

    def _generate_random_mask(self, roi_shape: Tuple[int, int]) -> Image.Image:
        """
        ABLATION: Generates a random rectangular mask instead of using a saliency map.
        The mask covers a smaller area of the ROI.
        """
        h, w = roi_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # --- CHANGE 1: REDUCE MASK SIZE ---
        # The mask now covers between 10% and 25% of the ROI area,
        # making the perturbation less "major".
        min_area = 0.10  # Was 0.2
        max_area = 0.25  # Was 0.5
        rect_area = random.uniform(min_area, max_area) * (w * h)
        
        # Adjust width constraints to allow for smaller, more focused rectangles
        rect_w = int(random.uniform(0.2, 0.5) * w) # Was 0.3 to 0.8
        rect_h = int(rect_area / rect_w)
        
        rect_w = min(w, rect_w)
        rect_h = min(h, rect_h)

        # Define random top-left corner for the rectangle
        rect_x = random.randint(0, w - rect_w)
        rect_y = random.randint(0, h - rect_h)
        
        # Draw the white rectangle on the black mask
        mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = 255
        
        return Image.fromarray(mask)


    def random_diffusion_perturbation(self, frame: np.ndarray, bbox: Tuple,
                                      strength: float, category: str) -> np.ndarray:
        """
        ABLATION: Applies diffusion perturbation using a RANDOM mask.
        """
        roi, valid_bbox = self._get_roi(frame, bbox)
        if roi is None:
            return frame

        roi_height, roi_width = roi.shape[:2]
        
        # --- KEY ABLATION CHANGE ---
        # Generate a random mask instead of loading a RAG-based one.
        mask_pil = self._generate_random_mask((roi_height, roi_width))
        # --- END OF CHANGE ---
        
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
        # Use the same prompt structure as the original pipeline
        category_prompts = {
            'front': "detailed automotive front view, sharp headlights, defined grille, clear bumper and tyre details",
            'back-left': "detailed rear-left automotive view, bright taillights, clear trunk lines, defined rear bumper, tyres and lights",
            'back-right': "detailed rear-right automotive view, bright taillights, clear trunk lines, defined rear bumper and tyres",
            'front-left': "detailed front-left automotive view, sharp headlight details, visible wheel, clear hood lines and tyres",
            'front-right': "detailed front-right automotive view, sharp headlight details, visible wheel, clear hood lines and tyres",
            'back': "detailed automotive rear profile, clear bumber lines, prominent wheel, defined window structure and vehicle rear lights"
        }
        base_prompt = "high quality automotive photo, sharp details, realistic lighting, photorealistic"
        prompt = f"{base_prompt}, {category_prompts.get(category, 'detailed automotive view')}"
        negative_prompt = "blurry, low quality, distorted, unrealistic, cartoon, painting, sketch, low resolution, artifacts"

        # Run diffusion model
        prior_out = self.prior_pipe(prompt, negative_prompt=negative_prompt, guidance_scale=1.0, num_images_per_prompt=1, num_inference_steps=25, generator=torch.Generator(device="mps").manual_seed(42))
        image_embeds = prior_out.image_embeds
        negative_image_embeds = prior_out.negative_image_embeds
        
        inpainted_roi_pil = self.diffusion_pipe(prompt=prompt, image=roi_pil, mask_image=mask_pil, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, num_inference_steps=30, guidance_scale=4.0, strength=strength, generator=torch.Generator(device="mps").manual_seed(42)).images[0]
        
        inpainted_roi = cv2.cvtColor(np.array(inpainted_roi_pil), cv2.COLOR_RGB2BGR)
        if inpainted_roi.shape[:2] != (roi_height, roi_width):
            inpainted_roi = cv2.resize(inpainted_roi, (roi_width, roi_height))

        # Blend the original and inpainted ROI based on the random mask
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        blend_ratio_3ch = np.stack([mask_np] * 3, axis=2)
        final_roi = (roi.astype(np.float32) * (1.0 - blend_ratio_3ch) + inpainted_roi.astype(np.float32) * blend_ratio_3ch).astype(np.uint8)
        
        # Apply final adjustments
        final_roi = cv2.convertScaleAbs(final_roi, alpha=1.05, beta=5)
        
        x1, y1, x2, y2 = valid_bbox
        result = frame.copy()
        result[y1:y2, x1:x2] = cv2.resize(final_roi, (x2 - x1, y2 - y1))
        return result

    # --- CHANGE 2: SIMPLIFY TO A SINGLE ATTEMPT ---
    def find_best_random_perturbation(
        self, frame: np.ndarray, bbox: Tuple, category: str,
        strength: float
    ) -> Tuple[float, np.ndarray, float, str]:
        """
        ABLATION (MODIFIED): Tries a SINGLE random perturbation.
        The loop has been removed for efficiency.
        """
        original_conf = self.get_confidence(frame)
        strategy_name = "single_random_perturb"
        
        print("  Attempting perturbation with a single Random Mask...")

        try:
            perturbed_frame = self.random_diffusion_perturbation(
                frame, bbox, strength, category
            )
            new_conf = self.get_confidence(perturbed_frame)
            improvement = new_conf - original_conf
            
            print(f"    Single Mask Result: Conf={new_conf:.4f}, Improvement={improvement:+.4f}")

            # Return the result of this single attempt
            return improvement, perturbed_frame, strength, strategy_name
            
        except Exception as e:
            print(f"    Error during random perturbation: {e}")
            # If an error occurs, return no improvement and the original frame
            return 0.0, frame, strength, "error"
    
    def determine_category_from_frame(self, frame: np.ndarray, bbox: Tuple = None) -> str:
        """Determines the car view category using TFLite classifier."""
        if self.tflite_model is None: return 'front'
        try:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: return 'front'
            else:
                roi = frame
            predicted_class, _ = self.classify_car(roi)
            return predicted_class
        except Exception as e:
            print(f"Error in category determination: {e}")
            return 'front'

    def process_video(self, video_path: str, conf_upper: float, conf_lower: float, output_dir: str,
                     strength_lower: float, strength_upper: float, strength_jump: float):
        """Main video processing loop with RANDOM diffusion-based perturbation."""
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_dir = os.path.join(output_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_enhanced = cv2.VideoWriter(os.path.join(output_dir, 'ablation_enhanced_video.mp4'), 
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        results = []
        
        with tqdm(total=total_frames, desc="Processing video (Ablation Study)", unit="frame") as pbar:
            frame_idx = 0
            improvements_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                try:
                    original_conf = self.get_confidence(frame)
                    
                    if conf_lower <= original_conf <= conf_upper:
                        try:
                            bbox = self.detect_bbox(frame)
                            category = self.determine_category_from_frame(frame, bbox)
                            
                            # ABLATION: Use the random perturbation finder
                            improvement, best_frame, best_strength, strategy = self.find_best_random_perturbation(
                                frame, bbox, category, 0.9 # Using a fixed high strength for consistency
                            )
                            
                            final_conf = self.get_confidence(best_frame)
                            
                            if improvement > 0.001:
                                out_enhanced.write(best_frame)
                                improvements_count += 1
                                self._create_side_by_side_comparison(
                                    frame, best_frame, original_conf, final_conf, frame_idx, output_dir
                                )
                                pbar.set_postfix({
                                    'Improved': improvements_count,
                                    'Last_gain': f'{improvement:.3f}',
                                    'Strategy': strategy,
                                })
                            else:
                                out_enhanced.write(frame)
                            
                        except Exception as e:
                            print(f"Error processing frame {frame_idx}: {e}")
                            out_enhanced.write(frame)
                            improvement = 0.0; best_strength = 0.0; strategy = "error"; final_conf = original_conf
                    else:
                        out_enhanced.write(frame)
                        improvement = 0.0; best_strength = 0.0; strategy = "no_enhancement"; final_conf = original_conf
                    
                    results.append({
                        'frame_idx': frame_idx,
                        'original_conf': original_conf,
                        'final_conf': final_conf,
                        'improvement': improvement,
                        'strength': best_strength,
                        'strategy': strategy,
                        'needs_enhancement': conf_lower <= original_conf <= conf_upper
                    })
                    
                except Exception as e:
                    print(f"Critical error processing frame {frame_idx}: {e}")
                    out_enhanced.write(frame)
                    
                frame_idx += 1
                pbar.update(1)
                
        cap.release()
        out_enhanced.release()
        
        print(f"\n✓ Ablation processing complete! {improvements_count}/{total_frames} frames improved.")
        print(f"✓ Side-by-side comparisons saved in '{output_dir}/comparisons/' directory.")
        self.generate_analysis_plots(results, conf_upper, conf_lower, output_dir)
        return results

    def _create_side_by_side_comparison(self, original_frame: np.ndarray, perturbed_frame: np.ndarray, 
                                      original_conf: float, perturbed_conf: float, 
                                      frame_idx: int, output_dir: str) -> None:
        """Creates and saves a side-by-side comparison image."""
        h, w = original_frame.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original_frame
        comparison[:, w:] = perturbed_frame
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        cv2.putText(comparison, f"Original: {original_conf:.3f}", (10, 30), font, font_scale, (0, 255, 0), thickness)
        # ABLATION: Changed text to reflect random perturbation
        cv2.putText(comparison, f"Ablation Enhanced: {perturbed_conf:.3f}", (w + 10, 30), font, font_scale, (0, 255, 0), thickness)
        
        improvement = perturbed_conf - original_conf
        improvement_color = (0, 255, 0) if improvement > 0 else (0, 0, 255)
        cv2.putText(comparison, f"Improvement: {improvement:+.3f}", (w // 2 - 50, h - 20), font, font_scale, improvement_color, thickness)
        
        cv2.putText(comparison, "Random Perturbation", (w + 10, h - 50), font, font_scale, (255, 255, 0), thickness)
        
        comparison_dir = os.path.join(output_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        comparison_path = os.path.join(comparison_dir, f'frame_{frame_idx:06d}_ablation_comparison.jpg')
        cv2.imwrite(comparison_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def generate_analysis_plots(self, results: List[Dict], conf_upper: float, conf_lower: float, output_dir: str):
        """Generates analysis plots for the ABLATION study."""
        frame_indices = [r['frame_idx'] for r in results]
        original_confs = [r['original_conf'] for r in results]
        final_confs = [r['final_conf'] for r in results]
        
        improved_results = [r for r in results if r['improvement'] > 0.001]
        
        # ABLATION: Simplified plots, removed RAG-specific analysis
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=120)
        fig.suptitle('Ablation Study: Random Perturbation Video Enhancement', fontsize=20, fontweight='bold')
        ax = axes.flatten()
        
        # Plot 0: Confidence over time
        ax[0].plot(frame_indices, original_confs, 'r-', label='Original Conf', alpha=0.6, linewidth=1)
        ax[0].plot(frame_indices, final_confs, 'b-', label='Ablation Enhanced Conf', alpha=0.7, linewidth=1.5)
        ax[0].axhline(conf_upper, color='orange', ls='--', label=f'Upper={conf_upper}')
        ax[0].axhline(conf_lower, color='purple', ls='--', label=f'Lower={conf_lower}')
        ax[0].fill_between(frame_indices, conf_lower, conf_upper, color='yellow', alpha=0.15, label='Target Range')
        ax[0].set_title('Confidence Over Time (Random Perturbation)')
        ax[0].set_xlabel('Frame'); ax[0].set_ylabel('Confidence'); ax[0].legend(); ax[0].grid(True, alpha=0.3)

        # Plot 1: Improvement distribution
        ax[1].hist([r['improvement'] for r in improved_results], bins=25, color='lightblue', edgecolor='black')
        ax[1].set_title('Distribution of Positive Improvements (Random)')
        ax[1].set_xlabel('Confidence Gain'); ax[1].set_ylabel('Frequency'); ax[1].grid(True, alpha=0.3)
        
        # Plot 2: Removed RAG plots, added empty plot for future use or can be removed.
        ax[2].axis('off') # Turn off unused plot

        # Plot 3: Summary statistics
        total_improved = len(improved_results)
        total_in_range = sum(1 for r in results if r['needs_enhancement'])
        
        stats_text = f"""
        ABLATION STUDY SUMMARY
        (Random Perturbation)
        {'-'*30}
        Total Frames: {len(results):,}
        Frames in Target Range: {total_in_range:,}
        Frames Improved: {total_improved:,}
        Success Rate: {total_improved/total_in_range*100 if total_in_range > 0 else 0:.1f}%
        
        Avg. Improvement: {np.mean([r['improvement'] for r in improved_results] or [0]):.4f}
        """
        ax[3].text(0.05, 0.95, stats_text.strip(), transform=ax[3].transAxes, fontsize=12, 
                  verticalalignment='top', fontfamily='monospace', 
                  bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
        ax[3].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, 'ablation_analysis_summary.png'))
        plt.show()

# --- Main Execution ---

def main_ablation_analysis(
    video_path: str, conf_upper: float, conf_lower: float,
    strength_lower: float, strength_upper: float, strength_jump: float
):
    """Main function to run the ablation study video analysis."""
    set_random_seeds(42)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # ABLATION: Use the new analyzer class and output directory
    analyzer = AblationStudyVideoAnalyzer()
    output_dir = "results_ablation_study"
    
    analyzer.process_video(
        video_path, conf_upper, conf_lower, output_dir,
        strength_lower, strength_upper, strength_jump
    )
    print(f"\n✅ Ablation study analysis complete. Results saved in '{output_dir}'.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    VIDEO_FILE = "test_video_pertubation.mp4"
    
    CONF_UPPER_THRESHOLD = 0.75
    CONF_LOWER_THRESHOLD = 0.65
    
    # Strength settings are less critical here as we aren't iterating through them,
    # but we pass them along for consistency with the function signature.
    STRENGTH_LOWER_BOUND = 0.5
    STRENGTH_UPPER_BOUND = 0.9
    STRENGTH_JUMP_FACTOR = 0.3
    
    print("🚀 Running ABLATION STUDY: Single, smaller random perturbation-based video analysis.")
    print("   (RAG and guided masks are disabled)")
    print(f"Video: {VIDEO_FILE}")
    print(f"Confidence Target: {CONF_LOWER_THRESHOLD} - {CONF_UPPER_THRESHOLD}")
    
    main_ablation_analysis(
        video_path=VIDEO_FILE,
        conf_upper=CONF_UPPER_THRESHOLD,
        conf_lower=CONF_LOWER_THRESHOLD,
        strength_lower=STRENGTH_LOWER_BOUND,
        strength_upper=STRENGTH_UPPER_BOUND,
        strength_jump=STRENGTH_JUMP_FACTOR
    )