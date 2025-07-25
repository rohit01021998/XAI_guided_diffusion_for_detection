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
# from diffusers import StableDiffusionInpaintPipeline
import tensorflow as tf

# Import Visual RAG for enhanced saliency map retrieval
import sys
sys.path.append('.')
# from CropedSalencyMapCreator import yolov8_cropped_heatmap, get_params

# Import Visual RAG functionality from our dedicated script
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("visual_rag_module", "2.visual_RAG_and_combined_mask.py")
    visual_rag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visual_rag_module)
    
    # Import the components we need
    VisualRAG = visual_rag_module.VisualRAG
    get_image_embedding = visual_rag_module.get_image_embedding
    combine_saliency_maps = visual_rag_module.combine_saliency_maps
    retrieve_and_combine_saliency = visual_rag_module.retrieve_and_combine_saliency
    retrieve_saliency_and_original = visual_rag_module.retrieve_saliency_and_original
    cosine_similarity = visual_rag_module.cosine_similarity
    extract_red_importance = visual_rag_module.extract_red_importance
    
    print("✓ Visual RAG components imported successfully")
except Exception as e:
    print(f"Error importing Visual RAG components: {e}")
    print("Make sure 2.visual_RAG_and_combined_mask.py is in the same directory")
    sys.exit(1)

# Import Visual RAG functionality
import pickle
from pathlib import Path

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

# --- Core Video Processing Class ---

class RAGDiffusionVideoAnalyzer:
    """Enhanced video analyzer using RAG-based combined masks with diffusion perturbation."""
    
    def __init__(self):
        # Cache for repeated computations
        self.detection_cache = {}
        self.rag_mask_cache = {}
        
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
        

        # --- Diffusion Model Initialization: Kandinsky 2.2 (Option 3) ---
        self.initialize_kandinsky()
    # Option 3: Use Kandinsky 2.2 (High Quality + Fast)
    def initialize_kandinsky(self):
        """Initialize Kandinsky 2.2 for high quality and speed"""
        from diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline

        # Load prior and inpainting pipelines
        self.prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch.float16
        ).to("mps")

        self.diffusion_pipe = KandinskyV22InpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint",
            torch_dtype=torch.float16
        ).to("mps")



        print("✓ Kandinsky 2.2 model loaded on Apple M2 GPU")
        
        # Initialize Visual RAG system with compatibility checking
        self.visual_rag = VisualRAG()
        database_loaded = self.visual_rag.load_database()
        
        if database_loaded:
            # Check compatibility with current embedding method
            try:
                # Find a test image for compatibility check
                test_image_path = None
                for category in self.visual_rag.categories:
                    category_path = os.path.join(self.visual_rag.base_path, category, "cropped_images")
                    if os.path.exists(category_path):
                        demo_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
                        if demo_files:
                            test_image_path = os.path.join(category_path, demo_files[0])
                            break
                
                # Check database compatibility using imported function
                if test_image_path:
                    # Generate test embedding to check compatibility
                    test_embedding = get_image_embedding(test_image_path)
                    if test_embedding is not None:
                        # Check if stored embeddings have different shapes
                        needs_rebuild = False
                        for category, db in self.visual_rag.vector_db.items():
                            if db:  # If category has embeddings
                                stored_embedding = db[0][0]  # First embedding
                                if stored_embedding.shape != test_embedding.shape:
                                    print(f"⚠️  Database incompatibility detected:")
                                    print(f"   Stored embeddings: {stored_embedding.shape}")
                                    print(f"   Current method: {test_embedding.shape}")
                                    print("🔄 Rebuilding database with compatible embeddings...")
                                    needs_rebuild = True
                                    break
                        
                        if needs_rebuild:
                            self.visual_rag.build_database()
                            self.visual_rag.save_database()
                    else:
                        print("Could not generate test embedding, rebuilding database...")
                        self.visual_rag.build_database()
                        self.visual_rag.save_database()
                else:
                    print("No test images found, rebuilding database...")
                    self.visual_rag.build_database()
                    self.visual_rag.save_database()
                    
            except Exception as e:
                print(f"Error checking database compatibility: {e}")
                print("Rebuilding Visual RAG database...")
                self.visual_rag.build_database()
                self.visual_rag.save_database()
        else:
            print("📊 Building new Visual RAG database...")
            self.visual_rag.build_database()
            self.visual_rag.save_database()
            
        print("✓ Visual RAG system initialized")

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
        # Create cache key from frame hash
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
        
        # Cache the result
        self.detection_cache[frame_hash] = detections
        return detections

    def get_confidence(self, frame: np.ndarray) -> float:
        """Gets the highest YOLO confidence for a frame."""
        detections = self.get_all_detections(frame)
        if not detections:
            return 0.0
        return max(d['score'] for d in detections)

    def detect_bbox(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Detects the bounding box of the highest-confidence object."""
        detections = self.get_all_detections(frame)
        if not detections:
            raise ValueError("No detections found in frame")
        # Return the box of the detection with the highest score
        best_detection = max(detections, key=lambda d: d['score'])
        return best_detection['box']

    def get_rag_combined_mask_and_original(self, frame: np.ndarray, temp_dir: str, category: str = 'front', top_k: int = 3) -> list:
        """
        Gets a list of combined saliency masks and top original images using Visual RAG.
        
        Returns a list of tuples: [(mask_path_1, original_path_1), (mask_path_2, original_path_2), ...].
        """
        frame_hash = hash(frame.tobytes())
        cache_key = f"{frame_hash}_{category}_top_{top_k}"
        if cache_key in self.rag_mask_cache:
            return self.rag_mask_cache[cache_key]

        frame_file = os.path.join(temp_dir, 'temp_frame_rag.jpg')
        cv2.imwrite(frame_file, frame)

        try:
            # Use retrieve_saliency_and_original with return_multiple=True to get top_k results
            results = retrieve_saliency_and_original(
                frame_file, category, self.visual_rag, top_k=top_k, debug=False, return_multiple=True
            )
            
            if not results:
                raise RuntimeError(f"Visual RAG returned no results for category {category}")

            # The function now returns a list of (mask_path, original_path) tuples
            self.rag_mask_cache[cache_key] = results
            
        except Exception as e:
            raise RuntimeError(f"Error getting Top-{top_k} RAG masks and originals: {e}")
        finally:
            if os.path.exists(frame_file):
                os.remove(frame_file)
                
        return results

    def classify_car(self, image):
        """Classify a car image using TFLite model (from simple_car_detection.py)."""
        if self.tflite_model is None:
            return 'front', 0.5  # Default fallback
            
        try:
            # Convert and resize image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            image = image.convert('RGB').resize((224, 224))
            img_array = np.array(image)[None].astype('float32')
            
            # Update tensor size and run inference
            self.tflite_model.resize_tensor_input(self.input_details[0]['index'], (1, 224, 224, 3))
            self.tflite_model.allocate_tensors()
            self.tflite_model.set_tensor(self.input_details[0]['index'], img_array)
            self.tflite_model.invoke()
            
            # Get prediction
            scores = self.tflite_model.get_tensor(self.output_details[0]['index'])
            predicted_class = CLASSES[scores.argmax()]
            confidence = float(scores.max())
            
            return predicted_class, confidence
        except Exception as e:
            print(f"Error in classification: {e}")
            return 'front', 0.5  # Default fallback

    def _apply_perturbation_to_frame(self, frame: np.ndarray, perturbed_roi: np.ndarray, bbox: Tuple) -> np.ndarray:
        """Applies a perturbed ROI back onto the full frame."""
        x1, y1, x2, y2 = bbox
        result = frame.copy()
        result[y1:y2, x1:x2] = cv2.resize(perturbed_roi, (x2 - x1, y2 - y1))
        return result

    def _calculate_iou(self, boxA: Tuple, boxB: Tuple) -> float:
        """Calculates Intersection over Union (IoU) for two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / (float(boxAArea + boxBArea - interArea) + 1e-6)
        return iou

    def _calculate_pr_map(self, all_gt_dets: List[List[Dict]], all_pred_dets: List[List[Dict]], iou_threshold: float = 0.5):
        """Calculates precision, recall, and mean Average Precision (mAP)."""
        detections = []
        total_gt_boxes = 0

        for frame_idx, gt_dets in enumerate(all_gt_dets):
            pred_dets = all_pred_dets[frame_idx]
            total_gt_boxes += len(gt_dets)
            
            gt_matched = [False] * len(gt_dets)

            for pred in pred_dets:
                best_iou = 0
                best_gt_idx = -1
                for i, gt in enumerate(gt_dets):
                    if not gt_matched[i]:
                        iou = self._calculate_iou(pred['box'], gt['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                
                is_tp = best_iou >= iou_threshold
                if is_tp and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                
                detections.append({
                    'score': pred['score'],
                    'is_tp': is_tp
                })

        detections.sort(key=lambda x: x['score'], reverse=True)
        tps = np.cumsum([d['is_tp'] for d in detections])
        fps = np.cumsum([not d['is_tp'] for d in detections])
        
        recalls = tps / (total_gt_boxes + 1e-6)
        precisions = tps / (tps + fps + 1e-6)

        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        recall_levels = np.linspace(0, 1, 11)
        ap = 0.0
        for r in recall_levels:
            try:
                p = precisions[np.where(recalls >= r)[0][0]]
                ap += p
            except IndexError:
                ap += 0.0
            
        return precisions, recalls, ap / len(recall_levels)

    def rag_guided_diffusion_perturbation(self, frame: np.ndarray, bbox: Tuple, temp_dir: str, 
                                        strength: float, category: str, 
                                        rag_mask_path: str, top_original_path: str) -> np.ndarray:
        """
        Applies RAG-guided diffusion perturbation using a *specific* RAG mask and original image.
        This version is refactored to accept the mask and original image paths directly.
        """
        roi, valid_bbox = self._get_roi(frame, bbox)
        if roi is None:
            return frame

        # Load the provided RAG mask and top original image
        rag_mask = cv2.imread(rag_mask_path)
        top_original = cv2.imread(top_original_path)
        
        if rag_mask is None:
            raise ValueError(f"Could not load RAG mask from path: {rag_mask_path}")
        if top_original is None:
            raise ValueError(f"Could not load top original image from path: {top_original_path}")

        # --- The rest of the function remains the same as the original ---
        # (Resizing, mask processing, diffusion, etc.)
        roi_height, roi_width = roi.shape[:2]
        
        if len(rag_mask.shape) == 3:
            rag_mask = cv2.cvtColor(rag_mask, cv2.COLOR_BGR2GRAY)
        rag_mask = rag_mask.astype(np.uint8)

        rag_mask_resized = cv2.resize(rag_mask, (roi_width, roi_height))
        top_original_resized = cv2.resize(top_original, (roi_width, roi_height))

        importance = cv2.normalize(rag_mask_resized, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        importance = cv2.bilateralFilter(importance, 7, 50, 50)
        importance = np.power(importance, 1.5)

        mask_levels = np.full_like(importance, 100, dtype=np.float32)
        mask_levels[importance >= 0.7] = 255
        mask_levels[(importance >= 0.5) & (importance < 0.7)] = 180
        mask_levels[(importance >= 0.3) & (importance < 0.5)] = 120
        mask_levels[(importance >= 0.1) & (importance < 0.3)] = 90
        mask_levels = np.clip(mask_levels, 100, 255).astype(np.uint8)
        scaled_range = (mask_levels - 100) * strength
        mask_levels = np.clip(scaled_range + 100, 100, 255).astype(np.uint8)

        mask_pil = Image.fromarray(mask_levels)
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
        category_prompts = {
            'front': "detailed automotive front view, sharp headlights, defined grille, clear bumper and tyre details",
            'back-left': "detailed rear-left automotive view, bright taillights, clear trunk lines, defined rear bumper, tyres and lights",
            'back-right': "detailed rear-right automotive view, bright taillights, clear trunk lines, defined rear bumper and tyres",
            'front-left': "detailed front-left automotive view, sharp headlight details, visible wheel, clear hood lines and tyres",
            'front-right': "detailed front-right automotive view, sharp headlight details, visible wheel, clear hood lines and tyres",
            'back': "detailed automotive rear profile, clear bumber lines, prominent wheel, defined window structure and vehicle rear lights"
        }
        base_prompt = "high quality automotive photo, sharp details, realistic lighting, photorealistic"
        category_specific = category_prompts.get(category, "detailed automotive view")
        prompt = f"{base_prompt}, {category_specific}"
        negative_prompt = "blurry, low quality, distorted, unrealistic, cartoon, painting, sketch, low resolution, artifacts"

        prior_out = self.prior_pipe(prompt, negative_prompt=negative_prompt, guidance_scale=1.0, num_images_per_prompt=1, num_inference_steps=25, generator=torch.Generator(device="mps").manual_seed(42))
        image_embeds = prior_out.image_embeds
        negative_image_embeds = prior_out.negative_image_embeds
        
        inpainted_roi_pil = self.diffusion_pipe(prompt=prompt, image=roi_pil, mask_image=mask_pil, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, num_inference_steps=30, guidance_scale=4.0, strength=0.8, generator=torch.Generator(device="mps").manual_seed(42)).images[0]
        
        inpainted_roi = cv2.cvtColor(np.array(inpainted_roi_pil), cv2.COLOR_RGB2BGR)
        if inpainted_roi.shape[:2] != (roi_height, roi_width):
            inpainted_roi = cv2.resize(inpainted_roi, (roi_width, roi_height))

        blend_ratio = cv2.normalize(mask_levels, None, 0.2, 0.9, cv2.NORM_MINMAX, cv2.CV_32F)
        blend_ratio_3ch = np.stack([blend_ratio] * 3, axis=2)
        final_roi = (roi.astype(np.float32) * (1.0 - blend_ratio_3ch) + inpainted_roi.astype(np.float32) * blend_ratio_3ch).astype(np.uint8)
        final_roi = cv2.convertScaleAbs(final_roi, alpha=1.05, beta=5)
        
        return self._apply_perturbation_to_frame(frame, final_roi, valid_bbox)


    def find_best_rag_perturbation(
        self, frame: np.ndarray, bbox: Tuple, temp_dir: str, category: str,
        strength: float, improvement_threshold: float = 0.05
    ) -> Tuple[float, np.ndarray, float, str]:
        """
        Sequentially tries the top 3 RAG references to find the best perturbation.
        - If any attempt improves confidence by `improvement_threshold`, it returns immediately.
        - Otherwise, it returns the best result from all three attempts.
        """
        original_conf = self.get_confidence(frame)
        
        best_improvement = -1.0  # Initialize with a value lower than any possible improvement
        best_frame = frame
        best_strategy = "no_improvement"

        try:
            # Get the top 3 reference images and their saliency masks
            top_references = self.get_rag_combined_mask_and_original(frame, temp_dir, category, top_k=3)
        except Exception as e:
            print(f"Could not retrieve RAG references: {e}")
            return 0.0, frame, 0.0, "error_retrieving_rag_data"

        # Loop through the top 3 references
        for i, (mask_path, original_path) in enumerate(top_references):
            strategy_name = f"rag_diffusion_ref_{i+1}"
            print(f"  Attempting perturbation with Reference #{i+1}...")

            try:
                # Apply perturbation using the current reference image and mask
                perturbed_frame = self.rag_guided_diffusion_perturbation(
                    frame, bbox, temp_dir, strength, category, mask_path, original_path
                )
                
                new_conf = self.get_confidence(perturbed_frame)
                improvement = new_conf - original_conf
                
                print(f"    Reference #{i+1} Result: Conf={new_conf:.4f}, Improvement={improvement:+.4f}")

                # If this is the best improvement so far, store it
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_frame = perturbed_frame
                    best_strategy = strategy_name
                
                # If improvement meets the threshold, we are done with this frame
                if improvement >= improvement_threshold:
                    print(f"  ✅ Improvement threshold met. Finalizing with Reference #{i+1}.")
                    return best_improvement, best_frame, strength, best_strategy
                    
            except Exception as e:
                print(f"    Error during perturbation with Reference #{i+1}: {e}")
                continue # Try the next reference image

        # If the loop finishes, it means no attempt met the threshold.
        # We return the best result found among the attempts.
        print(f"  Threshold not met. Selecting best result from all attempts (Improvement: {best_improvement:+.4f}).")
        return best_improvement, best_frame, strength, best_strategy

    def determine_category_from_frame(self, frame: np.ndarray, bbox: Tuple = None) -> str:
        """Determines the car view category using TFLite classifier (like simple_car_detection.py)."""
        if self.tflite_model is None:
            return 'front'  # Default fallback
            
        try:
            # If bbox is provided, crop the car region; otherwise use the full frame
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    return 'front'
            else:
                roi = frame
            
            # Classify the car region
            predicted_class, confidence = self.classify_car(roi)
            return predicted_class
            
        except Exception as e:
            print(f"Error in category determination: {e}")
            return 'front'  # Default fallback

    def process_video(self, video_path: str, conf_upper: float, conf_lower: float, output_dir: str,
                     strength_lower: float, strength_upper: float, strength_jump: float):
        """Main video processing loop with RAG-guided diffusion-based perturbation."""
        os.makedirs(output_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=output_dir)
        
        comparison_dir = os.path.join(output_dir, 'comparisons')
        rag_masks_dir = os.path.join(output_dir, 'rag_masks')
        os.makedirs(comparison_dir, exist_ok=True)
        os.makedirs(rag_masks_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_enhanced = cv2.VideoWriter(os.path.join(output_dir, 'rag_enhanced_video.mp4'), 
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        results = []
        
        with tqdm(total=total_frames, desc="Processing video with RAG-guided diffusion", unit="frame") as pbar:
            frame_idx = 0
            improvements_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    original_conf = self.get_confidence(frame)
                    original_detections = self.get_all_detections(frame)
                    
                    if conf_lower <= original_conf <= conf_upper:
                        try:
                            bbox = self.detect_bbox(frame)
                            category = self.determine_category_from_frame(frame, bbox)
                            
                            improvement, best_frame, best_strength, strategy = self.find_best_rag_perturbation(
                                frame, bbox, temp_dir, category, 0.9
                            )
                            
                            final_conf = self.get_confidence(best_frame)
                            final_detections = self.get_all_detections(best_frame)
                            
                            # Save RAG mask for this frame
                            try:
                                rag_mask, top_original = self.get_rag_combined_mask_and_original(frame, temp_dir, category)
                                cv2.imwrite(os.path.join(rag_masks_dir, f'frame_{frame_idx:06d}_rag_mask.png'), rag_mask)
                                cv2.imwrite(os.path.join(rag_masks_dir, f'frame_{frame_idx:06d}_top_original.png'), top_original)
                                
                                # Save images for debugging
                                bbox = self.detect_bbox(frame)
                                roi, _ = self._get_roi(frame, bbox)
                                if roi is not None:
                                    rag_mask_resized = cv2.resize(rag_mask, (roi.shape[1], roi.shape[0]))
                                    top_original_resized = cv2.resize(top_original, (roi.shape[1], roi.shape[0]))
                                    importance = cv2.normalize(rag_mask_resized, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
                                    
                                    # Create reference blend for debugging (masked version)
                                    importance_mask = (importance > 0.3).astype(np.float32)
                                    importance_3ch = np.stack([importance_mask] * 3, axis=2)
                                    reference_blend = (top_original_resized.astype(np.float32) * importance_3ch).astype(np.uint8)
                                    
                                    # Save debugging images
                                    cv2.imwrite(os.path.join(rag_masks_dir, f'frame_{frame_idx:06d}_base_image.png'), roi)  # Base image (original ROI)
                                    cv2.imwrite(os.path.join(rag_masks_dir, f'frame_{frame_idx:06d}_guiding_image.png'), top_original_resized)  # Guiding image (colored)
                                    cv2.imwrite(os.path.join(rag_masks_dir, f'frame_{frame_idx:06d}_reference_blend.png'), reference_blend)  # Masked guidance
                                    cv2.imwrite(os.path.join(rag_masks_dir, f'frame_{frame_idx:06d}_diffusion_mask.png'), mask_levels)  # Multi-level diffusion mask
                                    cv2.imwrite(os.path.join(rag_masks_dir, f'frame_{frame_idx:06d}_importance_mask.png'), (importance * 255).astype(np.uint8))  # Original importance
                                    
                            except Exception as e:
                                print(f"Warning: Could not save RAG mask for frame {frame_idx}: {e}")
                            
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
                                    'Category': category
                                })
                            else:
                                out_enhanced.write(frame)
                            
                        except Exception as e:
                            print(f"Error processing frame {frame_idx}: {e}")
                            out_enhanced.write(frame)
                            improvement = 0.0
                            best_strength = 0.0
                            strategy = "error"
                            category = "unknown"
                            final_conf = original_conf
                            final_detections = original_detections
                    else:
                        out_enhanced.write(frame)
                        improvement = 0.0
                        best_strength = 0.0
                        strategy = "no_enhancement"
                        category = "n/a"
                        final_conf = original_conf
                        final_detections = original_detections
                    
                    # Store results
                    results.append({
                        'frame_idx': frame_idx,
                        'original_conf': original_conf,
                        'final_conf': final_conf,
                        'improvement': improvement,
                        'strength': best_strength,
                        'strategy': strategy,
                        'category': category,
                        'original_detections': original_detections,
                        'final_detections': final_detections,
                        'needs_enhancement': conf_lower <= original_conf <= conf_upper
                    })
                    
                except Exception as e:
                    print(f"Critical error processing frame {frame_idx}: {e}")
                    out_enhanced.write(frame)
                    
                frame_idx += 1
                pbar.update(1)
                
        cap.release()
        out_enhanced.release()
        shutil.rmtree(temp_dir)
        
        print(f"\n✓ Processing complete! {improvements_count}/{total_frames} frames improved.")
        print(f"✓ Side-by-side comparisons saved in '{output_dir}/comparisons/' directory.")
        self.generate_analysis_plots(results, conf_upper, conf_lower, output_dir)
        return results

    def _create_side_by_side_comparison(self, original_frame: np.ndarray, perturbed_frame: np.ndarray, 
                                      original_conf: float, perturbed_conf: float, 
                                      frame_idx: int, output_dir: str) -> None:
        """Creates and saves a side-by-side comparison image with confidence scores and RAG masks."""
        # Get frame dimensions
        h, w = original_frame.shape[:2]
        
        # Create side-by-side image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original_frame
        comparison[:, w:] = perturbed_frame
        
        # Add text with confidence scores
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Original frame text
        original_text = f"Original: {original_conf:.3f}"
        cv2.putText(comparison, original_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
        
        # Perturbed frame text
        perturbed_text = f"RAG Enhanced: {perturbed_conf:.3f}"
        cv2.putText(comparison, perturbed_text, (w + 10, 30), font, font_scale, (0, 255, 0), thickness)
        
        # Improvement text
        improvement = perturbed_conf - original_conf
        improvement_text = f"Improvement: {improvement:+.3f}"
        improvement_color = (0, 255, 0) if improvement > 0 else (0, 0, 255)
        cv2.putText(comparison, improvement_text, (w // 2 - 50, h - 20), font, font_scale, improvement_color, thickness)
        
        # Add frame index
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(comparison, frame_text, (10, h - 50), font, font_scale, (255, 255, 255), thickness)
        
        # RAG indicator
        rag_text = "RAG-Guided"
        cv2.putText(comparison, rag_text, (w + 10, h - 50), font, font_scale, (255, 255, 0), thickness)
        
        # Create directories
        comparison_dir = os.path.join(output_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Save comparison image
        comparison_path = os.path.join(comparison_dir, f'frame_{frame_idx:06d}_rag_comparison.jpg')
        cv2.imwrite(comparison_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Print debug info for first few saves
        if frame_idx < 5:
            print(f"  📸 Saved RAG comparison image: {comparison_path}")
            print(f"     Original conf: {original_conf:.3f}, RAG Enhanced conf: {perturbed_conf:.3f}, Improvement: {improvement:+.3f}")

    def generate_analysis_plots(
        self, results: List[Dict], conf_upper: float, conf_lower: float, output_dir: str
    ):
        """Generates comprehensive analysis plots for RAG-guided diffusion."""
        # Extract data
        frame_indices = [r['frame_idx'] for r in results]
        original_confs = [r['original_conf'] for r in results]
        improvements = [r['improvement'] for r in results]
        final_confs = [o + i for o, i in zip(original_confs, improvements)]
        
        improved_results = [r for r in results if r['improvement'] > 0.001]
        strategies_used = [r['strategy'] for r in improved_results]
        strengths = [r['strength'] for r in improved_results]
        categories_used = [r['category'] for r in improved_results]
        
        all_original_detections = [r['original_detections'] for r in results]
        all_final_detections = [r['final_detections'] for r in results]
        
        # Calculate mAP
        precisions, recalls, mAP = self._calculate_pr_map(all_original_detections, all_final_detections)
        
        # Create plots
        fig, axes = plt.subplots(3, 2, figsize=(22, 20), dpi=120)
        fig.suptitle('RAG-Guided Diffusion Video Enhancement Analysis', fontsize=20, fontweight='bold')
        ax = axes.flatten()
        
        # Plot 0: Confidence over time
        ax[0].plot(frame_indices, original_confs, 'r-', label='Original Conf', alpha=0.6, linewidth=1)
        ax[0].plot(frame_indices, final_confs, 'g-', label='RAG Enhanced Conf', alpha=0.7, linewidth=1.5)
        ax[0].axhline(conf_upper, color='orange', ls='--', label=f'Upper={conf_upper}')
        ax[0].axhline(conf_lower, color='purple', ls='--', label=f'Lower={conf_lower}')
        ax[0].fill_between(frame_indices, conf_lower, conf_upper, color='yellow', alpha=0.15, label='Target Range')
        ax[0].set_title('Confidence Over Time (RAG-Guided)')
        ax[0].set_xlabel('Frame')
        ax[0].set_ylabel('Confidence')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # Plot 1: Improvement distribution
        ax[1].hist([r['improvement'] for r in improved_results], bins=25, color='lightgreen', edgecolor='black')
        ax[1].set_title('Distribution of Positive Improvements (RAG-Guided)')
        ax[1].set_xlabel('Confidence Gain')
        ax[1].set_ylabel('Frequency')
        ax[1].grid(True, alpha=0.3)

        # Plot 2: Category performance
        if improved_results and categories_used:
            from collections import Counter
            category_counts = Counter(categories_used)
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            
            ax[2].bar(categories, counts, color='skyblue', edgecolor='black')
            ax[2].set_title('RAG Category Usage in Improvements')
            ax[2].set_xlabel('Car View Category')
            ax[2].set_ylabel('Number of Improvements')
            ax[2].tick_params(axis='x', rotation=45)
            ax[2].grid(True, alpha=0.3)
        else:
            ax[2].text(0.5, 0.5, 'No category data available', ha='center', va='center', transform=ax[2].transAxes)

        # Plot 3: Precision-Recall curve
        ax[3].plot(recalls, precisions, color='navy', lw=2, label=f'RAG Enhanced vs. Original (mAP = {mAP:.3f})')
        ax[3].set_title('Precision-Recall Curve (mAP @ 0.5 IoU)')
        ax[3].set_xlabel('Recall')
        ax[3].set_ylabel('Precision')
        ax[3].set_xlim([0.0, 1.0])
        ax[3].set_ylim([0.0, 1.05])
        ax[3].legend()
        ax[3].grid(True, alpha=0.4)

        # Plot 4: Strength distribution
        if strengths:
            ax[4].hist(strengths, bins=20, color='lightblue', edgecolor='black')
            ax[4].axvline(np.mean(strengths), color='red', ls='--', label=f'Mean: {np.mean(strengths):.3f}')
            ax[4].set_title('Distribution of Effective RAG Strengths')
            ax[4].set_xlabel('Perturbation Strength')
            ax[4].set_ylabel('Frequency')
            ax[4].legend()
            ax[4].grid(True, alpha=0.3)

        # Plot 5: Summary statistics
        total_improved = len(improved_results)
        total_in_range = sum(1 for r in results if r['needs_enhancement'])
        from collections import Counter
        strategy_counts = Counter(strategies_used)
        category_counts = Counter(categories_used)
        
        stats_text = f"""
        RAG-GUIDED DIFFUSION SUMMARY
        {'-'*30}
        Total Frames: {len(results):,}
        Frames in Target Range: {total_in_range:,}
        Frames Improved: {total_improved:,}
        Success Rate: {total_improved/total_in_range*100 if total_in_range > 0 else 0:.1f}%
        
        Avg. Improvement: {np.mean([i for i in improvements if i > 0] or [0]):.4f}
        Avg. Effective Strength: {np.mean(strengths or [0]):.4f}
        mAP @ 0.5 IoU: {mAP:.3f}
        """
        
        if category_counts:
            stats_text += "\n\nRAG CATEGORY USAGE:\n" + '-'*20
            for category, count in category_counts.most_common():
                percentage = count / len(improved_results) * 100
                stats_text += f"\n{category}: {count} ({percentage:.1f}%)"
        
        # Add performance metrics
        cache_efficiency = len(self.detection_cache) / max(len(results), 1) * 100
        rag_cache_efficiency = len(self.rag_mask_cache) / max(len(results), 1) * 100
        stats_text += f"\n\nPERFORMANCE:\n" + '-'*15
        stats_text += f"\nDetection Cache Hit Rate: {cache_efficiency:.1f}%"
        stats_text += f"\nRAG Mask Cache Hit Rate: {rag_cache_efficiency:.1f}%"
        stats_text += f"\nDetection Cache: {len(self.detection_cache)} entries"
        stats_text += f"\nRAG Mask Cache: {len(self.rag_mask_cache)} entries"
        
        ax[5].text(0.05, 0.95, stats_text.strip(), transform=ax[5].transAxes, fontsize=11, 
                  verticalalignment='top', fontfamily='monospace', 
                  bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
        ax[5].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, 'rag_diffusion_analysis_summary.png'))
        plt.show()

# --- Main Execution ---

def main_rag_diffusion_analysis(
    video_path: str, conf_upper: float, conf_lower: float,
    strength_lower: float, strength_upper: float, strength_jump: float
):
    """Main function to run RAG-guided diffusion-based video analysis."""
    set_random_seeds(42)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    analyzer = RAGDiffusionVideoAnalyzer()
    output_dir = "results_rag_diffusion"
    
    results = analyzer.process_video(
        video_path, conf_upper, conf_lower, output_dir,
        strength_lower, strength_upper, strength_jump
    )
    print(f"\n✅ RAG-guided diffusion analysis complete. Results saved in '{output_dir}'.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    VIDEO_FILE = "test_video_pertubation.mp4"  # Use existing video file
    
    # Optimized confidence thresholds for better targeting
    CONF_UPPER_THRESHOLD = 0.75  # Slightly lower upper bound for more frames
    CONF_LOWER_THRESHOLD = 0.65  # Lower threshold to catch more improvement opportunities
    
    # Strength settings for multiple strength testing - optimized for confidence improvement
    STRENGTH_LOWER_BOUND = 0.5   # Start with higher minimum strength
    STRENGTH_UPPER_BOUND = 0.9  # Higher maximum for stronger effects
    STRENGTH_JUMP_FACTOR = 0.3  # Smaller steps for finer tuning (0.4, 0.55, 0.7, 0.85)
    
    print("🚀 Running RAG-guided diffusion-based video analysis.")
    print(f"Video: {VIDEO_FILE}")
    print(f"Confidence Target: {CONF_LOWER_THRESHOLD} - {CONF_UPPER_THRESHOLD}")
    print(f"Strength Range: {STRENGTH_LOWER_BOUND} - {STRENGTH_UPPER_BOUND} (step: {STRENGTH_JUMP_FACTOR})")
    print("Using Visual RAG for intelligent saliency mask retrieval")
    
    main_rag_diffusion_analysis(
        video_path=VIDEO_FILE,
        conf_upper=CONF_UPPER_THRESHOLD,
        conf_lower=CONF_LOWER_THRESHOLD,
        strength_lower=STRENGTH_LOWER_BOUND,
        strength_upper=STRENGTH_UPPER_BOUND,
        strength_jump=STRENGTH_JUMP_FACTOR
    )