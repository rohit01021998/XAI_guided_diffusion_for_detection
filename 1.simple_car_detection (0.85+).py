#!/usr/bin/env python3
"""
Simple Car Detection and Classification Script

This script processes videos to:
1. Detect cars using YOLOv8n model
2. Classify detected cars using TFLite model
3. Save cropped images and saliency maps organized by class

Usage:    # Show results
    print_summary(output_dir)
    print(f"\n✅ Processing complete! Total cars detected: {total_detections}")
    print(f"📁 Results saved in: {output_dir}/")
    
    # Clean up temporary directory
    if os.path.exists("temp_saliency"):
        shutil.rmtree("temp_saliency")hon simple_car_detection.py
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import warnings
from tqdm import tqdm
from CropedSalencyMapCreator import yolov8_cropped_heatmap, get_params
warnings.filterwarnings('ignore')

# Classification classes
CLASSES = [  "back" ,  "front-right" ,  "back-left" ,  "back-right" ,  "front-left" ,  "front" ]

# Standard size for all saved images
STANDARD_SIZE = (224, 224)  # Width, Height

def create_output_dirs():
    """Create output directories for each class."""
    base_dir = "classified_cars_output"
    for class_name in CLASSES:
        os.makedirs(f"{base_dir}/{class_name}/cropped_images", exist_ok=True)
        os.makedirs(f"{base_dir}/{class_name}/saliency_maps", exist_ok=True)
    return base_dir

def load_models():
    """Load YOLO, TFLite and Saliency models."""
    print("Loading models...")
    
    # Load YOLO model
    yolo_model = YOLO('yolov8n.pt')
    
    # Load TFLite model
    tflite_model = tf.lite.Interpreter(model_path='model.tflite')
    tflite_model.allocate_tensors()
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()
    
    # Load Cropped Saliency Map model
    saliency_model = yolov8_cropped_heatmap(**get_params())
    
    print("Models loaded successfully!")
    return yolo_model, tflite_model, input_details, output_details, saliency_model

def classify_car(image, tflite_model, input_details, output_details):
    """Classify a car image using TFLite model."""
    # Convert and resize image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image = image.convert('RGB').resize((224, 224))
    img_array = np.array(image)[None].astype('float32')
    
    # Update tensor size and run inference
    tflite_model.resize_tensor_input(input_details[0]['index'], (1, 224, 224, 3))
    tflite_model.allocate_tensors()
    tflite_model.set_tensor(input_details[0]['index'], img_array)
    tflite_model.invoke()
    
    # Get prediction
    scores = tflite_model.get_tensor(output_details[0]['index'])
    predicted_class = CLASSES[scores.argmax()]
    confidence = float(scores.max())
    
    return predicted_class, confidence

def process_video(video_path, yolo_model, tflite_model, input_details, output_details, saliency_model, output_dir):
    """Process a single video file."""
    video_name = Path(video_path).stem
    print(f"Processing: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return 0
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    total_detections = 0
    
    # Create progress bar for frames
    with tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = yolo_model(frame)
            
            detection_idx = 0
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Check if it's a car with good confidence
                    if yolo_model.names[cls] == "car" and conf > 0.85:
                        # Crop the car region
                        cropped_car = frame[y1:y2, x1:x2]
                        
                        if cropped_car.size > 0:
                            # Resize cropped car to standard size
                            cropped_car_resized = cv2.resize(cropped_car, STANDARD_SIZE)
                            
                            # Classify the car using resized image
                            predicted_class, class_conf = classify_car(
                                cropped_car_resized, tflite_model, input_details, output_details
                            )
                            
                            # Create filenames
                            base_name = f"{video_name}_frame_{frame_count:04d}_det_{detection_idx}"
                            
                            # Save cropped image (resized)
                            crop_path = f"{output_dir}/{predicted_class}/cropped_images/{base_name}.jpg"
                            cv2.imwrite(crop_path, cropped_car_resized)
                            
                            # Create and save saliency map using CropedSalencyMapCreator
                            # Save current frame temporarily
                            temp_frame_path = f"temp_frame_{frame_count}.jpg"
                            cv2.imwrite(temp_frame_path, frame)
                            
                            # Create temporary saliency output directory
                            temp_saliency_dir = "temp_saliency"
                            os.makedirs(temp_saliency_dir, exist_ok=True)
                            
                            # Generate saliency map
                            try:
                                saliency_model.process(temp_frame_path, temp_saliency_dir, base_name)
                                
                                # Find the generated saliency file
                                saliency_files = [f for f in os.listdir(temp_saliency_dir) if f.startswith(base_name)]
                                if saliency_files:
                                    # Load the generated saliency map
                                    saliency_img = cv2.imread(os.path.join(temp_saliency_dir, saliency_files[0]))
                                    if saliency_img is not None:
                                        # Resize to standard size
                                        saliency_resized = cv2.resize(saliency_img, STANDARD_SIZE)
                                        saliency_path = f"{output_dir}/{predicted_class}/saliency_maps/{base_name}.jpg"
                                        cv2.imwrite(saliency_path, saliency_resized)
                                    
                                    # Clean up temporary saliency file
                                    os.remove(os.path.join(temp_saliency_dir, saliency_files[0]))
                            except Exception as e:
                                print(f"Warning: Could not generate saliency map for {base_name}: {e}")
                            
                            # Clean up temporary frame
                            if os.path.exists(temp_frame_path):
                                os.remove(temp_frame_path)
                            
                            total_detections += 1
                            detection_idx += 1
                            
                            # Update progress bar description with detection info
                            pbar.set_postfix({
                                'Cars': total_detections,
                                'Current': f"{predicted_class} ({class_conf:.2f})"
                            })
            
            frame_count += 1
            pbar.update(1)
            
            # Process every 10th frame for faster processing (optional)
            # for _ in range(9):
            #     ret, _ = cap.read()
            #     if not ret:
            #         break
            #     frame_count += 9
            #     pbar.update(9)
    
    cap.release()
    print(f"✅ Completed {video_name}: {total_detections} cars detected")
    return total_detections

def print_summary(output_dir):
    """Print processing summary."""
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    total_cars = 0
    for class_name in CLASSES:
        crop_dir = f"{output_dir}/{class_name}/cropped_images"
        count = len([f for f in os.listdir(crop_dir) if f.endswith('.jpg')]) if os.path.exists(crop_dir) else 0
        total_cars += count
        print(f"{class_name:12} - {count:3d} cars")
    
    print(f"{'TOTAL':12} - {total_cars:3d} cars")

def main():
    """Main function."""
    print("🚗 Simple Car Detection and Classification System")
    print("="*50)
    
    # Check required files
    required_files = ['yolov8n.pt', 'model.tflite']
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing files: {missing}")
        return
    
    if not os.path.exists('input_videos'):
        print("❌ Missing 'input_videos' folder")
        return
    
    # Get video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.MOV')
    video_files = [f for f in os.listdir('input_videos') 
                   if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print("❌ No video files found in input_videos/")
        return
    
    print(f"📹 Found {len(video_files)} video files")
    
    # Setup
    output_dir = create_output_dirs()
    yolo_model, tflite_model, input_details, output_details, saliency_model = load_models()
    
    # Process videos
    total_detections = 0
    print(f"\n🚀 Starting processing of {len(video_files)} video(s)...")
    
    # Create progress bar for videos
    with tqdm(video_files, desc="Overall Progress", unit="video") as video_pbar:
        for i, video_file in enumerate(video_pbar, 1):
            video_pbar.set_description(f"Video {i}/{len(video_files)}")
            video_path = os.path.join('input_videos', video_file)
            detections = process_video(
                video_path, yolo_model, tflite_model, 
                input_details, output_details, saliency_model, output_dir
            )
            total_detections += detections
            video_pbar.set_postfix({'Total Cars': total_detections})
    
    # Show results
    print_summary(output_dir)
    print(f"\n✅ Processing complete! Total cars detected: {total_detections}")
    print(f"📁 Results saved in: {output_dir}/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
