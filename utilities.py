import cv2
from ultralytics import YOLO
import os
import shutil
import re

def image_seperator(yolo_path = r'code\yolov8n.pt', input_video_path = r'SpotOnCounterFactualAnalysis/test-1.mp4'):
    '''
    applies yolo model detection on input video and seperates each folder into two folders
    i.e detected_frames and no_detected_frames folders.
    Supports both single video file and folder containing multiple videos.
    '''
    # Load your custom YOLOv8 model
    model = YOLO(yolo_path)  # Replace with your custom model's path
    
    # Check if input is a directory or a single file
    if os.path.isdir(input_video_path):
        # Process all video files in the directory
        video_files = [f for f in os.listdir(input_video_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
        
        if not video_files:
            print(f"No video files found in directory: {input_video_path}")
            return
            
        print(f"Found {len(video_files)} video files to process")
        
        for video_file in video_files:
            video_path = os.path.join(input_video_path, video_file)
            print(f"Processing video: {video_file}")
            process_single_video(model, video_path, video_file)
    else:
        # Process single video file
        video_filename = os.path.basename(input_video_path)
        process_single_video(model, input_video_path, video_filename)

def process_single_video(model, video_path, video_filename):
    '''
    Process a single video file for object detection
    '''
    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    
    # Use shared directories for all videos
    detected_dir = "detected_frames"
    no_detection_dir = "no_detection_frames"
    
    # Create directories to save frames if they don't exist
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(no_detection_dir, exist_ok=True)
    
    frame_count = 0  # Initialize frame counter
    
    while True:
        max_confidence = 0
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            print(f"Finished processing video: {video_filename}")
            break

        # Make a copy of the frame for display (with bounding boxes)
        display_frame = frame.copy()

        # Apply your YOLOv8 model for object detection
        results = model(frame)  # Make predictions on the frame

        detected = False  # Flag to check if any detection occurs

        # Iterate through detections and draw bounding boxes only on display_frame
        for result in results:
            boxes = result.boxes  # Get the boxes for this result

            for box in boxes:
                # Extract box coordinates, confidence, and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])  # Class index
                if conf>max_confidence:
                    max_confidence = conf
                # Check if the confidence is above threshold
                if model.names[cls] == "car" and conf > 0.1:
                    detected = True  # Set flag if detection occurs

                    # Draw the bounding box on the display frame only
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label and confidence to display frame only
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the original frame (without boxes) to the appropriate folder
        # Include video name in frame filename to avoid conflicts
        video_base_name = os.path.splitext(video_filename)[0]
        frame_filename = f"{video_base_name}_frame_{frame_count:04d}_{max_confidence:.2f}.jpg"
        if detected:
            cv2.imwrite(os.path.join(detected_dir, frame_filename), frame)
        else:
            cv2.imwrite(os.path.join(no_detection_dir, frame_filename), frame)

        # Display the frame with bounding boxes
        resized_frame = cv2.resize(display_frame, (640, 480))
        cv2.imshow(f'YOLOv8 Live Detection - {video_filename}', resized_frame)

        # Increment frame count
        frame_count += 1

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

import cv2

def process_single_image(model, image_path):
    """
    Process a single image file for object detection.

    Args:
        model: YOLOv8 model object.
        image_path: Path to the image file.

    Returns:
        max_confidence (float): Highest detection confidence in the image.
                                Returns 0 if no detections.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not read image at {image_path}")

    # Run detection
    results = model(image)

    max_confidence = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf > max_confidence:
                max_confidence = conf

    return max_confidence


def get_confidence_range(confidence):
    # Create the range bins
    lower = int(confidence * 10) / 10  # Lower bound of the range (rounded to nearest 0.1)
    upper = lower + 0.1  # Upper bound (next range)
    return f"{lower:.1f}-{upper:.1f}"

def collect_images_by_confidence(source_folder=r'detected_frames', target_folder=r'group_by_confidence'):
    # Ensure the target directory exists, create if not
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Iterate through the files in the source folder
    for filename in os.listdir(source_folder):
        print(f"Processing: {filename}")  # Debugging line to see each filename
        
        # Extract the confidence value using a regex pattern
        # Updated pattern to handle new filename format: {video_name}_frame_{number}_{confidence}.jpg
        match = re.search(r'_([0-9]+\.[0-9]+)\.', filename)  # Match confidence followed by dot
        if match:
            confidence = float(match.group(1))  # Extract the confidence value
            confidence_range = get_confidence_range(confidence)  # Get the confidence range
            range_folder = os.path.join(target_folder, confidence_range)

            # Create the folder for the confidence range if it doesn't exist
            if not os.path.exists(range_folder):
                os.makedirs(range_folder)
                print(f"Created folder: {range_folder}")  # Debugging line

            # Move the file to the corresponding folder
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(range_folder, filename)

            try:
                shutil.move(source_file, target_file)  # Move the file
                print(f"Moved {filename} to {range_folder}")  # Debugging line
            except FileNotFoundError as e:
                print(f"File not found: {filename} - {e}")  # Print any error during the move
            except PermissionError as e:
                print(f"Permission error: {filename} - {e}")  # Print any error during the move
            except Exception as e:
                print(f"Error moving {filename}: {e}")  # Print any error during the move
        else:
            print(f"No confidence value found in filename: {filename}")  # Debugging line if no match