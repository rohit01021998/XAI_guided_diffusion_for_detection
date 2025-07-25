import os
import cv2

# Input and output directories
input_root = "images"
output_root = "videos"

# Create output directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Loop over folders
for folder_name in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Collect all jpg files sorted
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    )
    if not image_files:
        print(f"No images in '{folder_name}', skipping.")
        continue

    # Read first image to get frame size
    first_image_path = os.path.join(folder_path, image_files[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        print(f"Could not read first image in '{folder_name}', skipping.")
        continue

    height, width, _ = first_frame.shape
    size = (width, height)

    # Output video path
    output_video_path = os.path.join(output_root, f"{folder_name}.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0  # 30 FPS
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    print(f"Generating video for '{folder_name}' with {len(image_files)} images at {fps} FPS.")

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: could not read {img_file}, skipping.")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, size)
        out.write(frame)

    out.release()
    print(f"Created video: {output_video_path}")
