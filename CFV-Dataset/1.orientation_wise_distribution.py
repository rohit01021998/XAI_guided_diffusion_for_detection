import os
import shutil

# Paths
input_root = "images"
output_root = "output"

# Categories
categories = ["front", "front-right", "back-right", "back", "back-left", "front-left"]
for cat in categories:
    os.makedirs(os.path.join(output_root, cat), exist_ok=True)

def angle_to_category(angle):
    angle = angle % 360

    if angle >= 345 or angle <= 15:
        return "front"
    elif 15 < angle <= 105:
        return "front-right"
    elif 105 < angle <= 165:
        return "back-right"
    elif 165 < angle <= 195:
        return "back"
    elif 195 < angle <= 255:
        return "back-left"
    elif 255 < angle <= 345:
        return "front-left"
    else:
        raise ValueError(f"Unhandled angle: {angle}")

# Process
for folder_name in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".jpg"):
            continue

        name_part = os.path.splitext(filename)[0]
        try:
            angle = int(name_part)
        except ValueError:
            print(f"Skipping invalid filename: {filename}")
            continue

        category = angle_to_category(angle)

        src = os.path.join(folder_path, filename)
        dst = os.path.join(output_root, category, f"{folder_name}_{filename}")

        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
