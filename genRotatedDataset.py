import cv2
import os
import numpy as np
from tqdm import tqdm

# Function to rotate an image by a given angle
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

# Function to generate rotated images and labels
def generate_rotated_dataset(dataset_path, output_dir):
    rotation_angles = [0, 90, 180, 270]
    image_list = []

    try:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_list.append(os.path.join(root, file))

        progress_bar = tqdm(total=len(image_list) * len(rotation_angles), desc="Processing", unit="image")

        for image_path in image_list:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to read image '{image_path}'")
                    progress_bar.update(len(rotation_angles))
                    continue

                for angle in rotation_angles:
                    rotated_image = rotate_image(image, angle)
                    angle_dir = os.path.join(output_dir, f"{angle}_Degrees")
                    os.makedirs(angle_dir, exist_ok=True)
                    rotated_image_name = f"{angle}_img{len(os.listdir(angle_dir)) + 1}.jpg"
                    rotated_image_path = os.path.join(angle_dir, rotated_image_name)
                    cv2.imwrite(rotated_image_path, rotated_image)
                    progress_bar.update(1)
            
            except Exception as e:
                print(f"Error processing image '{image_path}': {e}")
                progress_bar.update(len(rotation_angles))
        
        progress_bar.close()

    except KeyboardInterrupt:
        print("Processing interrupted.")
    finally:
        print("Processing completed.")

# Specify the paths
dataset_path = "Instagram/"
output_dir = "Dataset/insta"

# Generate the rotated dataset
generate_rotated_dataset(dataset_path, output_dir)
