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
def generate_rotated_dataset(coco_dataset_path, output_dir):
    rotation_angles = [0, 90, 180, 270]
    image_list = os.listdir(coco_dataset_path)

    # Initialize the progress bar
    progress_bar = tqdm(total=len(image_list) * len(rotation_angles), desc="Processing", unit="image")

    for image_name in image_list:
        image_path = os.path.join(coco_dataset_path, image_name)
        image = cv2.imread(image_path)

        # Rotate and save images under the corresponding angle subdirectory
        for angle in rotation_angles:
            rotated_image = rotate_image(image, angle)
            angle_dir = os.path.join(output_dir, f"{angle}_Degrees")
            os.makedirs(angle_dir, exist_ok=True)
            rotated_image_name = f"{angle}_img{len(os.listdir(angle_dir)) + 1}.jpg"
            rotated_image_path = os.path.join(angle_dir, rotated_image_name)
            cv2.imwrite(rotated_image_path, rotated_image)

            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar when the loop is finished
    progress_bar.close()

# Specify the paths
dataset_path = "Dataset/flickr30k_images"
output_dir = "Dataset/Rotated Images"

# Generate the rotated dataset
generate_rotated_dataset(dataset_path, output_dir)
