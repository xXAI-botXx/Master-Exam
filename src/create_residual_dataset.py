# extracts sub datasets from multiprocessing

import os
import shutil 
import re

import numpy as np
import cv2

# Change these lines:
base_folder = "D:/Cache/nms10000_0_0_2500_2500"
complex_folder = "D:/Cache/nms10000_1_0_2500_2500"
new_output_folder = "D:/Cache/nms1000_residual_reflection"    # the name will be advanced adjusted

# Get all used idx
used_idx = []
for cur_image in os.listdir(os.path.join(base_folder, "buildings")):
    numbers = re.findall(r'\d+', string=cur_image)
    if len(numbers) <= 0:
        raise ValueError(f"No numbers in '{cur_image}' found!")
    used_idx += [numbers[0]]

# Create paths / clear path
new_buildings_path = f"{new_output_folder}/buildings"
new_interpolated_path = f"{new_output_folder}/interpolated"
if os.path.exists(new_output_folder):
    shutil.rmtree(new_output_folder)
os.makedirs(new_output_folder, exist_ok=False)
os.makedirs(new_buildings_path, exist_ok=False)
os.makedirs(new_interpolated_path, exist_ok=False)

# Create new Dataset
for cur_idx in used_idx:
    input_img_path = f"{base_folder}/buildings/buildings_{cur_idx}.png"
    base_img_path = f"{base_folder}/interpolated/{cur_idx}_LAEQ_256.png"
    complex_img_path = f"{complex_folder}/interpolated/{cur_idx}_LAEQ_256.png"

    shutil.copyfile(input_img_path, f"{new_buildings_path}/buildings_{cur_idx}.png")

    base_img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # Min: 51.000000, Max: 255.000000, Mean: 161.898163
    print(f"Base Img = Min: {base_img.min():.6f}, Max: {base_img.max():.6f}, Mean: {base_img.mean():.6f}")
    complex_img = cv2.imread(complex_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # Min: 51.000000, Max: 255.000000, Mean: 161.224197
    print(f"Complex Img = Min: {complex_img.min():.6f}, Max: {complex_img.max():.6f}, Mean: {complex_img.mean():.6f}")

    complex_only_img = (complex_img - base_img) *-1 # *-2
    # Min: -0.000000, Max: 166.000000, Mean: 1.347931 -> Before: Min: -80.0, Max: 0.0
    print(f"Complex Only Img = Min: {complex_only_img.min():.6f}, Max: {complex_only_img.max():.6f}, Mean: {complex_only_img.mean():.6f}")

    # change to np.uint8 or np.uint16, cv2 only support unsigned int
    if np.any(complex_only_img < 0.0):
        raise ValueError(f"Image have minus values which will be overflowd/removed! Number of minus values: {np.sum(complex_only_img[complex_only_img < 0.0])}")

    if np.any(complex_only_img > 255.0):
        raise ValueError("Found Values bigger than 255.0")
    
    complex_only_img = (complex_only_img).astype(np.uint8)

    cv2.imwrite(f"{new_interpolated_path}/{cur_idx}_LAEQ_256.png", complex_only_img)






