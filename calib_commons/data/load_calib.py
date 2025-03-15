import json 
import numpy as np

from calib_commons.intrinsics import Intrinsics
import os
import cv2

def load_intrinsics(filename):
    
    # Load the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Extract camera matrix
    intrinsics = data['sensors']['RGB']['intrinsics']['data']
    camera_matrix = np.array(intrinsics).reshape(3, 3).astype(np.float32)
    
    # Extract distortion coefficients
    distortion_coeffs = data['sensors']['RGB']['distortionCoefficients']['data']
    distortion_coeffs = np.array(distortion_coeffs).astype(np.float32)
    
    return camera_matrix, distortion_coeffs

def construct_cameras_intrinsics(images_parent_folder, 
                         intrinsics_folder) -> Intrinsics:
    
    image_folders = {cam: os.path.join(images_parent_folder, cam) for cam in os.listdir(images_parent_folder) if os.path.isdir(os.path.join(images_parent_folder, cam))}
    intrinsics_paths = {cam: os.path.join(intrinsics_folder, cam + "_intrinsics.json") for cam in image_folders}

    intrinsics = {}
    for cam in image_folders:
        intrinsics[cam] = construct_camera_intrinsics(image_folders[cam], intrinsics_paths[cam])
    
    return intrinsics
    


def construct_camera_intrinsics(images_folder: str, 
                         intrinsics_path: str) -> Intrinsics:

    # Find all images in the folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Take the image with the smallest number
    first_image_path = os.path.join(images_folder, image_files[0])

    # Load the image
    image = cv2.imread(first_image_path)
    resolution = (image.shape[1], image.shape[0])  


    K, _ = load_intrinsics(intrinsics_path)
    return Intrinsics(K, resolution)