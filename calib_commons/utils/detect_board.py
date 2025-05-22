import cv2 as cv
import os
import numpy as np
from typing import Dict
from enum import Enum
from tqdm import tqdm
import re

from calib_commons.data.load_calib import load_intrinsics
# from calib_board.core.observationCheckerboard import ObservationCheckerboard


class BoardType(Enum): 
    CHESSBOARD = "chessboard"
    CHARUCO = "charuco"

def detect_board_corners(images_parent_folder: str, 
                         board_type: BoardType,
                         charuco_detector: cv.aruco.CharucoDetector,
                        columns: int, 
                        rows: int, 
                        intrinsics_folder: str,
                        undistort: bool = True, 
                        display: bool = False, 
                        save_images_with_overlayed_detected_corners: bool = True) -> Dict[str, Dict[int, np.array]]:
    
    # camera_names = list({image_file.stem.split('_')[0] for ext in ["*.jpg", "*.png", "*.jpeg"] for image_file in images_parent_folder.glob(ext)})

    
    # intrinsics_folder = 
    correspondences_array = {}  # dict by cam id
    if board_type == BoardType.CHESSBOARD:
        correspondences_array = detect_chessboards(images_parent_folder, columns, rows, intrinsics_folder, undistort, display, save_images_with_overlayed_detected_corners)
    elif board_type == BoardType.CHARUCO:
        correspondences_array = detect_charuco(images_parent_folder, charuco_detector, columns, rows, intrinsics_folder, undistort, display, save_images_with_overlayed_detected_corners)
    else:
        raise ValueError("board_type must be either 'chessboard' or 'charuco'")
    return correspondences_array

def numeric_prefix(filename):
    # Attempt to parse out a leading integer
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return 0  # or raise an error if strictly required
        

def detect_chessboards(images_parent_folder: str, 
                       columns: int, 
                       rows: int, 
                       intrinsics_folder: str = None, 
                       undistort: bool = True, 
                       display: bool = False, 
                       save_images_with_overlayed_detected_corners: bool = True) -> Dict[str, Dict[int, np.array]]:
    
    # camera_names = list({image_file.stem.split('_')[0] for ext in ["*.jpg", "*.png", "*.jpeg"] for image_file in images_parent_folder.glob(ext)})
    image_folders = {cam: os.path.join(images_parent_folder, cam) for cam in os.listdir(images_parent_folder) if os.path.isdir(os.path.join(images_parent_folder, cam)) and not 'info' in cam}

    intrinsics_paths = {cam: os.path.join(intrinsics_folder, cam + "_intrinsics.json") for cam in image_folders}

    correspondences = {}  # dict by cam id

    ## PARAMS ##
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corner_sub_pix_win_size = 5

    
    for cam, image_folder in tqdm(image_folders.items(), desc=f"Processing cameras", total=len(image_folders), leave=False): 
        correspondences[cam] = {}
        if save_images_with_overlayed_detected_corners: 
            corners_folder = os.path.join(image_folder, "extracted_corners")
            if not os.path.exists(corners_folder):
                os.makedirs(corners_folder)
                
        if undistort:
            # camera_matrix, distortion_coeffs, _ = load_intrinsics(intrinsics_paths[cam])
            camera_matrix, distortion_coeffs = load_intrinsics(intrinsics_paths[cam])

        files = os.listdir(image_folder)
        files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=numeric_prefix)
        
        for filename in tqdm(files, desc=f"Processing camera {cam}", total=len(files), leave=False):        
            k = int(filename.split('.')[0])

            file_path = os.path.join(image_folder, filename)
            print(filename)

            img = cv.imread(file_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, (columns, rows), cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE)  # row by row from top left point 
            if ret:

                corners2 = np.squeeze(cv.cornerSubPix(gray, corners, (corner_sub_pix_win_size, corner_sub_pix_win_size), (-1, -1), criteria)) 

                if display:
                    cv.drawChessboardCorners(img, (columns, rows), corners2, ret)

                    font = cv.FONT_HERSHEY_SIMPLEX  # Font style
                    font_scale = 0.5  # Font scale (font size)
                    color = (255, 255, 255)  # Color of the text (BGR - white)
                    thickness = 1  # Thickness of the lines used to draw the text

                    text = "0"
                    org = tuple(corners2[0, :].astype(int))
                    cv.putText(img, text, org, font, font_scale, color, thickness, cv.LINE_AA)

                    text = "1"
                    org = tuple(corners2[1, :].astype(int))
                    cv.putText(img, text, org, font, font_scale, color, thickness, cv.LINE_AA)

                    # text = "14"
                    # org = tuple(corners2[14, :].astype(int))
                    # cv.putText(img, text, org, font, font_scale, color, thickness, cv.LINE_AA)
                    
                    cv.imshow('img', img)
                    cv.waitKey(1)

                    if save_images_with_overlayed_detected_corners:
                        file_path_save = os.path.join(corners_folder, filename)
                        cv.imwrite(file_path_save, img)

                if undistort:
                    points_reshaped = corners2.reshape(-1, 1, 2)
                    undistorted_points = cv.undistortPoints(points_reshaped, camera_matrix, distortion_coeffs, P=camera_matrix)
                    _2dpoints = undistorted_points.reshape(-1, 2)
                else: 
                    _2dpoints = corners2

                # correspondences[cam][k] = ObservationCheckerboard(_2dpoints)
                correspondences[cam][k] = _2dpoints
            else:
                print(f"Chessboard not detected in {filename}")

    return correspondences

def detect_charuco(images_parent_folder: str, 
                   charuco_detector, 
                   columns: int, 
                   rows: int, 
                   intrinsics_folder: str = None, 
                   undistort: bool = True, 
                   display: bool = False, 
                   save_images_with_overlayed_detected_corners: bool = True) -> Dict[str, Dict[int, np.array]]:
    
    image_folders = {cam: os.path.join(images_parent_folder, cam) for cam in os.listdir(images_parent_folder) if os.path.isdir(os.path.join(images_parent_folder, cam))}


    intrinsics_paths = {cam: os.path.join(intrinsics_folder, cam + "_intrinsics.json") for cam in image_folders}

    correspondences = {}  # dict by cam id

    for cam, image_folder in tqdm(image_folders.items(), desc=f"Processing cameras", total=len(image_folders), leave=False): 

    # for cam, image_folder in image_folders.items():
        correspondences[cam] = {}
        if save_images_with_overlayed_detected_corners: 
            corners_folder = os.path.join(image_folder, "extracted_corners")
            if not os.path.exists(corners_folder):
                os.makedirs(corners_folder)

        if undistort:
            camera_matrix, distortion_coeffs = load_intrinsics(intrinsics_paths[cam])

        files = os.listdir(image_folder)
        files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        files.sort(key=lambda x: int(x.split('.')[0]))
        
        # for filename in files:
        for filename in tqdm(files, desc=f"Processing camera {cam}", total=len(files), leave=False):        
            k = int(filename.split('.')[0])
            file_path = os.path.join(image_folder, filename)

            img = cv.imread(file_path)
            # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)    

            if display:
                if charuco_ids is not None:
                    for i in range(len(charuco_ids)):
                        corner = charuco_corners[i]
                        charuco_id = charuco_ids[i]
                        cv.circle(img, tuple(corner[0].astype(int)), 5, (0, 255, 0), -1)
                        cv.putText(img, str(charuco_id), tuple(corner[0].astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    for i in range(len(marker_corners)):
                        corner4 = marker_corners[i]
                        for j in range(4):
                            cv.circle(img, tuple(corner4[0, j, :].astype(int)), 2, (255, 255, 0), -1)

                    cv.imshow('img', img)
                    cv.waitKey(1)
                
                    if save_images_with_overlayed_detected_corners:
                        file_path_save = os.path.join(corners_folder, filename)
                        cv.imwrite(file_path_save, img)

            if charuco_corners is not None:
                if undistort:
                    points_reshaped = charuco_corners.reshape(-1, 1, 2)
                    undistorted_points = cv.undistortPoints(points_reshaped, camera_matrix, distortion_coeffs, P=camera_matrix)
                    _2dpoints = undistorted_points.reshape(-1, 2)
                else: 
                    _2dpoints = charuco_corners

                _2dpoints = np.full((columns * rows, 2), np.nan)

                for i in range(len(charuco_ids)):
                    _2dpoints[charuco_ids[i], :] = charuco_corners[i, 0, :]

                # correspondences[cam][k] = ObservationCheckerboard(_2dpoints)
                correspondences[cam][k] = _2dpoints

                # print(f"board added for {cam} at frame {k}")

    return correspondences

# if __name__ == "__main__":

