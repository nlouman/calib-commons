import os
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
from pathlib import Path

def save_calibration_to_json(mtx, dist, filename):
    calibration_data = {
        "sensors": {
            "RGB": {
                "intrinsics": {
                    "type_id": "opencv-matrix",
                    "rows": mtx.shape[0],
                    "cols": mtx.shape[1],
                    "dt": "d",
                    "data": mtx.flatten().tolist()
                },
                "distortionCoefficients": {
                    "type_id": "opencv-matrix",
                    "rows": dist.shape[0],
                    "cols": dist.shape[1],
                    "dt": "d",
                    "data": dist.flatten().tolist()
                }
            }
        }
    }

    with open(filename, 'w') as file:
        json.dump(calibration_data, file, indent=4)

def extract_frames_from_video(video_path, output_dir, sampling_step):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_dir, exist_ok=True)
    for frame_count in tqdm(range(0, total_frames, sampling_step), desc="Extracting frames"):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = video.read()
        if not success:
            print(f"Warning: Failed to read frame at position {frame_count}")
            continue
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
    video.release()

def calibrate_camera_from_images(images_path, square_size, col, row, show_corners=False):
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corner_sub_pix_win_size = 11

    objpoints, imgpoints = [], []
    files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    for filename in tqdm(files, desc="Processing images"):
        file_path = os.path.join(images_path, filename)
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray) # ADDED FOR SONY INTERNAL CALIBRATION

        # ret, corners = cv2.findChessboardCorners(gray, (col, row), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        # if ret == 0:
            # print(f"Chessboard corners not found in {filename}. Trying with findChessboardCornersSB...")
        ret, corners = cv2.findChessboardCornersSB(gray, (col, row))
        if ret:
            # corners2 = np.squeeze(cv2.cornerSubPix(gray, corners, (corner_sub_pix_win_size, corner_sub_pix_win_size), (-1, -1), criteria))
            objpoints.append(objp)
            imgpoints.append(corners)

            if show_corners:
                img = cv2.drawChessboardCorners(img, (col, row), corners, ret)
        else:
            print(f"Chessboard corners not found in {filename}.")
        # elif show_corners:
        #     print(f"No corners detected in {filename}. Showing raw image.")

        if show_corners:
            cv2.putText(img, f"Image: {filename}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            corner_dir = os.path.join(images_path, 'detected_corners')
            if not os.path.exists(corner_dir):
                os.makedirs(corner_dir)
            cv2.imwrite(os.path.join(corner_dir, filename), img)
            # cv2.imshow('Image with Corners', img)
            # key = cv2.waitKey(0)
            # if key == 27:  # Press 'Esc' to exit early
                # break

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        raise ValueError("No valid chessboard patterns were detected.")

    print(f"Detected chessboard patterns in {len(objpoints)} out of {len(files)} images.")

    if "dslr" not in images_path.lower() and "kinect" not in images_path.lower() and "d405" not in images_path.lower():
        flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
        ret, mtx, _, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)
        dist = np.zeros((1, 5))
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"Camera calibrated with reprojection error (RMSE): {ret:.2f} [pix]")

     # --- Calculate Mean and Per-Image Reprojection Error ---
    total_error = 0
    total_points = 0

    print("\nPer-image reprojection errors:")
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        mean_error_i = error / len(imgpoints2)
        print(f"  [{i:03}] {files[i]}: {mean_error_i:.4f} px")
        total_error += error
        total_points += len(imgpoints2)

    mean_error = total_error / total_points
    print(f"\nMean reprojection error: {mean_error:.2f} [pix]")


    return ret, mtx, dist

def main():
    parser = argparse.ArgumentParser(description="Camera calibration script using videos or images.")
    parser.add_argument("--use_videos", action="store_true", help="Set this flag to use videos instead of images.")
    parser.add_argument("--data_directory", type=str, default=None, help="Path to the data directory.")
    parser.add_argument("--output_parent_directory", type=str, default=None, help="Path to the directory where the output folder will be created.")
    parser.add_argument("--square_size", type=float, required=True, help="Size of a chessboard square in meters.")
    parser.add_argument("--chessboard_width", type=int, required=True, help="Number of inner corners in chessboard width.")
    parser.add_argument("--chessboard_height", type=int, required=True, help="Number of inner corners in chessboard height.")
    parser.add_argument("--sampling_step", type=int, default=45, help="Number of frames to skip during frame extraction (used only with videos).")
    parser.add_argument("--show_corners", action="store_true", help="Show detected corners on the images.")

    args = parser.parse_args()

    data_directory = args.data_directory or os.getcwd()
    output_parent_directory = args.output_parent_directory or data_directory

    # output_subfolder_name = "calibrate_intrinsics_output"

    output_directory = output_parent_directory # os.path.join(output_parent_directory, output_subfolder_name)
    os.makedirs(output_directory, exist_ok=True)

    # If folder is not empty, move the existing files to a backup folder   
    existing_files = [f for f in os.listdir(output_directory) if os.path.isfile(os.path.join(output_directory, f))]
    if existing_files:
        backup_folders = [folder for folder in os.listdir(output_directory) if folder.startswith("backup_") and os.path.isdir(os.path.join(output_directory, folder))]
        backup_folders.sort()
        if backup_folders:
            last_backup = int(backup_folders[-1].split("_")[1])
            new_backup_folder = os.path.join(output_directory, f"backup_{last_backup + 1:03d}")
        else:
            new_backup_folder = os.path.join(output_directory, "backup_000")
        os.makedirs(new_backup_folder, exist_ok=True)
        for file in existing_files:
            os.rename(os.path.join(output_directory, file), os.path.join(new_backup_folder, file))
    intrinsics_dir = output_directory
    os.makedirs(intrinsics_dir, exist_ok=True)

    if args.use_videos:
        sampled_frames_dir = os.path.join(output_directory, "sampled_frames")
        os.makedirs(sampled_frames_dir, exist_ok=True)
        images_directory = sampled_frames_dir

        for video_file in os.listdir(data_directory):
            if video_file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                camera_name = Path(video_file).stem
                camera_output_dir = os.path.join(sampled_frames_dir, camera_name)
                os.makedirs(camera_output_dir, exist_ok=True)

                video_path = os.path.join(data_directory, video_file)
                print(f"Extracting frames from {video_file}...")
                extract_frames_from_video(video_path, camera_output_dir, args.sampling_step)
        print("")
    else:
        images_directory = data_directory

    for folder in os.listdir(images_directory):
        # if folder == output_subfolder_name:
        #     continue
        folder_path = os.path.join(images_directory, folder)
        if os.path.isdir(folder_path) and not 'info' in folder:
            print(f"Calibrating camera for {folder}...")
            ret, mtx, dist = calibrate_camera_from_images(folder_path, args.square_size, args.chessboard_width, args.chessboard_height, args.show_corners)
            intrinsics_file = os.path.join(intrinsics_dir, f'{folder}_intrinsics.json')
            save_calibration_to_json(mtx, dist, intrinsics_file)
            print(f"Calibration data saved for {folder}.")

if __name__ == "__main__":
    main()
