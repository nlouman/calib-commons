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

        ret, corners = cv2.findChessboardCorners(gray, (col, row))
        if ret:
            corners2 = np.squeeze(cv2.cornerSubPix(gray, corners, (corner_sub_pix_win_size, corner_sub_pix_win_size), (-1, -1), criteria))
            objpoints.append(objp)
            imgpoints.append(corners2)

            if show_corners:
                img = cv2.drawChessboardCorners(img, (col, row), corners2, ret)
        # elif show_corners:
        #     print(f"No corners detected in {filename}. Showing raw image.")

        if show_corners:
            cv2.imshow('Image with Corners', img)
            key = cv2.waitKey(0)
            if key == 27:  # Press 'Esc' to exit early
                break

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        raise ValueError("No valid chessboard patterns were detected.")

    print(f"Detected chessboard patterns in {len(objpoints)} out of {len(files)} images.")

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"Camera calibrated with reprojection error (RMSE): {ret:.2f} [pix]")

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

    output_subfolder_name = "calibrate_intrinsics_output"

    output_directory = os.path.join(output_parent_directory, output_subfolder_name)
    os.makedirs(output_directory, exist_ok=True)

    intrinsics_dir = os.path.join(output_directory, "camera_intrinsics")
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
        if folder == output_subfolder_name:
            continue
        folder_path = os.path.join(images_directory, folder)
        if os.path.isdir(folder_path):
            print(f"Calibrating camera for {folder}...")
            ret, mtx, dist = calibrate_camera_from_images(folder_path, args.square_size, args.chessboard_width, args.chessboard_height, args.show_corners)
            if "dslr" not in folder.lower():
                dist = np.zeros(dist.shape, dtype=dist.dtype)
            intrinsics_file = os.path.join(intrinsics_dir, f'{folder}_intrinsics.json')
            save_calibration_to_json(mtx, dist, intrinsics_file)
            print(f"Calibration data saved for {folder}.")

if __name__ == "__main__":
    main()
