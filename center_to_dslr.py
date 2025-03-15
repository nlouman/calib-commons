#!/usr/bin/env python3
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Name of the JSON file that contains the camera poses
json_filename = '/home/balgrist/dev/calib-board/results/camera_poses.json'

def load_poses(json_filename):
    with open(json_filename, 'r') as f:
        data = json.load(f)
    return data

def compute_relative_poses(data):
    # Get the DSLR's transform (reference frame)
    dslr_data = data.get('dslr')
    if dslr_data is None:
        raise ValueError("Reference camera 'dslr' not found in the JSON data.")
    
    # Create a Rotation from the DSLR Euler angles (ZYX order)
    R_dslr = R.from_euler('ZYX', dslr_data['euler_ZYX'], degrees=False)
    t_dslr = np.array(dslr_data['t'])
    
    pose_lines = []
    centered_poses = {}

    # For the reference DSLR, the relative pose is the identity (translation zero, identity quaternion)
    centered_poses['dslr'] = {
        "t": [0.0, 0.0, 0.0],
        "quat": [0.0, 0.0, 0.0, 1.0]
    }
    
    # Loop over each camera in the JSON data
    for cam_name, cam_data in data.items():
        if cam_name == 'dslr':
            continue

        # Get camera rotation and translation
        R_cam = R.from_euler('ZYX', cam_data['euler_ZYX'], degrees=False)
        t_cam = np.array(cam_data['t'])
        
        # Compute the relative transformation:
        #   R_rel = R_dslr^{-1} * R_cam
        #   t_rel = R_dslr^{-1} * (t_cam - t_dslr)
        R_rel = R_dslr.inv() * R_cam
        t_rel = R_dslr.inv().apply(t_cam - t_dslr)
        
        # Convert the relative rotation to a quaternion (order: [x, y, z, w])
        quat_rel = R_rel.as_quat()
        
        # Round the results to 5 decimals for neatness
        t_rel_rounded = np.around(t_rel, 5).tolist()
        quat_rel_rounded = np.around(quat_rel, 5).tolist()
        
        # Save in dictionary for JSON output
        centered_poses[cam_name] = {
            "t": t_rel_rounded,
            "quat": quat_rel_rounded
        }
        
        # Create a formatted output line for console printing
        line = (
            f'PoseStruct("{cam_name}", 1, np.array({t_rel_rounded}), '
            f'R.from_quat({quat_rel_rounded}))'
        )
        pose_lines.append(line)
    
    return pose_lines, centered_poses

def write_centered_poses(output_filename, centered_poses):
    with open(output_filename, 'w') as f:
        json.dump(centered_poses, f, indent=4)

def main():
    data = load_poses(json_filename)
    pose_lines, centered_poses = compute_relative_poses(data)
    
    # Print to console
    print("[")
    for line in pose_lines:
        print("    " + line + ",")
    print("]")
    
    # Determine output file path (same directory as the input file)
    output_filename = os.path.join(os.path.dirname(json_filename), "camera_poses_centered.json")
    write_centered_poses(output_filename, centered_poses)
    print(f"\nCentered poses written to: {output_filename}")

if __name__ == '__main__':
    main()
