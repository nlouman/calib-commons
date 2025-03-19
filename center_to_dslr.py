#!/usr/bin/env python3
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Name of the JSON file that contains the camera poses
json_filename = '/home/fred/dev/calib-board/results/camera_poses.json'

R_tool_dslr = R.from_matrix([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

def load_poses(json_filename):
    with open(json_filename, 'r') as f:
        data = json.load(f)
    return data

def compute_relative_poses(data):
    # Find reference camera name
    for cam in data.keys():
        t = np.array(data[cam]['t'])
        if np.all(t == 0):
            ref_cam = cam
            break

    if 'dslr' in ref_cam:
        return  # already in dslr cam frame
    
    t_ref_dslr = np.array(data['dslr']['t'])
    t_dslr_ref = -1 * t_ref_dslr
    R_ref_dslr = R.from_euler('ZYX', data['dslr']['euler_ZYX'], degrees=True)
    print(f"R_ref_dslr angle: {np.linalg.norm(R_ref_dslr.as_rotvec(degrees=True))}")
    R_dslr_ref = R_ref_dslr.inv()
    t_dslr_ref = R_dslr_ref.apply(t_dslr_ref)
    
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
        R_ref_cam = R.from_euler('ZYX', cam_data['euler_ZYX'], degrees=True)
        t_ref_cam = np.array(cam_data['t'])
        
        # # Compute the relative transformation:
        # #   R_rel = R_dslr^{-1} * R_cam
        # #   t_rel = R_dslr^{-1} * (t_cam - t_dslr)
        # R_rel = R_dslr.inv() * R_cam
        # t_rel = R_dslr.inv().apply(t_cam - t_dslr)

        t_dslr_cam = t_dslr_ref + R_dslr_ref.apply(t_ref_cam)
        R_dslr_cam = R_dslr_ref * R_ref_cam

        t_dslr_cam_in_tool = R_tool_dslr.apply(t_dslr_cam)
        R_dslr_cam_in_tool = R_tool_dslr.inv() * R_dslr_cam * R_tool_dslr
        
        # Convert the relative rotation to a quaternion (order: [x, y, z, w])
        quat_dslr_cam_in_tool = R_dslr_cam_in_tool.as_quat()
        
        # Round the results to 5 decimals for neatness
        t_dslr_cam_in_tool = np.around(t_dslr_cam_in_tool, 5).tolist()
        quat_dslr_cam_in_tool = np.around(quat_dslr_cam_in_tool, 5).tolist()
        
        # Save in dictionary for JSON output
        centered_poses[cam_name] = {
            "t": t_dslr_cam_in_tool,
            "quat": quat_dslr_cam_in_tool
        }
        
        # Create a formatted output line for console printing
        line = (
            f'PoseStruct("{cam_name}", 1, np.array({t_dslr_cam_in_tool}), '
            f'R.from_quat({quat_dslr_cam_in_tool}))'
        )
        pose_lines.append(line)
    
    return pose_lines, centered_poses

def write_centered_poses(output_filename, centered_poses):
    with open(output_filename, 'w') as f:
        json.dump(centered_poses, f, indent=4)

def main():
    data = load_poses(json_filename)
    pose_lines, centered_poses = compute_relative_poses(data)

    print("Centered")
    
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
