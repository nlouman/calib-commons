import os
import shutil
import re

def sort_and_rename_images(folder_path, overwrite=False):
    """
    Sorts all image files in the given folder and renames them to 1.jpg, 2.jpg, etc.
    If the images are PNGs, the new names will be 1.png, 2.png, etc. The renaming is based on
    alphabetical order of the file names.
    
    Parameters:
        folder_path (str): Path to the folder containing the images.
        overwrite (bool): 
            If True, images in the original folder will be renamed in place.
            If False, a new folder named "<original_folder>_structured" will be created
            in the same parent directory, and the renamed images will be copied there.
            
    Returns:
        str: The path to the folder containing the renamed images.
    """
    # Validate folder existence
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")
    
    # List all files in the folder and filter for image files (case insensitive)
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

    def natural_key(string):
        # Splits the string into a list of strings and numbers for natural sorting.
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]

    # Sort image files alphabetically
    image_files.sort(key=natural_key)

    # Determine destination folder based on the overwrite flag
    if not overwrite:
        parent_dir, folder_name = os.path.split(os.path.abspath(folder_path))
        new_folder_name = folder_name + "_structured"
        new_folder_path = os.path.join(parent_dir, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
    else:
        new_folder_path = folder_path

    # Rename or copy files with sequential numbering
    for idx, filename in enumerate(image_files, start=1):
        # Get the file extension in lowercase (e.g., '.jpg' or '.png')
        ext = os.path.splitext(filename)[1].lower()
        new_filename = f"{idx}{ext}"
        src_path = os.path.join(folder_path, filename)
        dst_path = os.path.join(new_folder_path, new_filename)
        if overwrite:
            os.rename(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    return new_folder_path

if __name__ == "__main__":
    folder = sort_and_rename_images('/home/fred/dev/orx/orx_middleware/ros2_ws/src/kuka_camera/src/user_applications/calibration_captures/calibration_019/dslr', overwrite=False)
    print("Renamed images are in:", folder)

# Example usage:
# folder = sort_and_rename_images('/path/to/your/images', overwrite=False)
# print("Renamed images are in:", folder)
