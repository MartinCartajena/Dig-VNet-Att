import os
import json


def move_files_to_directory(input_directory, output_directory):
    """
    Moves all files from subdirectories within the input directory to the output directory.

    Args:
        input_directory (str): Path to the input directory containing subdirectories with files.
        output_directory (str): Path to the output directory where files will be moved.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for root, _, files in os.walk(input_directory):
        if root == input_directory:
            continue  # Skip the root input directory itself

        for file_name in files:
            source_path = os.path.join(root, file_name)
            destination_path = os.path.join(output_directory, file_name)

            # Move the file
            os.rename(source_path, destination_path)


input_directory_path = "/app/data/LNDb/nifti_images/"  # Replace with your input directory path
output_directory_path = "/app/data/LNDb/nifti_img/"  # Replace with your output directory path
move_files_to_directory(input_directory_path, output_directory_path)
