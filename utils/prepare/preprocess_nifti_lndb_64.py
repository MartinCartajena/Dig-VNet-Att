import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import label
from tqdm import tqdm

def process_nifti_images(images_dir, labels_dir, output_images_dir, output_labels_dir):
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    image_files = {f: os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii') or f.endswith('.nii.gz')}
    label_files = {f: os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii') or f.endswith('.nii.gz')}

    for file_name in tqdm(label_files.keys(), desc="Processing files"):
        if file_name not in image_files:
            print(f"Warning: Label file {file_name} does not have a corresponding image file.")
            continue

        # Load image and label
        label_path = label_files[file_name]
        image_path = image_files[file_name]
        label_nifti = nib.load(label_path)
        image_nifti = nib.load(image_path)
        
        label_data = label_nifti.get_fdata()
        image_data = image_nifti.get_fdata()

        # Identify connected components in the label
        labeled_array, num_features = label(label_data)

        for i in range(1, num_features + 1):
            # Extract the current segmentation
            current_segmentation = (labeled_array == i)

            # Find bounding box around the segmentation
            coords = np.array(np.nonzero(current_segmentation))
            min_coords = coords.min(axis=1)
            max_coords = coords.max(axis=1) + 1

            # Crop image and label
            cropped_label = label_data[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
            cropped_image = image_data[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

            # Resize to 64x64x16
            resized_label = resize(cropped_label, (64, 64, 16), order=0, preserve_range=True, anti_aliasing=False)
            resized_image = resize(cropped_image, (64, 64, 16), order=1, preserve_range=True, anti_aliasing=True)

            # Transpose to 16x64x64
            transposed_label = np.transpose(resized_label, (2, 0, 1))
            transposed_image = np.transpose(resized_image, (2, 0, 1))

            # Save as .npy
            base_name = file_name.replace('.nii', '').replace('.gz', '')
            np.save(os.path.join(output_labels_dir, f"{base_name}_seg_{i}.npy"), transposed_label)
            np.save(os.path.join(output_images_dir, f"{base_name}_seg_{i}.npy"), transposed_image)

if __name__ == "__main__":
    images_directory = "/app/data/LNDb/solo_nodulos/images"
    labels_directory = "/app/data/LNDb/solo_nodulos/labels"
    output_images_directory = "/app/data/LNDb/voxels64/imagesTs/"
    output_labels_directory = "/app/data/LNDb/voxels64/labelsTs/"

    process_nifti_images(images_directory, labels_directory, output_images_directory, output_labels_directory)
