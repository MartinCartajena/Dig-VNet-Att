import os
import json
import numpy as np
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from skimage.transform import resize
from totalsegmentator.python_api import totalsegmentator
import tempfile

# 1. Function to convert DICOM to volumes
def dicom_to_volumes(dicom_dir):
    volumes = {}

    for root, _, files in os.walk(dicom_dir):
        slices = []
        slice_ids = []
        base_name = os.path.basename(root)
        for file in sorted(files):
            file_path = os.path.join(root, file)
            try:
                dicom = pydicom.dcmread(file_path)
                image = apply_modality_lut(dicom.pixel_array, dicom)
                slices.append(image)
                slice_ids.append(dicom.SOPInstanceUID)
            except Exception as e:
                print(f"Skipping file {file_path}: {e}")

        if slices:
            volume = np.stack(slices, axis=-1)
            volumes[base_name] = {
                "volume": volume,
                "slice_ids": slice_ids
            }

    return volumes

# 2. Placeholder for lung segmentation (to be implemented by the user)
def segment_lungs(volume):
    # Placeholder: Return the segmented lung mask
    raise NotImplementedError("Please implement the lung segmentation logic.")

# 3. Intensity normalization (-1 to 1)
def normalize_intensity(volume):
    volume = np.clip(volume, -1000, 400)  # Clip Hounsfield units to [-1000, 400]
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))  # Normalize to [0, 1]
    volume = 2 * volume - 1  # Scale to [-1, 1]
    return volume

# 4. Extract patches from the nodule coordinates
def extract_patches(volume, slice_ids, nodule_data, patch_size=(96, 96, 16)):
    patches = []
    half_sizes = [dim // 2 for dim in patch_size]

    for nodule in nodule_data:
        xmin, xmax = nodule["xmin"], nodule["xmax"]
        ymin, ymax = nodule["ymin"], nodule["ymax"]
        slices = nodule["slices"]

        # Map slice IDs to indices
        slice_indices = [slice_ids.index(s) for s in slices if s in slice_ids]
        if not slice_indices:
            continue

        z_min, z_max = min(slice_indices), max(slice_indices)
        z_center = (z_min + z_max) // 2
        y_center = (ymin + ymax) // 2
        x_center = (xmin + xmax) // 2

        z_start, z_end = z_center - half_sizes[2], z_center + half_sizes[2]
        y_start, y_end = y_center - half_sizes[1], y_center + half_sizes[1]
        x_start, x_end = x_center - half_sizes[0], x_center + half_sizes[0]

        patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
        if patch.shape == tuple(patch_size):
            patches.append(patch)

    return patches

def segment_lungs(image):
    """
    Segmenta los pulmones utilizando TotalSegmentator.
    
    Parameters:
        image (np.ndarray): Imagen original en formato 3D.
    
    Returns:
        lung_mask (np.ndarray): Máscara binaria de los pulmones.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        temp_input_path = os.path.join(tmp_dir, "image.nii.gz")
        temp_output_dir = os.path.join(tmp_dir, "output")
        img_nifti = nib.Nifti1Image(image, affine=np.eye(4))
            
        nib.save(img_nifti, temp_input_path) 
        
        lung_parts = [
                'lung_upper_lobe_left', 
                'lung_upper_lobe_right', 
                'lung_lower_lobe_left', 
                'lung_lower_lobe_right', 
                'lung_middle_lobe_right'
                ]
        # Ejecuta la segmentación sin logs de terminal
        totalsegmentator(
            temp_input_path, 
            temp_output_dir, 
            roi_subset= lung_parts
        )
         
        lungs = []
        for part in lung_parts:
            rigth_lung_mask = os.path.join(temp_output_dir, part + ".nii.gz")
            lungs.append(nib.load(rigth_lung_mask).get_fdata())
            
    return lungs

# Main function to process the pipeline
def process_dicom_directory(dicom_dir, json_path, output_dir):
    # Step 1: Convert DICOM to volumes
    volumes = dicom_to_volumes(dicom_dir)

    # Load JSON data
    with open(json_path, 'r') as f:
        nodule_data = json.load(f)

    all_patches = {}

    for base_name, data in volumes.items():
        if base_name not in nodule_data:
            print(f"Skipping {base_name}: No corresponding data in JSON.")
            continue

        volume = data["volume"]
        slice_ids = data["slice_ids"]

        # Step 2: Segment lungs (placeholder)
        lungs = segment_lungs(volume)
        
        segmented_mask = np.zeros_like(volume)

        for mask in lungs:
            segmented_mask += volume * mask

        # Step 3: Normalize intensity
        normalized_volume = normalize_intensity(segmented_mask)

        # Step 4: Extract patches based on JSON coordinates
        nodules = nodule_data.get(base_name, [])
        patches = extract_patches(normalized_volume, slice_ids, nodules)
        all_patches[base_name] = patches

    # Save patches as .npy files
    os.makedirs(output_dir, exist_ok=True)
    for base_name, patches in all_patches.items():
        base_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(base_output_dir, exist_ok=True)
        for i, patch in enumerate(patches):
            patch_path = os.path.join(base_output_dir, f"patch_{i}.npy")
            np.save(patch_path, patch)

if __name__ == "__main__":
    dicom_directory = "/app/data/NLST/dicom_images/"
    json_file = "/app/data/NLST/nlst_3D_bb_no_instance_number.json"
    output_directory = "path/to/output_directory"

    process_dicom_directory(dicom_directory, json_file, output_directory)
