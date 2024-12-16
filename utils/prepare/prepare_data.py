import os
import csv
import numpy as np
import nibabel as nib
import pandas as pd

def read_csv(csv_path):
    """Lee el archivo CSV y devuelve los datos como un DataFrame."""
    return pd.read_csv(csv_path)

def load_nifti(file_path):
    """Carga un archivo .nii y devuelve la imagen, su affine, el espaciado y el origen."""
    nifti_image = nib.load(file_path)
    array = nifti_image.get_fdata()
    affine = nifti_image.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # Resolución espacial
    origin = affine[:3, 3]  # Coordenadas de origen
    return array, affine, spacing, origin

def create_nifti_from_segmentations(csv_path, segmentations_dir, output_dir):
    """Convierte las segmentaciones .nii en archivos NIfTI usando las anotaciones del CSV."""
    # Leer anotaciones del CSV
    nodules = read_csv(csv_path)

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Procesar cada CT según el CSV
    ct_ids = nodules['LNDbID'].unique()

    for ct_id in ct_ids:
        # Filtrar nódulos para este CT
        ct_nodules = nodules[nodules['LNDbID'] == ct_id]

        exist = False
        for _, nodule in ct_nodules.iterrows():
            rads = nodule['RadID'].split(',')
            segmentation_files = [
                f for f in os.listdir(segmentations_dir)
                if f.startswith(f"LNDb-{str(ct_id).zfill(4)}_rad{str(rads[0])}") and f.endswith(".nii.gz")
            ]

            if 1 < len(rads) and nodule['Nodule'] == 1:
                exist = True

        if exist:
            for _, nodule in ct_nodules.iterrows():
                rads = nodule['RadID'].split(',')

                if 1 < len(rads) and nodule['Nodule'] == 1:
                    segmentation_files = [
                        f for f in os.listdir(segmentations_dir)
                        if f.startswith(f"LNDb-{str(ct_id).zfill(4)}_rad{str(rads[0])}") and f.endswith(".nii.gz")
                    ]
                    if not segmentation_files:
                        print(f"No se encontraron segmentaciones para CT {ct_id}")
                        continue

                    # Cargar la segmentación del primer radiólogo como plantilla
                    array, affine, spacing, origin = load_nifti(os.path.join(segmentations_dir, segmentation_files[0]))
                    combined_volume = np.zeros_like(array, dtype=np.uint8)

            # Añadir las segmentaciones de los nódulos indicados en el CSV
            for _, nodule in ct_nodules.iterrows():
                rads = nodule['RadID'].split(',')

                if 1 < len(rads) and nodule['Nodule'] == 1:
                    x, y, z = nodule['x'], nodule['y'], nodule['z']
                    world_coords = np.array([x, -y, z])

                    # # Convertir coordenadas del mundo a índices en la matriz usando affine
                    # voxel_coords = nib.affines.apply_affine(np.linalg.inv(affine), world_coords)
                    # voxel_coords = np.round(voxel_coords).astype(int)
                    

                    # Convertir coordenadas del mundo a índices en la matriz
                    voxel_coords = np.round((world_coords - np.array(origin)) / np.array(spacing)).astype(int)

                    x_idx, y_idx, z_idx = voxel_coords

                    if (0 <= x_idx < array.shape[0] and
                        0 <= y_idx < array.shape[1] and
                        0 <= z_idx < array.shape[2]):
                        # Identificar los píxeles del nódulo segmentado
                        nodule_mask = (array == 1)  # Asume que el nódulo tiene valores de 1 en el array
                        coords = np.argwhere(nodule_mask)
                        for coord in coords:
                            xx, yy, zz = coord
                            dx, dy, dz = xx - x_idx, yy - y_idx, zz - z_idx
                            if (0 <= x_idx + dx < combined_volume.shape[0] and
                                0 <= y_idx + dy < combined_volume.shape[1] and
                                0 <= z_idx + dz < combined_volume.shape[2]):
                                combined_volume[x_idx + dx, y_idx + dy, z_idx + dz] = 1

            # Crear el archivo NIfTI
            nifti_image = nib.Nifti1Image(combined_volume, affine)

            output_path = os.path.join(output_dir, f"LNDb-{str(ct_id).zfill(4)}.nii.gz")
            nib.save(nifti_image, output_path)
            print(f"Archivo NIfTI guardado: {output_path}")

        else:
            array, affine, spacing, origin = load_nifti(os.path.join(segmentations_dir, segmentation_files[0]))
            combined_volume = np.zeros_like(array, dtype=np.uint8)

            nifti_image = nib.Nifti1Image(combined_volume, affine)

            output_path = os.path.join(output_dir, f"LNDb-{str(ct_id).zfill(4)}.nii.gz")
            nib.save(nifti_image, output_path)
            print(f"Archivo NIfTI de ceros: {output_path}")

# Ejecución del programa
csv_path = "/app/data/LNDb/trainNodules_gt.csv"
segmentations_dir = "/app/data/LNDb/nifti_segs/"  # Reemplazar con el directorio de segmentaciones
output_dir = "/app/data/LNDb/segmentations_gt"  # Reemplazar con el directorio de salida

create_nifti_from_segmentations(csv_path, segmentations_dir, output_dir)
