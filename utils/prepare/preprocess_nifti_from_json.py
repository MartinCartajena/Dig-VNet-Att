import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
from totalsegmentator.python_api import totalsegmentator
import tempfile
from scipy.ndimage import label, center_of_mass, binary_dilation, binary_erosion, binary_fill_holes
import json
from utils.prepare.dig_module import BitwiseImageTransformer


# import matplotlib

# matplotlib.use("WebApp")

def load_file(path):
    """Carga un archivo dependiendo de su extensión."""
    if path.endswith(".npy"):
        return np.load(path), None
    elif path.endswith(".nii.gz"):
        img = nib.load(path)
        return img.get_fdata(), img.affine
    elif path.endswith(".mhd"):
        image = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(image), None
    else:
        raise ValueError(f"Formato de archivo no soportado: {path}")


def save_file(data, affine, output_path):
    """Guarda los datos en formato NIfTI."""
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)


def clip_pixel_intensities(image, min_value=-1000, max_value=400):
    """
    Acota los valores de intensidad de los píxeles a un rango específico.
    
    Parameters:
        image (np.ndarray): Imagen a procesar.
        min_value (float): Valor mínimo permitido.
        max_value (float): Valor máximo permitido.
    
    Returns:
        np.ndarray: Imagen con intensidades acotadas.
    """
    return np.clip(image, min_value, max_value)


def verify_mask(original_image, segmented_image):
    mask = segmented_image > 0
    original_values = original_image[mask]
    segmented_values = segmented_image[mask]
    if not np.array_equal(original_values, segmented_values):
        segmented_image[mask] = original_image[mask]
    return segmented_image


def extract_patches(file_name, ct_image, bboxes_json, patch_size=(96, 96, 16)):
    """
    Extrae recortes centrados en el centroide de cada bounding box especificado en bboxes_json.

    Args:
        file_name (str): Nombre del archivo de la imagen.
        ct_image (numpy.ndarray): Imagen 3D (CT scan).
        bboxes_json (dict): Diccionario que contiene los bounding boxes por archivo.
        patch_size (tuple): Tamaño del parche a extraer (profundidad, altura, anchura).

    Returns:
        list: Lista de recortes centrados en los centroides de los bounding boxes.
    """
    bboxes = bboxes_json[file_name.replace(".nii.gz", "")]
    
    bboxes_list = []
    patch_x, patch_y, patch_z = patch_size

    for bbox in bboxes:
        # Coordenadas del bounding box
        x1, x2 = bbox['xmin'], bbox['xmax']
        y1, y2 = bbox['ymin'], bbox['ymax']
        z1, z2 = bbox['zmin'], bbox['zmax']

        # Calcular el centroide del bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_z = (z1 + z2) // 2

        # Calcular los límites del recorte centrado en el centroide
        start_x = max(center_x - patch_x // 2, 0)
        end_x = start_x + patch_x

        start_y = max(center_y - patch_y // 2, 0)
        end_y = start_y + patch_y

        start_z = max(center_z - patch_z // 2, 0)
        end_z = start_z + patch_z

        # Ajustar los límites para que no excedan los límites de la imagen
        end_x = min(end_x, ct_image.shape[0])
        start_x = end_x - patch_x

        end_y = min(end_y, ct_image.shape[1])
        start_y = end_y - patch_y

        end_z = min(end_z, ct_image.shape[2])
        start_z = end_z - patch_z

        # Extraer el parche
        patch_ct = ct_image[start_x:end_x, start_y:end_y, start_z:end_z]

        bboxes_list.append(patch_ct)

    return bboxes_list

    
def preprocess(file_name, image, affine, bboxes_json, target_size=(96, 96, 16)):
    """
    Preprocesa una imagen:
    - Redimensiona al tamaño objetivo.
    - Normaliza los valores de píxel.
    """
    
    """ SEGMENTAR PULMONES """
    # image = orient_RAS(image, affine)
    # label = orient_RAS(label, affine)
    
    # dig_module = BitwiseImageTransformer(image)    
    # dig_x = dig_module.transform()
    
    lungs = _segment_lungs(image)
           
    binary_lung = np.zeros_like(image)

    for mask in lungs:
        binary_lung += mask
            
    """ CLIPPEAR INTENSIDADES """
    image = clip_pixel_intensities(image) 
    binary_lung = clip_pixel_intensities(binary_lung) 
    
    """ RELLENAR AGUJEROS """
    dilated_mask = binary_dilation(binary_lung, iterations=8)
    dilated_mask = binary_fill_holes(dilated_mask)
    eroded_mask = binary_erosion(dilated_mask, iterations=8)
    
    lung_only_image = image * eroded_mask
    
    lung_only_image = verify_mask(image, lung_only_image)

    """ CENTRO DEL NODULO Y RECORTAR POR 96x96x16 """
    patches_ct = extract_patches(file_name, lung_only_image, bboxes_json, target_size)
    
    """ ACOTAR INTENSIDADES DE LOS PÍXELES """
    patches_ct = [clip_pixel_intensities(patch) for patch in patches_ct]
    
    """ TRANSPOSE Y NORMALIZAR INTENSIDADES """
    patches_ct = [np.transpose(patch, axes=[2, 0, 1]) for patch in patches_ct]
    # patches_label = [np.transpose(patch_label, axes=[2, 0, 1]) for patch_label in patches_label]
    
    patches_ct = [normalized_values(pacth) for pacth in patches_ct]
    # patches_label = [(2 * (pacth - np.min(pacth)) / (np.max(pacth) - np.min(pacth) - 1)) for pacth in patches_label]
    
    return patches_ct


def normalized_values(image):
        min_val = np.min(image)
        max_val = np.max(image)
        range_val = max_val - min_val

        if range_val == 0:
            # Si todos los valores son iguales, normalizamos a un array de ceros
            return np.zeros_like(image)

        # Fórmula de normalización entre -1 y 1
        return 2 * (image - min_val) / range_val - 1

        
def _segment_lungs(image):
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


def load_json(file_path):
    """
    Load a JSON file and return its contents.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The content of the JSON file as a Python dictionary or list.
        None: If the file doesn't exist or contains invalid JSON.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON - {e}")
    
    return None


def preprocess_images(input_dir, json_path, output_dir, target_size=(96, 96, 16)):
    """
    Procesa todas las imágenes en un directorio y guarda los resultados.
    
    Args:
        input_dir (str): Directorio de entrada con imágenes.
        output_dir (str): Directorio donde se guardarán las imágenes procesadas.
        target_size (tuple): Tamaño objetivo para las imágenes procesadas.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    bboxes_json = load_json(json_path)

    for file_name in os.listdir(os.path.join(input_dir)):
        
        input_path = os.path.join(input_dir, file_name)
        
        try:
            image, affine = load_file(input_path)
            
            patches_ct = preprocess(file_name, image, affine, bboxes_json, target_size)
            
            for i, ct_patch in enumerate(zip(patches_ct)):
            
                output_path_image = os.path.join(output_dir, file_name.replace(".nii.gz",f"_{i}.npy"))
                
                np.save(output_path_image, ct_patch)
                
            print(f"Save {file_name}")
        
        except Exception as e:
            print(f"Error procesando {file_name}: {e}")


if __name__ == "__main__":
    input_dir = "/app/data/NLST/nifti_images/"
    output_dir = "/app/data/NLST/voxels/imagesTs/"
    json_path = "/app/data/NLST/nlst_3D_bboxes.json"
    
    preprocess_images(input_dir, json_path, output_dir, target_size=(96, 96, 16))
