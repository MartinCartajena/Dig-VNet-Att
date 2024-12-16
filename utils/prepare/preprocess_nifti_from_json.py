import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
from totalsegmentator.python_api import totalsegmentator
import tempfile
from scipy.ndimage import label, center_of_mass
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


def remove_noise_using_start_of_other_lung(lung_image, threshold=5, neighborhood_size=3):

    values_x = np.max(lung_image, axis=(1, 2))
            
    agrupations = []
    group = []
    desv = 0
    in_group = False
    
    for i in range(len(values_x)):
        if values_x[i] != 0:  
            group.append(i) 
            in_group = True
            
        else:
            if in_group:
                desv += 1
                if desv > 40:
                    in_group = False
                    if len(group) > 0:
                        agrupations.append(group)
                    group = []
                    desv = 0
                        
    for groups in agrupations:
        if len(groups) < 10:
            for i in groups:
                lung_image[i,:,:] = 0
                
    return lung_image

def verify_mask(original_image, segmented_image):
    mask = segmented_image > 0
    original_values = original_image[mask]
    segmented_values = segmented_image[mask]
    if not np.array_equal(original_values, segmented_values):
        segmented_image[mask] = original_image[mask]
    return segmented_image


def extract_patches(ct_image, label_image, patch_size=(96, 96, 16)):
    """
    Extrae recortes de la imagen CT y del label en torno a los nódulos etiquetados.
    
    :param ct_image: numpy array de la imagen CT.
    :param label_image: numpy array del label con los nódulos segmentados.
    :param patch_size: dimensiones del recorte (x, y, z).
    :return: lista de patches de CT y labels.
    """
    # Etiquetar componentes conectados en el label
    labeled_array, num_features = label(label_image > 0)
    print(f"Se detectaron {num_features} componentes conectados.")
    
    patches_ct = []
    patches_label = []
    half_size = np.array(patch_size) // 2
    
    for i in range(1, num_features + 1):  
        
        component_mask = (labeled_array == i)
        
        # Calcular el centroide
        coords = np.round(center_of_mass(component_mask)).astype(int)
        
        start = coords - half_size
        end = coords + half_size
        
        start = np.maximum(start, 0)
        end = np.minimum(end, label_image.shape)
        
        patch_ct = ct_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        patch_label = label_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        if patch_ct.shape == tuple(patch_size) and patch_label.shape == tuple(patch_size):
            patches_ct.append(patch_ct)
            patches_label.append(patch_label)
        else:
            print(f"El nódulo en {coords} tiene un recorte incompleto y se descarta.")
    
    return patches_ct, patches_label

    
def preprocess(image, label, affine, target_size=(96, 96, 16)):
    """
    Preprocesa una imagen:
    - Redimensiona al tamaño objetivo.
    - Normaliza los valores de píxel.
    """
    
    """ SEGMENTAR PULMONES """
    # image = orient_RAS(image, affine)
    # label = orient_RAS(label, affine)
        
    lungs = _segment_lungs(image)
       
    lung_only_image = np.zeros_like(image)

    for mask in lungs:
        lung_only_image += image * mask
        
    lung_only_image = remove_noise_using_start_of_other_lung(lung_only_image)

    lung_only_image = verify_mask(image, lung_only_image)
    
    """ CENTRO DEL NODULO Y RECORTAR POR 96x96x16 """
    patches_ct, patches_label = extract_patches(lung_only_image, label, target_size)
    
    """ ACOTAR INTENSIDADES DE LOS PÍXELES """
    patches_ct = [clip_pixel_intensities(patch) for patch in patches_ct]
    
    
    """ TRANSPOSE Y NORMALIZAR INTENSIDADES """
    patches_ct = [np.transpose(patch, axes=[2, 0, 1]) for patch in patches_ct]
    patches_label = [np.transpose(patch_label, axes=[2, 0, 1]) for patch_label in patches_label]
    
    patches_ct = [normalized_values(pacth) for pacth in patches_ct]
    # patches_label = [(2 * (pacth - np.min(pacth)) / (np.max(pacth) - np.min(pacth) - 1)) for pacth in patches_label]
    
    return  patches_ct, patches_label


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

def preprocess_images(input_dir, output_dir, target_size=(96, 96, 16)):
    """
    Procesa todas las imágenes en un directorio y guarda los resultados.
    
    Args:
        input_dir (str): Directorio de entrada con imágenes.
        output_dir (str): Directorio donde se guardarán las imágenes procesadas.
        target_size (tuple): Tamaño objetivo para las imágenes procesadas.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(os.path.join(input_dir, "images")):
        input_path = os.path.join(input_dir, "images", file_name)
        
        try:
            image, affine = load_file(input_path)
            label, affine = load_file(os.path.join(input_dir, "labels", file_name))
            
            patches_ct, patches_label = preprocess(image, label, affine, target_size)
            
            
            for i, (ct_patch, label_patch) in enumerate(zip(patches_ct, patches_label)):
            
                output_path_image = os.path.join(output_dir, "imagesTs" , file_name.replace(".nii.gz",f"_{i}.npy"))
                output_path_label = os.path.join(output_dir, "labelsTs" , file_name.replace(".nii.gz",f"_{i}.npy"))
                
                np.save(output_path_image, ct_patch)
                np.save(output_path_label, label_patch) 
                
            print(f"Save {file_name}")
        
        except Exception as e:
            print(f"Error procesando {file_name}: {e}")


if __name__ == "__main__":
    input_dir = "/app/data/LNDb/solo_nodulos/"
    output_dir = "/app/data/LNDb/voxels/"
    
    preprocess_images(input_dir, output_dir, target_size=(96, 96, 16))
