import numpy as np
import os

def add_salt_and_pepper_noise(data, amount=0.05, salt_ratio=0.5):
    """
    Añade ruido "salt and pepper" a un volumen 3D.
    
    Args:
        data (np.ndarray): Volumen 3D de entrada.
        amount (float): Proporción de elementos afectados por el ruido.
        salt_ratio (float): Proporción de píxeles "sal" (1s) respecto a "pimienta" (0s).
    
    Returns:
        np.ndarray: Volumen con ruido "salt and pepper".
    """
    noisy_data = data.copy()
    num_voxels = data.size
    num_salt = int(amount * num_voxels * salt_ratio)
    num_pepper = int(amount * num_voxels * (1 - salt_ratio))
    
    # Coordenadas aleatorias para "sal" (1s)
    salt_coords = [np.random.randint(0, i, num_salt) for i in data.shape]
    noisy_data[tuple(salt_coords)] = 1
    
    # Coordenadas aleatorias para "pimienta" (0s)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in data.shape]
    noisy_data[tuple(pepper_coords)] = 0
    
    return noisy_data

def save_transformed(data, output_dir, prefix, transform_name):
    """
    Guarda un volumen transformado en un archivo separado.
    
    Args:
        data (np.ndarray): Volumen transformado.
        output_dir (str): Directorio para guardar el archivo.
        prefix (str): Prefijo para el nombre del archivo.
        transform_name (str): Nombre de la transformación aplicada.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_{transform_name}.npy")
    np.save(output_path, data)
    print(f"Guardado: {output_path}")

def data_augmentation_with_reverse_and_masks(image_path, mask_path, image_output_dir, mask_output_dir, name_original, noise_amount=0.05, salt_ratio=0.5):
    """
    Aplica data augmentation a imágenes y sus máscaras asociadas incluyendo transformaciones y orden inverso.
    
    Args:
        image_path (str): Ruta al archivo .npy de imágenes.
        mask_path (str): Ruta al archivo .npy de máscaras.
        image_output_dir (str): Directorio para guardar las imágenes transformadas.
        mask_output_dir (str): Directorio para guardar las máscaras transformadas.
        noise_amount (float): Proporción de elementos afectados por el ruido "salt and pepper".
        salt_ratio (float): Proporción de píxeles "sal" respecto a "pimienta".
    """
    # Verificar que los archivos de entrada existen
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"El archivo de imágenes {image_path} no existe.")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"El archivo de máscaras {mask_path} no existe.")
    
    # Cargar los datos
    images = np.load(image_path)
    masks = np.load(mask_path)
    
    if images.shape != masks.shape:
        raise ValueError(f"Las dimensiones de las imágenes {images.shape} y máscaras {masks.shape} no coinciden.")
    
    # Orden inverso de cortes
    reversed_images = images[::-1, :, :]
    reversed_masks = masks[::-1, :, :]
    
    # Lista de transformaciones
    transformations = {
        "rotate180": lambda x: np.rot90(x, k=2, axes=(1, 2)),
        "transpose":  lambda x: np.flip(np.rot90(x, k=1, axes=(1, 2)), axis=0),
        "flip_y": lambda x: np.flip(x, axis=1),
        "flip_x": lambda y: np.flip(y, axis=2),
    }
    
    # Guardar la imagen y máscara original
    # save_transformed(images, image_output_dir, name_original, "original")
    # save_transformed(masks, mask_output_dir, name_original, "original")
    
    # Transformaciones para imagen y máscara originales
    for name, transform in transformations.items():
        transformed_image = transform(images)
        transformed_mask = transform(masks)
        
        save_transformed(transformed_image, image_output_dir, name_original, name)
        save_transformed(transformed_mask, mask_output_dir, name_original, name)
    
    # Transformaciones para imagen y máscara en orden inverso
    for name, transform in transformations.items():
        transformed_image = transform(reversed_images)
        transformed_mask = transform(reversed_masks)
        
        save_transformed(transformed_image, image_output_dir, name_original, f"reversed_{name}")
        save_transformed(transformed_mask, mask_output_dir, name_original, f"reversed_{name}")
    
    # Añadir ruido "salt and pepper" solo a las imágenes originales e invertidas
    noisy_image = add_salt_and_pepper_noise(images, amount=noise_amount, salt_ratio=salt_ratio)
    save_transformed(noisy_image, image_output_dir, name_original, "salt_and_pepper")
    save_transformed(masks, mask_output_dir, name_original, "salt_and_pepper")

    noisy_reversed_image = add_salt_and_pepper_noise(reversed_images, amount=noise_amount, salt_ratio=salt_ratio)
    save_transformed(noisy_reversed_image, image_output_dir, name_original, "reversed_salt_and_pepper")
    save_transformed(masks, mask_output_dir, name_original, "reversed_salt_and_pepper")


if __name__ == "__main__":
    image_dir = r"/media/data/mcartajena/LUCIA/Data/LUNA16/Task_LUNA16_test/imagesTs/"
    mask_dir = r"/media/data/mcartajena/LUCIA/Data/LUNA16/Task_LUNA16_test/labelsTs/"
    image_output_dir = r"/media/data/mcartajena/LUCIA/Data/LUNA16/Task_LUNA16_test/imagesAugTs/"
    mask_output_dir = r"/media/data/mcartajena/LUCIA/Data/LUNA16/Task_LUNA16_test/labelsAugTs/"
    
    for name in os.listdir(image_dir):
        image_file = os.path.join(image_dir, name)
        mask_file = os.path.join(mask_dir, name)
        data_augmentation_with_reverse_and_masks(image_file, mask_file, image_output_dir, mask_output_dir, name.replace(".npy",""), noise_amount=0.03, salt_ratio=0.5)
