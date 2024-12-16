import numpy as np

class DataAugmentation:
    def __init__(self, noise_amount=0.05, salt_ratio=0.5, orientation = "RAS"):
        """
        Inicializa la clase de aumentación de datos.
        
        Args:
            noise_amount (float): Proporción de elementos afectados por el ruido "salt and pepper".
            salt_ratio (float): Proporción de píxeles "sal" respecto a "pimienta".
        """
        self.noise_amount = noise_amount
        self.salt_ratio = salt_ratio
        self.orientation = orientation
        
        if orientation == "RAS":
            self.transformations = {
                "rotate180": lambda x: np.rot90(x, k=1, axes=(0, 1)),
                "transpose": lambda x: np.flip(np.rot90(x, k=1, axes=(0, 1)), axis=2),
                "flip_y": lambda x: np.flip(x, axis=1),
                "flip_x": lambda x: np.flip(x, axis=0),
            }
        else:
            self.transformations = {
                "rotate180": lambda x: np.rot90(x, k=2, axes=(1, 2)),
                "transpose": lambda x: np.flip(np.rot90(x, k=2, axes=(1, 2)), axis=0),
                "flip_y": lambda x: np.flip(x, axis=2),
                "flip_x": lambda x: np.flip(x, axis=1),
            }

    def add_salt_and_pepper_noise(self, data):
        """
        Añade ruido "salt and pepper" a un volumen 3D.
        
        Args:
            data (np.ndarray): Volumen 3D de entrada.
        
        Returns:
            np.ndarray: Volumen con ruido "salt and pepper".
        """
        noisy_data = data.copy()
        num_voxels = data.size
        num_salt = int(self.noise_amount * num_voxels * self.salt_ratio)
        num_pepper = int(self.noise_amount * num_voxels * (1 - self.salt_ratio))

        # Coordenadas aleatorias para "sal" (1s)
        salt_coords = [np.random.randint(0, i, num_salt) for i in data.shape]
        noisy_data[tuple(salt_coords)] = 1

        # Coordenadas aleatorias para "pimienta" (0s)
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in data.shape]
        noisy_data[tuple(pepper_coords)] = 0

        return noisy_data

    def __call__(self, images, masks):
        """
        Aplica todas las transformaciones a una imagen y su máscara.
        
        Args:
            images (np.ndarray): Volumen 3D de imágenes.
            masks (np.ndarray): Volumen 3D de máscaras.
        
        Returns:
            dict: Diccionario con todas las imágenes transformadas y sus nombres.
        """
        if images.shape != masks.shape:
            raise ValueError(f"Las dimensiones de las imágenes {images.shape} y máscaras {masks.shape} no coinciden.")

        augmented_images = {}
        augmented_labels = {}

        # Orden inverso
        if self.orientation == "RAS":
            reversed_images = images[:, :, ::-1]
            reversed_masks = masks[:, :, ::-1]
        else:   
            reversed_images = images[::-1, :, :]
            reversed_masks = masks[::-1, :, :]

        # Transformaciones sobre las imágenes originales y máscaras
        for name, transform in self.transformations.items():
            augmented_images[f"{name}_image"] = transform(images)
            augmented_labels[f"{name}_mask"] = transform(masks)

        # Transformaciones sobre las imágenes y máscaras invertidas
        for name, transform in self.transformations.items():
            augmented_images[f"reversed_{name}_image"] = transform(reversed_images)
            augmented_labels[f"reversed_{name}_mask"] = transform(reversed_masks)

        # Ruido "salt and pepper" en imágenes originales e invertidas
        augmented_images["salt_and_pepper_image"] = self.add_salt_and_pepper_noise(images)
        augmented_labels["salt_and_pepper_mask"] = masks  # Las máscaras no cambian
        augmented_images["reversed_salt_and_pepper_image"] = self.add_salt_and_pepper_noise(reversed_images)
        augmented_labels["reversed_salt_and_pepper_mask"] = reversed_masks  # Las máscaras no cambian

        return augmented_images, augmented_labels