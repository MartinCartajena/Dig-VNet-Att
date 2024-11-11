
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class LungNoduleSegmentationDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        split='train', 
        transform=None
    ):
        """
        Inicializa el dataset.
        
        Parameters:
            root_dir (str): Directorio raíz que contiene las carpetas de imágenes y segmentaciones.
            split (str): Tipo de split ('train', 'val' o 'test') para elegir el subconjunto correspondiente.
            transform (callable, optional): Transformaciones a aplicar a las imágenes y/o etiquetas.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'imagesTr')
            self.label_dir = os.path.join(root_dir, 'labelsTr')
        elif split == 'val':
            self.image_dir = os.path.join(root_dir, 'imagesVal')
            self.label_dir = os.path.join(root_dir, 'labelsVal')
        elif split == 'test':
            self.image_dir = os.path.join(root_dir, 'imagesTs')
            self.label_dir = os.path.join(root_dir, 'labelsTs')
        else:
            raise ValueError("El parámetro split debe ser 'train', 'val' o 'test'.")

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.npy')])

        assert len(self.image_files) == len(self.label_files), "Número de imágenes y etiquetas no coincide."
    
    def __len__(self):
        """
        Retorna el número de muestras en el dataset.
        """
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Retorna una muestra del dataset.
        
        Parameters:
            idx (int): Índice de la muestra a retornar.
        
        Returns:
            image (torch.Tensor): Imagen procesada.
            label (torch.Tensor): Etiqueta (segmentación) correspondiente a la imagen.
        """
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = np.load(image_path)
        label = np.load(label_path)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long() 

        if self.transform:
            image, label = self.transform(image, label)

        return image, label
