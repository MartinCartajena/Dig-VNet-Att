import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

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
            image_dir = os.path.join(root_dir, 'imagesTr')
            label_dir = os.path.join(root_dir, 'labelsTr')
        elif split == 'val':
            image_dir = os.path.join(root_dir, 'imagesVal')
            label_dir = os.path.join(root_dir, 'labelsVal')
        elif split == 'test':
            image_dir = os.path.join(root_dir, 'imagesTs')
            label_dir = os.path.join(root_dir, 'labelsTs')
        else:
            raise ValueError("El parámetro split debe ser 'train', 'val' o 'test'.")

        self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.labels = sorted([os.path.join(image_dir, f) for f in os.listdir(label_dir)])

        assert len(self.images) == len(self.labels), "Número de imágenes y etiquetas no coincide."
    
    
    def __len__(self):
        """
        Retorna el número de muestras en el dataset.
        """
        return len(self.images)
    
    
    def __getitem__(self, idx):
        """
        Retorna una muestra del dataset.
        
        Parameters:
            idx (int): Índice de la muestra a retornar.
        
        Returns:
            image (torch.Tensor): Imagen procesada.
            label (torch.Tensor): Etiqueta (segmentación) correspondiente a la imagen.
        """
        if self.images[idx].endswith(".npy"):
            image = np.load(self.images[idx])
            label = np.load(self.labels[idx])
  
        elif self.images[idx].endswith(".nii.gz"):
            image = nib.load(self.images[idx]).get_fdata()
            label = nib.load(self.labels[idx]).get_fdata()

        if self.transform:
            image, label = self.transform(image, label)
                
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long() 

        return image, label
