import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch.nn.functional as F

class LungNoduleSegmentationDataset_chino(Dataset):
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
        self.labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        
        """ Transfomracion del paper chino de Aritz"""
        self.normalize_gray = Normalize(mean=[0.448749], std=[0.399953])

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
            
            image = torch.from_numpy(image).float()  
            label = torch.from_numpy(label).float() 
           
            image = image.unsqueeze(0).unsqueeze(0)  # Dimensiones (1, 1, H, W, Z)
            label = label.unsqueeze(0).unsqueeze(0)  # Dimensiones (1, 1, H, W, Z)
            
            target_size = (16, 64, 64)
            image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=False)
            label = F.interpolate(label, size=target_size, mode='nearest')  # Para segmentación, nearest es mejor
            
            image = image.squeeze(0).squeeze(0)  # Dimensiones (16, 64, 64)
            label = label.squeeze(0).squeeze(0)  # Dimensiones (16, 64, 64)
            
            # normalizar de los chinos que dice que ayuda a mejorar el dice
            image = self.normalize_gray(image)
            label = self.normalize_gray(label)

            # binarizar
            label[label > 0.5] = 1
            label[label < 0.5] = 0

        if self.transform:
            image, label = self.transform((image, label))
                
        return image, label
