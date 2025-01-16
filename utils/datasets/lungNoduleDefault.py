import os
import torch
from torch.utils.data import Dataset
import numpy as np

class LungNoduleDefault(Dataset):
    def __init__(
        self, 
        root_dir, 
        split='train', 
        transform=None,
        data_aug=None,
    ):
        """
        Inicializa el dataset.
        
        Parameters:
            root_dir (str): Directorio raíz que contiene las carpetas de imágenes y segmentaciones.
            split (str): Tipo de split ('train', 'val' o 'test') para elegir el subconjunto correspondiente.
            preprocess (bool): tipo de imagen, recorte o imagen con size original para segmentar 
            transform (callable, optional): Transformaciones a aplicar a las imágenes y/o etiquetas.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data_aug = data_aug

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
        
        assert len(self.images) == len(self.labels), "Número de imágenes y etiquetas no coincide."
    
        
    def __len__(self):
        return len(self.images)
        
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        
        image = np.load(image_path)
        label = np.load(label_path)
        
        if self.transform:
            image = image.cpu().numpy()
            label = label.cpu().numpy()
            image, label = self.transform((image, label))
            
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).long() 
            
        return image, label

