import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import label as label_ndimage
from scipy.ndimage import uniform_filter
from scipy.ndimage import zoom


from totalsegmentator.python_api import totalsegmentator


# from TotalSegmentator import TotalSegmentator
# from TotalSegmentator.libs import nostdout

import tempfile

class LungNoduleSegmentationDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        split='train', 
        preprocess=False,
        transform=None,
        data_aug=None,
        use_cache=False,
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
        self.preprocess = preprocess
        self.transform = transform
        self.data_aug = data_aug
        
        self.cache_images= []
        self.cache_labels= []
        self.use_cache = use_cache

        
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
        
        
    def setCache(self, use_cache):
        self.use_cache = use_cache
        
        
    def getCache(self):
        return self.cached_data
    
        
    def __len__(self):
        if self.preprocess: # de cada imagen segmentas y devuelves dos recortes de pulmon. Bueno habria que ver si se puede vivir con un pulmon... :P
            if self.data_aug:
                if self.use_cache: # la primera vuelta para cargar imagen en cache coge las originales
                    return len(self.images) * 22 # aqui ya tiene que coger las transformadas
                else:
                    return len(self.images)
            else:
                if self.use_cache:
                    return len(self.images) * 2
                else:
                    return len(self.images)
        else:
            return len(self.images)
        
    
    def _load_file(self, path):
        """Carga un archivo dependiendo de su extensión."""
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".nii.gz"):
            return nib.load(path).get_fdata(), nib.load(path).affine
        
        # estos no he probado.. 
        elif path.endswith(".mhd"):
            image = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(image)  
        elif os.path.isdir(path):  # Si es un directorio, asumimos que es una serie DICOM
            return self._load_dicom_series(path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {path}")


    def _load_dicom_series(self, dicom_dir):
        """
        Carga una serie de imágenes DICOM desde un directorio.

        Parameters:
            dicom_dir (str): Ruta del directorio que contiene la serie DICOM.

        Returns:
            np.ndarray: Imagen 3D representada como un arreglo NumPy.
        """
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_files)
        image = reader.Execute()  # Carga la serie DICOM como una imagen 3D
        return sitk.GetArrayFromImage(image)  # Convierte a NumPy

    
    
    def _segment_lungs(self, image):
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

            left_lung_parts_masks = []
            rigth_lung_parts_masks = []
            for part in lung_parts:
                if "left" in part:
                    left_lung_mask = os.path.join(temp_output_dir, part + ".nii.gz")
                    left_lung_parts_masks.append(nib.load(left_lung_mask).get_fdata())
                else: 
                    rigth_lung_mask = os.path.join(temp_output_dir, part + ".nii.gz")
                    rigth_lung_parts_masks.append(nib.load(rigth_lung_mask).get_fdata())

        return left_lung_parts_masks, rigth_lung_parts_masks   
    
    
    def _trim_lung(self, image, label):
        non_zero_indices = np.argwhere(image != 0)
        
        # Calcula los límites del recorte en cada dimensión
        min_z, max_z = non_zero_indices[:, 0].min(), non_zero_indices[:, 0].max()
        min_y, max_y = non_zero_indices[:, 1].min(), non_zero_indices[:, 1].max()
        min_x, max_x = non_zero_indices[:, 2].min(), non_zero_indices[:, 2].max()
        
        # Realiza el recorte en la imagen y la máscara
        cropped_image = image[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        cropped_label = label[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

        return cropped_image, cropped_label
    
    def resize_image_and_label(self, image, label, target_size=(128, 128, 128)):
        """
        Redimensiona una imagen y su máscara a las dimensiones especificadas.

        Args:
            image (np.ndarray): Imagen 3D original.
            label (np.ndarray): Máscara 3D original.
            target_size (tuple): Dimensiones objetivo (z, y, x).

        Returns:
            tuple: Imagen y máscara redimensionadas.
        """
        if image.shape != label.shape:
            raise ValueError("La imagen y la máscara deben tener las mismas dimensiones.")
        
        # Calcular los factores de escala para cada dimensión
        scale_factors = [t / s for t, s in zip(target_size, image.shape)]
        
        # Redimensionar la imagen con interpolación
        resized_image = zoom(image, scale_factors, order=3)  # Orden 3: interpolación cúbica
        
        # Redimensionar la máscara con redondeo (preserva valores binarios)
        resized_label = zoom(label, scale_factors, order=0)  # Orden 0: interpolación más cercana
        
        return resized_image, resized_label
    
    
    def clip_pixel_intensities(self, image, min_value=-1000, max_value=400):
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

    def normalized_values(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        range_val = max_val - min_val

        if range_val == 0:
            # Si todos los valores son iguales, normalizamos a un array de ceros
            return np.zeros_like(image)

        # Fórmula de normalización entre -1 y 1
        return 2 * (image - min_val) / range_val - 1
    
    
    def _preprocess(self, image, affine, label, idx):
        """
        Segmentar pulmones y recortar cada pulmón, eliminando ruido usando el otro pulmón.
        """

        """ SEGMENTAR PULMONES """
        
        left_lung_parts_masks, right_lung_parts_masks = self._segment_lungs(image)

        left_lung_only_image = np.zeros_like(image)
        right_lung_only_image = np.zeros_like(image)

        for mask in left_lung_parts_masks:
            left_lung_only_image += image * mask

        for mask in right_lung_parts_masks:
            right_lung_only_image += image * mask
            
        
        """ CLIP INTESITIES [-1000, 400]"""
        left_lung_only_image = self.clip_pixel_intensities(left_lung_only_image)
        right_lung_only_image = self.clip_pixel_intensities(right_lung_only_image)

        """ ELIMINAR RUIDO UTILIZANDO EL INICIO DEL OTRO PULMÓN """
        def remove_noise_using_start_of_other_lung(lung_image, affine):
            # binary_mask = lung_image > 0
    
            # # Calcular el número de vecinos no cero en una ventana 3D usando un filtro uniforme
            # neighborhood_count = uniform_filter(binary_mask.astype(np.float32), size=neighborhood_size) * (neighborhood_size**3)
            
            # # Generar una nueva máscara basada en el umbral de vecinos
            # refined_mask = neighborhood_count >= threshold
            
            # # Aplicar la máscara refinada a la imagen original
            # clean_lung = lung_image * refined_mask
            
            # return clean_lung 
                
            """ LAS COMPONENTES CONECTADAS TARDAN MUCHO.... """
            # # Identificar regiones conectadas
            # structure = np.ones((3, 3, 3), dtype=int)  # Estructura para considerar conexiones 26-vecinas
            # labeled_array, num_features = label_ndimage(lung_image > 0, structure=structure)
            
            # # Calcular el tamaño de cada región conectada
            # region_sizes = np.bincount(labeled_array.ravel())
            
            # # Eliminar el fondo (etiqueta 0)
            # region_sizes[0] = 0
            
            # # Identificar la etiqueta de la región más grande (que se supone es el pulmón)
            # largest_region_label = region_sizes.argmax()
            
            # # Crear una máscara para mantener solo la región más grande
            # cleaned_image = np.where(labeled_array == largest_region_label, lung_image, 0)
            
            # return cleaned_image
        
            """ Este es un codgio  que he hecho que da algun problema, mejor usar componentes conectadas 
            aunque sea  algo mas lento"""
            values_x = np.max(lung_image, axis=(1, 2))
            
            unique_indices, counts = np.unique(np.argwhere(lung_image != 0)[:,0], return_counts=True)
            
            agrupaciones = []
            grupo_actual = [unique_indices[0]]
            
            for i in range(1, len(unique_indices)):
                # Si la diferencia es menor o igual a 30, añadimos al grupo actual
                if unique_indices[i] - unique_indices[i - 1] <= 10:
                    grupo_actual.append(unique_indices[i])
                else:
                    # Si no, cerramos el grupo actual y comenzamos uno nuevo
                    agrupaciones.append(grupo_actual)
                    grupo_actual = [unique_indices[i]]

            # Añadimos el último grupo
            agrupaciones.append(grupo_actual)
                        
            for groups in agrupaciones:
                if len(groups) < 100:
                    for i in groups:
                        lung_image[i,:,:] = 0

            return lung_image

        # Aplica el filtro de eliminación de ruido a las imágenes de los pulmones
        left_lung_only_image = remove_noise_using_start_of_other_lung(left_lung_only_image, affine)
        right_lung_only_image = remove_noise_using_start_of_other_lung(right_lung_only_image, affine)


        """ COMPROBAR QUE LA MÁSCARA tiene los mismos valores que la imagen original en los pulmones y que no se han superpuesto las segmentaciones a la hoa de sumar"""
        def verify_mask(original_image, segmented_image):
            mask = segmented_image > 0
            original_values = original_image[mask]
            segmented_values = segmented_image[mask]
            if not np.array_equal(original_values, segmented_values):
                segmented_image[mask] = original_image[mask]
            return segmented_image

        left_lung_only_image = verify_mask(image, left_lung_only_image)
        right_lung_only_image = verify_mask(image, right_lung_only_image)


        """ RECORTAR PULMONES """        
        left_lung, left_label = self._trim_lung(left_lung_only_image, label)
        right_lung, right_label = self._trim_lung(right_lung_only_image, label)
        
        """ RESIZE A (128, 128, 128) """        
        resized_left_lung, resized_left_label = self.resize_image_and_label(left_lung, left_label)
        resized_right_lung, resized_right_label = self.resize_image_and_label(right_lung, right_label)

        """ NORMALIZE VALUES [-1, 1] """
        resized_left_lung = self.normalized_values(resized_left_lung)
        resized_right_lung = self.normalized_values(resized_right_lung)
        
        return resized_left_lung, resized_right_lung , resized_left_label, resized_right_label

    
    def __getitem__(self, idx):
        if not self.use_cache:
            image, affine = self._load_file(self.images[idx])
            label, affine = self._load_file(self.labels[idx])
            
            # segmentar lung -> recortar cada pulmon -> resize 
            if self.preprocess:
                left_lung, right_lung , left_label, right_label = self._preprocess(image, affine, label, idx)
                
                if self.data_aug:
                    left_lung_aug, left_label_aug = self.data_aug(left_lung, left_label)
                    right_lung_aug, right_label_aug = self.data_aug(right_lung, right_label)

                    if len(left_lung_aug) > 0:
                        for i in left_lung_aug: 
                            self.cache_images.append(torch.from_numpy(left_lung_aug[i].copy()).float())
                            self.cache_labels.append(torch.from_numpy(left_label_aug[i.replace("_image","_mask")].copy()).long())
                            
                    if len(right_lung_aug) > 0:
                        for i in left_lung_aug: 
                            self.cache_images.append(torch.from_numpy(right_lung_aug[i].copy()).float())
                            self.cache_labels.append(torch.from_numpy(right_label_aug[i.replace("_image","_mask")].copy()).long())
                               
                left_lung = torch.from_numpy(left_lung.copy()).float()
                right_lung = torch.from_numpy(right_lung.copy()).float()
                
                left_label = torch.from_numpy(left_label.copy()).long() 
                right_label = torch.from_numpy(right_label.copy()).long() 
                
                # name = self.images[idx].split("/")[len(self.images[idx].split("/"))-1]

                self.cache_images.append(left_lung)
                self.cache_images.append(right_lung)
                
                self.cache_labels.append(left_label)
                self.cache_labels.append(right_label)
             
                return left_lung, left_label
                
            else:        
                image = torch.from_numpy(image).float()
                label = torch.from_numpy(label).long() 
                
                # name = self.images[idx].split("/")[len(self.images[idx].split("/"))-1]

                self.cache_images.append(image)
                self.cache_labels.append(label)
                
                return image, label

        else:
            image = self.cache_images[idx]
            label = self.cache_labels[idx]
            
            if self.transform:
                image = image.cpu().numpy()
                label = label.cpu().numpy()
                image, label = self.transform((image, label))
                
                image = torch.from_numpy(image).float()
                label = torch.from_numpy(label).long() 

            return image, label

