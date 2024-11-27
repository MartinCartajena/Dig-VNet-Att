import torch
import numpy as np

class BitwiseImageTransformer:
    def __init__(self, input_tensor):
        """
        Inicializa la clase con el tensor de entrada.
        
        Argumentos:
        - input_tensor (torch.Tensor): Tensor de entrada con shape [batch_size, 1, depth, height, width].
        """
        self.input_tensor = input_tensor
        self.original_ = input_tensor
        self.batch_size, self.depth, self.height, self.width = input_tensor.shape

    def _generate_bit_masks(self):
        """
        Genera máscaras para los primeros 4 bits más significativos de cada píxel.
        
        Retorna:
        - masks (list of torch.Tensor): Lista de 4 tensores binarios, cada uno con valores de 128 o 0 en cada posición.
        """
        masks = []
        for i in range(4):  
            multiplier = 128 if i < 3 else 0  # Multiplica por 128 para los primeros tres bits y por 0 en el cuarto
            bit_mask = ((self.input_tensor.int() >> (7 - i)) & 1) * multiplier
            masks.append(bit_mask.float())
        return masks

    def _calculate_images(self, masks):
        """
        Calcula la resta entre la imagen original y cada máscara.
        
        Argumentos:
        - masks (list of torch.Tensor): Lista de máscaras de bits generadas.
        
        Retorna:
        - resta_images (list of torch.Tensor): Lista de imágenes de resta.
        """
        return [self.input_tensor - mask for mask in masks]
    
    def normalize_to_bit8(self):
        """
        Convierte datos normalizados en el rango [-1, 1] a pixeles de 8 bits en el rango [0, 255].
        """
        # Normalizamos y redondeamos para precisión en la conversión a 8 bits
        self.input_tensor = torch.round((self.input_tensor + 1) / 2 * 255).to(torch.uint8)

    def denormalize_from_bit8(self, tensor_bit8):
        """
        Convierte los datos de pixeles de 8 bits en el rango [0, 255] de vuelta al rango [-1, 1].
        """
        # Convertimos a float y revertimos la normalización
        return (tensor_bit8.float() / 255 * 2) - 1

    def transform(self):
        """
        Realiza la transformación para generar el tensor de salida con 16 canales.
        
        Retorna:
        - output_tensor (torch.Tensor): Tensor con shape [batch_size, 16, depth, height, width].
        """
    
        self.normalize_to_bit8()
        masks = self._generate_bit_masks()
        dig_sep_images = self._calculate_images(masks)
        
        
        duplicated_dig_sep_images = [torch.cat((t.unsqueeze(1), t.unsqueeze(1)), dim=1) for t in dig_sep_images]
        
        dig_sep_res = torch.cat([t for t in duplicated_dig_sep_images], dim=1)
        
        expanded_tensor = self.input_tensor.unsqueeze(1).repeat(1, 8, 1, 1, 1)
        
        output_tensor = torch.cat((dig_sep_res, expanded_tensor), dim=1)
        # distancia_max = 2 * (x[5,:,:,:].numel() ** 0.5)
        # (torch.norm(x[5,:,:,:] - dig_x[5,14,:,:,:]) / distancia_max).item()
        return self.denormalize_from_bit8(output_tensor)  
