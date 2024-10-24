import os
import numpy as np

def load_npy_files_from_directory(directory):
    """
    Carga archivos .npy desde un directorio y devuelve un diccionario con el nombre del archivo como clave y
    el contenido como valor.
    """
    npy_files = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory, file_name)
            npy_files[file_name] = np.load(file_path)
    return npy_files