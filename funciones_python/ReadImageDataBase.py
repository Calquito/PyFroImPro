import os
import cv2
import numpy as np


def ReadImageDataBase(Textpath, Extension='.jpg'):
    # Lista de extensiones de archivo válidas
    expectedExtensions = ['.jpg', '.pgm', '.bmp', '.png', '.tif']
    
    # Si la extensión del archivo no está en la lista de extensiones válidas, lanzar un error
    if not Extension.lower() in [ext.lower() for ext in expectedExtensions]:
        raise ValueError('Invalid file extension')
    
    # Obtener la lista de archivos en la ruta especificada
    files = os.listdir(Textpath)
    
    # Inicializar la matriz de datos y las dimensiones de la imagen
    A = None
    m, n = None, None
    
    # Contador de imágenes válidas
    num_img = 0
    
    # Iterar sobre los archivos en la carpeta
    for filename in files:
        # Obtener la ruta completa del archivo
        filepath = os.path.join(Textpath, filename)
        
        # Si el archivo es una carpeta o no tiene la extensión válida, omitirla
        if os.path.isdir(filepath) or not filename.lower().endswith(Extension.lower()):
            continue
        
        # Leer la imagen en escala de grises como una matriz de tipo float en el rango [0,1]
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        
        # Si es la primera imagen, establecer las dimensiones de la matriz de datos
        if A is None:
            m, n = img.shape
            A = img.reshape(-1, 1)
        else:
            # Si no es la primera imagen, agregar la imagen a la matriz de datos
            if A.shape[0] == 0:
                A = img.reshape(-1, 1)
            else:
                A = np.hstack((A, img.reshape(-1, 1)))
        
        # Incrementar el contador de imágenes válidas
        num_img += 1
    
    # Si no se encontraron imágenes válidas, lanzar un error
    if num_img == 0:
        raise ValueError(f"There are no images with extension {Extension} in {Textpath}")
    
    # Devolver la matriz de datos y las dimensiones de la imagen
    return A, m, n