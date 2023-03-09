import os
import cv2
import numpy as np

"""def ReadImageDataBase(Textpath, Extension='.jpg'):
    expectedExtensions = ['.jpg', '.pgm', '.bmp', '.png', '.tif']
    if not Extension.lower() in [ext.lower() for ext in expectedExtensions]:
        raise ValueError('Invalid file extension')
    
    files = os.listdir(Textpath)
    A = None
    m, n = None, None
    num_img = 0
    
    for filename in files:
        filepath = os.path.join(Textpath, filename)
        if os.path.isdir(filepath) or not filename.lower().endswith(Extension.lower()):
            continue
        
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        if A is None:
            m, n = img.shape
            A = img.reshape(-1, 1)
        else:
            if A.shape[0] == 0:
                A = img.reshape(-1, 1)
            else:
                A = np.hstack((A, img.reshape(-1, 1)))
        num_img += 1
    
    if num_img == 0:
        raise ValueError(f"There are no images with extension {Extension} in {Textpath}")
    
    return A, m, n"""

def ReadImageDataBase(Textpath, Extension='.jpg'):
    # Lista de extensiones de archivo válidas
    expectedExtensions = ['.jpg', '.pgm', '.bmp', '.png', '.tif']
    
    # Si la extensión del archivo no está en la lista de extensiones válidas, lanza un error
    if not Extension.lower() in [ext.lower() for ext in expectedExtensions]:
        raise ValueError('Invalid file extension')
    
    # Obtiene la lista de archivos en la ruta especificada
    files = os.listdir(Textpath)
    
    # Inicializa la matriz de datos y las dimensiones de la imagen
    A = None
    m, n = None, None
    
    # Contador de imágenes válidas
    num_img = 0
    
    # Itera sobre los archivos en la carpeta
    for filename in files:
        # Obtiene la ruta completa del archivo
        filepath = os.path.join(Textpath, filename)
        
        # Si el archivo es una carpeta o no tiene la extensión válida, omítelo
        if os.path.isdir(filepath) or not filename.lower().endswith(Extension.lower()):
            continue
        
        # Lee la imagen en escala de grises como una matriz de tipo float en el rango [0,1]
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        
        # Si es la primera imagen, establece las dimensiones de la matriz de datos
        if A is None:
            m, n = img.shape
            A = img.reshape(-1, 1)
        else:
            # Si no es la primera imagen, agrega la imagen a la matriz de datos
            if A.shape[0] == 0:
                A = img.reshape(-1, 1)
            else:
                A = np.hstack((A, img.reshape(-1, 1)))
        
        # Incrementa el contador de imágenes válidas
        num_img += 1
    
    # Si no se encontraron imágenes válidas, lanza un error
    if num_img == 0:
        raise ValueError(f"There are no images with extension {Extension} in {Textpath}")
    
    # Devuelve la matriz de datos y las dimensiones de la imagen
    return A, m, n