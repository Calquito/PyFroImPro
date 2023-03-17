import cv2
import numpy as np

def missing_pixels(Textpath, porc):
    # Si la fracción porc no está entre 0 y 1, mostrar error y terminar funcion
    if (porc<=0 or porc>=1):
        raise ValueError('Take into account that 0<porc<1')
           
    # Lee la imagen
    Y = cv2.imread(Textpath)

    # Si la imagen es a color, convertirla a escala de grises
    if len(Y.shape) == 3 and Y.shape[2] == 3:
        Y = cv2.cvtColor(Y, cv2.COLOR_BGR2GRAY)
        
    # Obtener el tamaño de la imagen
    m, n = Y.shape[:2]

    # Recorrer cada pixel de la imagen aleatoriamente y eliminarlo si es menor a porc
    for i in range(m):
        for j in range(n):
            if np.random.rand() < porc:
                Y[i,j] = 0

    # Obtener el nombre del archivo de la ruta de la imagen
    VectorPosicion = [pos for pos, char in enumerate(Textpath) if char == '/']
    if len(VectorPosicion) == 0:
        Name = 'MissingPixelsImage' + Textpath[-4:]
    else:
        NewPath = Textpath[:VectorPosicion[-1]+1]
        Name = NewPath + 'MissingPixelsImage_' + Textpath[VectorPosicion[-1]+1:]
        
    # Guardar la imagen con los píxeles faltantes eliminados en el archivo con nombre Name
    cv2.imwrite(Name, Y)