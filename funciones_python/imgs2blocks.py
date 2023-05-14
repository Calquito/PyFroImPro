import os
import cv2
import numpy as np
from PIL import Image

def imgs2blocks(Textpath, Extension):
    # Vector con los nombres completos de todos los archivos en la carpeta
    NameFilesVector = os.listdir(Textpath)

    # Controla el número de archivos válidos en la carpeta,
    # dado que en la carpeta podrían haber archivos del sistema,
    # los cuales hay que ignorar.
    i = 0
    Y = []

    # Recorre todos los archivos de la carpeta
    for j in range(len(NameFilesVector)):
        Filej = NameFilesVector[j] # Toma el archivo j de la carpeta
        _, Fileextension = os.path.splitext(Filej) # Lee la extensión del archivo

        # Valida el archivo y si la extensión coincide
        if not os.path.isdir(os.path.join(Textpath, Filej)) and \
           (Fileextension.lower() == Extension.lower() or Extension == ""):

            # Archivo válido, incrementa el contador de imágenes
            i += 1

            # Carga la dirección completa del archivo
            direccion = os.path.join(Textpath, Filej)

            # Lee la imagen y convierte la imagen en matriz de tipo doble
            ima = cv2.imread(direccion)
            if ima.shape[2] == 3:
                ImageMatrix = cv2.cvtColor(ima, cv2.COLOR_RGB2GRAY)
            else:
                ImageMatrix = ima

            # Registra las dimensiones de cada imagen
            if i == 1:
                m, n = ImageMatrix.shape

            # Convierte la matriz en bloques de 8x8
            Xaux1 = ImageMatrix
            Xaux2 = cv2.normalize(Xaux1, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
            Xaux3 = im2block8(Xaux2)
            Y.append(Xaux3)

    num_img = i
    if i == 0:
        raise ValueError(f"There are not images in your data base with the extension {Extension}")

    return np.hstack(Y)

def im2block8(X):
    # Esta función convierte una matriz X de tamaño m x n en una matriz Y de
    # tamaño (64 x mn/64), donde cada columna de Y vienen de un bloque
    # vectorizado de tamaño  8 x 8 de X.

    m, n = X.shape
    b1, b2 = int(m/8), int(n/8)
    Y = []

    for i in range(b1):
        for j in range(b2):
            Aux = X[i*8:(i+1)*8, j*8:(j+1)*8]
            Y.append(Aux.flatten())

    return np.asarray(Y).T

def block_img_8(X):
    # Esta función retorna una imagen, donde el número de filas y columnas
    # es divisible por 8. Las dimensiones de la nueva imagen Y son los
    # múltiplos de 8 más cercanos a las dimensiones de la imagen original X

    # Una imagen de tamaño m x n.
    m, n = X.shape

    # Calcula las dimensiones de la imagen redondeadas a múltiplos de 8
    s1, s2 = m % 8, n % 8
    if s1 == 0 and s2 == 0:
        m1, n1 = m, n
    elif s1 == 0 and s2 != 0:
        if s2 <= 4:
            m1, n1 = m, n-s2
        else:
            m1, n1 = m, n+8-s2
    elif s1 != 0 and s2 == 0:
        if s1 <= 4:
            m1, n1 = m-s1, n
        else:
            m1, n1 = m+8-s1, n
    elif s1 != 0 and s2 != 0:
        if s1 <= 4:
            m1 = m-s1
        else:
            m1 = m+8-s1
        if s2 <= 4:
            n1 = n-s2
        else:
            n1 = n+8-s2
            
    Y = np.array(Image.fromarray(X.astype(np.uint8)).resize((n1, m1), Image.ANTIALIAS), dtype=np.float32)
    
    return Y