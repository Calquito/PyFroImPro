
import os
from PIL import Image
import numpy as np
from double2uint8 import double2uint8

def Matrix2Image(X, m, n, TextPath, Name):

    # Tamaño de cada imagen.
    isize = [m, n]
    # Número de imágenes.
    p = X.shape[1]
    # Ruta completa donde se guardarán las imágenes.
    direccion = os.path.join(TextPath, Name)

    # Si el directorio no existe, crearlo.
    if not os.path.exists(TextPath):
        os.makedirs(TextPath)

    # Número de ceros necesarios para nombrar las imágenes.
    NumCeros = len(str(p))
    # Texto con los ceros necesarios.
    CerosText = NStr('0', NumCeros)

    # Para cada imagen, convertir a uint8 y guardar en disco.
    for i in range(p):
        # Número de ceros necesarios para el nombre actual.
        NumCerosActual = len(str(i+1))
        # Texto con los ceros necesarios para el nombre actual.
        CerosActual = CerosText[0:(NumCeros-NumCerosActual)]
        # Convertir imagen a uint8 y guardar en disco.
        im = Image.fromarray(double2uint8(np.reshape(X[:,i], isize)))
        im.save(direccion + CerosActual + str(i+1) + '.jpg')

def NStr(Text, N):

    # Cadena de caracteres vacía.
    TextOut = ''
    # Concatenar la cadena de caracteres N veces.
    for i in range(N):
        TextOut = TextOut + Text
    return TextOut