'''import os
from PIL import Image
import numpy as np
from double2uint8 import double2uint8

def Matrix2Image(X, m, n, TextPath, Name):
    isize = [m, n] # tamaño de cada imagen
    p = X.shape[1] # número de imágenes
    direccion = os.path.join(TextPath, Name)
    if not os.path.exists(TextPath):
        os.makedirs(TextPath)
    NumCeros = len(str(p))
    CerosText = NStr('0', NumCeros)
    for i in range(p):
        NumCerosActual = len(str(i+1))
        CerosActual = CerosText[0:(NumCeros-NumCerosActual)]
        #imwrite(double2uint8(reshape(X[:,i], isize)), direccion + CerosActual + str(i+1) + '.jpg')
        im = Image.fromarray(double2uint8(np.reshape(X[:,i], isize)))
        im.save(direccion + CerosActual + str(i+1) + '.jpg')

def NStr(Text, N):
    # Recibe una cadena de caracteres en "Text" y un número natural N.
    # Retorna una cadena donde se concatenó Text N veces.
    return Text*N'''

import os
from PIL import Image
import numpy as np
from double2uint8 import double2uint8

def Matrix2Image(X, m, n, TextPath, Name):
    """
    Convierte una matriz de imágenes en una serie de imágenes guardadas en disco.

    Args:
    - X: Matriz de imágenes, donde cada columna es una imagen a convertir.
    - m, n: Tamaño de cada imagen.
    - TextPath: Directorio donde se guardarán las imágenes.
    - Name: Nombre base de las imágenes.

    Returns:
    - Nada. Las imágenes se guardan en disco.
    """

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
    """
    Concatena una cadena de caracteres N veces.

    Args:
    - Text: Cadena de caracteres a repetir.
    - N: Número de veces a repetir.

    Returns:
    - TextOut: Cadena de caracteres concatenada N veces.
    """

    # Cadena de caracteres vacía.
    TextOut = ''
    # Concatenar la cadena de caracteres N veces.
    for i in range(N):
        TextOut = TextOut + Text
    return TextOut