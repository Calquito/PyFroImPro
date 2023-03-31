from ReadImageDataBase import ReadImageDataBase # Importa la función ReadImageDatabase desde un archivo externo
import numpy as np
import cv2
import os

def SaveFilteredImages(pathNoise,pathWriteFiltered,X):
    C, m, n = ReadImageDataBase(pathNoise)
    _, NumImgWithNoise = np.shape(C)[:2]
    for i in range(1, NumImgWithNoise+1):
        ImagenNIConRuidoColumna = C[:,i-1] # Leo la columna i de C
        ImagenNIConRuido = ImagenNIConRuidoColumna.reshape((m, n)) # La convierto en matriz
        cv2.imwrite(os.path.join(pathWriteFiltered, f"BlurredImage ({i}).jpg"), ImagenNIConRuido) # Genero la imagen i
        imagenNIFiltradaColumna = np.dot(X, ImagenNIConRuidoColumna)
        imagenNIFiltrada = imagenNIFiltradaColumna.reshape((m, n)) # La convierto en matriz
        cv2.imwrite(os.path.join(pathWriteFiltered, f"FilteredImage ({i}).jpg"), imagenNIFiltrada) # Genero la imagen i
        