from ReadImageDataBase import ReadImageDataBase # Importa la función ReadImageDatabase desde un archivo externo
import numpy as np
import cv2
import os

def SaveFilteredImages(pathNoise,pathWriteFiltered,X):
    C, m, n = ReadImageDataBase(pathNoise)
    #X=np.array(X,dtype=object)
    #print(X)
    print(C)
    _,NumImgWithNoise = np.shape(C)[:2]
    #NumImgWithNoise,_ = np.shape(C)[:2]
    print(NumImgWithNoise)
    for i in range(1, NumImgWithNoise):
        ImagenNIConRuidoColumna = C[:,i-1] # Leo la columna i de C
        ImagenNIConRuido = ImagenNIConRuidoColumna.reshape((m, n)) # La convierto en matriz
        cv2.imwrite(os.path.join(pathWriteFiltered, f"BlurredImage ({i}).jpg"), ImagenNIConRuido) # Genero la imagen i
        imagenNIFiltradaColumna = np.dot(X, ImagenNIConRuidoColumna)
        imagenNIFiltrada = imagenNIFiltradaColumna.reshape((m, n)) # La convierto en matriz
        cv2.imwrite(os.path.join(pathWriteFiltered, f"FilteredImage ({i}).jpg"), imagenNIFiltrada) # Genero la imagen i

        