from ReadImageDataBase import ReadImageDataBase # Importa la función ReadImageDatabase desde un archivo externo
import numpy as np
import os
from PIL import Image
from double2uint8 import double2uint8

def SaveFilteredImages(pathNoise,pathWriteFiltered,X):
    C, m, n = ReadImageDataBase(pathNoise)
    #X=np.array(X,dtype=object)
    #print(X)
    _,NumImgWithNoise = np.shape(C)[:2]
    #NumImgWithNoise,_ = np.shape(C)[:2]

    # Si el directorio no existe, crearlo.
    if not os.path.exists(pathWriteFiltered):
        os.makedirs(pathWriteFiltered)

    for i in range(NumImgWithNoise):
        ImagenNIConRuidoColumna = C[:,i-1] # Leo la columna i de C
        ImagenNIConRuido = ImagenNIConRuidoColumna.reshape((m, n)) # La convierto en matriz

        #Guardar imagen
        im = Image.fromarray(double2uint8(ImagenNIConRuido))
        im.save(pathWriteFiltered+"BlurredImage" + str(i+1) + '.jpg')

        #Obtener imagen filtrada y guardarla
        imagenNIFiltradaColumna = np.dot(X, ImagenNIConRuidoColumna)
        imagenNIFiltrada = imagenNIFiltradaColumna.reshape((m, n)) # La convierto en matriz
        im = Image.fromarray(double2uint8(imagenNIFiltrada))
        im.save(pathWriteFiltered+"FilteredImage" + str(i+1) + '.jpg')

        