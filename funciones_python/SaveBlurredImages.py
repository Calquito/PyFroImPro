import os
import cv2
import numpy as np
from NoiseFunction import NoiseFunction


def SaveBlurredImages(Textpath,pathWriteNoise, NoiseOption='gaussian', sigmagauss=0.01, meangauss=0, d=0.05, sigmaspeckle=0.05):
    C,_,m,n=NoiseFunction(Textpath, NoiseOption, sigmagauss, meangauss, d, sigmaspeckle)
    _, num_img = np.shape(C)[:2]
    for i in range(1, num_img+1):
        ImagenNIConRuidoColumna = C[:,i-1] # Leo la columna i de C
        ImagenNIConRuido = ImagenNIConRuidoColumna.reshape((m, n)) # La convierto en matriz
        cv2.imwrite(os.path.join(pathWriteNoise, f"BlurredImage ({i}).jpg"), ImagenNIConRuido) # Genero la imagen i
