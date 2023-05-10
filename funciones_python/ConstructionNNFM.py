import os
from nnfLS import nnfLS # Importa la función nnfLS del archivo nnfLS.py
from ReadImageDataBase import ReadImageDataBase # Importa la función ReadImageDataBase del archivo ReadImageDataBase.py
from Matrix2Image import Matrix2Image # Importa la función Matrix2Image del archivo Matrix2Image.py

def ConstructionNNFM(Textpath,Results_Texthpath, *args):
    # Lee la base de datos de imágenes
    A, m, n = ReadImageDataBase(Textpath, *args) # Llama a la función ReadImageDataBase

    # Calcula la factorización no negativa de A utilizando la función nnfLS
    W, H, _ = nnfLS(A, *args) # Llama a la función nnfLS

    # Crea un directorio 'Results/ImagesBase' si no existe
    direccion = Results_Texthpath+'/Results/ImagesBase'
    if not os.path.exists(direccion):
        os.makedirs(direccion)

    # Guarda W y H en archivos .mat en el directorio 'Results'
    from scipy.io import savemat
    savemat('Results/W.mat', {'W': W})
    savemat('Results/H.mat', {'H': H})

    # Convierte la matriz W en una imagen y la guarda en el directorio 'Results/ImagesBase'
    Matrix2Image(W, m, n, direccion, 'BaseImage') # Llama a la función Matrix2Image