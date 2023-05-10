import numpy as np
from PIL import Image
import os

def ReconstructionExternalFace(Textpath,Results_Texthpath, W, NameOutput):
    # Función para validar que la entrada W sea una matriz numérica
    def ValidMatrix(x):
        return isinstance(x, np.ndarray) and x.ndim == 2 and np.issubdtype(x.dtype, np.number)

    # Si W no es una matriz numérica, lanza un error
    if not ValidMatrix(W):
        raise ValueError('The value of W is not valid')

    direccion = Results_Texthpath+'/Results/ReconstructedFaces'
    # Si la carpeta "Results" no existe, crea la carpeta
    if not os.path.exists(direccion):
        os.mkdir(direccion)

    # Carga la imagen externa y obtiene sus dimensiones
    ImagenExterna = np.array(Image.open(Textpath).convert('L')).astype(np.float64)
    m, n = ImagenExterna.shape

    # Convierte la imagen en un vector y calcula los coeficientes de la representación de la imagen en el espacio de W
    ImagenExternaVector = ImagenExterna.reshape(m*n, 1)
    CoefImagenExterna = np.linalg.pinv(W) @ ImagenExternaVector

    # Reconstruye la imagen a partir de los coeficientes y la matriz W
    ImagenReconstruidaVector = W @ CoefImagenExterna
    ImagenReconstruida = ImagenReconstruidaVector.reshape(m, n)

    # Guarda la imagen reconstruida y la imagen original en la carpeta "Results/ReconstructedFaces"
    Image.fromarray(ImagenReconstruida.astype(np.uint8)).save(os.path.join(direccion, NameOutput + '.jpg'))
    Image.fromarray(ImagenExterna.astype(np.uint8)).save(os.path.join(direccion, NameOutput + '_Original.jpg'))