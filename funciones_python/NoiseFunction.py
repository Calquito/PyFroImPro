import numpy as np
from ReadImageDataBase import ReadImageDataBase # Importa la función ReadImageDatabase desde un archivo externo
from skimage import util

def NoiseFunction(Textpath, NoiseOption='gaussian', sigmagauss=0.01, meangauss=0, d=0.05, sigmaspeckle=0.05):

    # Lee la base de imágenes
    A, m, n = ReadImageDataBase(Textpath)
    _, num_img = np.shape(A)[:2]

    # Ajusta los valores por defecto
    expectedNoises = ['speckle', 's&p', 'gaussian']
    NoiseOption = NoiseOption.lower().replace(' ', '')
    if NoiseOption == 's&p':
        NoiseOption = 'salt & pepper'
    if NoiseOption not in expectedNoises:
        raise ValueError('The valid options for the parameter NoiseOption are gaussian, s&p and speckle.')

    N = np.zeros((m, n, num_img))
    C = np.zeros((m*n, num_img))

    for i in range(num_img):
        ImageMatrix = np.reshape(A[:, i], (m, n))

        if NoiseOption == 'speckle':
            # Aplica ruido speckle a la imagen
            N[:, :, i] = util.random_noise(ImageMatrix, mode='speckle', var=sigmaspeckle)
        elif NoiseOption == 'salt & pepper':
            # Aplica ruido de tipo 's&p' a la imagen
            N[:, :, i] = util.random_noise(ImageMatrix, mode='s&p', amount=d)
        elif NoiseOption == 'gaussian':
            # Aplica ruido gaussiano a la imagen
            N[:, :, i] = util.random_noise(ImageMatrix, mode='gaussian', mean=meangauss, var=sigmagauss)

        C[:, i] = np.reshape(N[:, :, i], (m*n,)) # Guarda la imagen con ruido en la matriz C

    return C, A, m, n