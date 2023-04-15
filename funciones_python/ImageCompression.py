from LowRankMatrixBRP import LowRankMatrixBRP
from double2uint8 import double2uint8
import cv2
import numpy as np
import os

def ImageCompression(Textpath, radio, Opcion):
    # Validación de la opción
    ExpectedOptions = ['BRP', 'SVD']
    Opcion = Opcion.upper().replace(' ', '')
    ValidarOpcion = lambda x: x in ExpectedOptions
    if not ValidarOpcion(Opcion):
        raise ValueError(f'Opción no admitida. Debe indicar BRP o SVD. Usted digitó {Opcion}')

    # Lectura de la imagen
    ima = cv2.imread(Textpath)
    if ima.shape[2] == 3:
        L = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        L = ima.astype(np.float64)
    m, n = L.shape
    Minmn = min(m, n)

    # Validación del radio de compresión
    Validporcentaje = lambda x: x > 1/Minmn and x <= 1
    if not Validporcentaje(radio):
        raise ValueError(f'El radio de compresión indicado para esta imagen no es válido. Debe ser un decimal d, tal que {1/Minmn} <= d <= 1.')

    r = int(np.floor(radio * Minmn))

    if Opcion == 'BRP':
        A, B = LowRankMatrixBRP(L,r)
    else:
        U, S, V = np.linalg.svd(L)
        Urecortada = U[:, :r]
        Srecortada = np.diag(S[:r])
        Vrecortada = V[:, :r]
        A = Urecortada @ Srecortada
        B = Vrecortada.T

    LTilde = A @ B
    direccion = 'Results'
    if not os.path.exists(direccion):
        os.mkdir(direccion)
    cv2.imwrite('Results/OriginalImage.jpg', double2uint8(L))
    cv2.imwrite('Results/CompressedImage.jpg', double2uint8(LTilde))
    np.savetxt('Results/L.txt', L)
    np.savetxt('Results/A.txt', A)
    np.savetxt('Results/B.txt', B)