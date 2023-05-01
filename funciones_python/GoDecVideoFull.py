from Video2Matrix import Video2Matrix
from Matrix2Video import Matrix2Video
from GoDec import GoDec
import numpy as np
import cv2
import os

def GoDecVideoFull(TextPath, r, k, Opcion):

    X, m1, n1 = Video2Matrix(TextPath)
    # k = fix(0.075*n1*m1*200);
    LRF, SRF, _ = GoDec(X, r, k, Opcion)
    Matrix2Video(SRF, m1, n1, './S.mp4')
    Matrix2Video(LRF, m1, n1, './L.mp4')
    #MatrixXLS2Video(X, LRF, SRF, m1, n1, 'XLS')


def MatrixXLS2Video(X, L, S, m, n, PathNameVideo):
    if not isinstance(L, np.ndarray) or L.ndim != 2:
        raise ValueError('The parameter L is invalid.')
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        raise ValueError('The parameter S is invalid.')
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError('The parameter X is invalid.')
    if not isinstance(m, int) or m <= 0:
        raise ValueError('The value for parameter "m" is invalid.')
    if not isinstance(n, int) or n <= 0:
        raise ValueError('The value for parameter "n" is invalid.')
    numPixeles, numFrames = X.shape
    if m * n != numPixeles:
        raise ValueError('The values of m and n do not correspond to the number of pixels in each frame: Pixels number = m*n')

    # Termina la validación de parámetros e inicia la creación del vídeo
    posUltiLinea = PathNameVideo.rfind('\\')
    if posUltiLinea > 1:
        if not os.path.exists(PathNameVideo[0:posUltiLinea-1]):
            os.makedirs(PathNameVideo[0:posUltiLinea-1])
    v = cv2.VideoWriter(PathNameVideo, cv2.VideoWriter_fourcc(*'mp4v'), 30, (n, m * 3))  # Define el vídeo.
    for ii in range(numFrames):
        Frame = np.vstack([X[:, ii].reshape(m, n), L[:, ii].reshape(m, n), S[:, ii].reshape(m, n)])