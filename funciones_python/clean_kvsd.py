import numpy as np
import cv2

from PIL import Image
import numpy as np
from scipy import ndimage

def clean_ksvd(TextPath, D):
    # Carga la imagen y convierte a una matriz de doble precisión
    I = Image.open(TextPath).convert('L')
    Z = np.array(I).astype(np.float64) / 255.0
    
    # Divide la imagen en bloques de 8x8
    VecBlock = im2block8(Z)
    mBlock, nBlock = VecBlock.shape
    
    t = 10
    iter = 0
    blockImg = []
    
    for i in range(mBlock):
        Aux = []
        for j in range(nBlock):
            iter += 1
            y = VecBlock[i, j]
            D_d = decim_mat(D, y)
            x = OMP(y, D_d, t)
            block = block_img_8(D @ x)
            Aux.append(block)
        blockImg.append(np.concatenate(Aux, axis=1))
    
    # Concatena los bloques de imagen en una matriz y redimensiona
    Y = np.concatenate(blockImg, axis=0)
    m_org, n_org = Z.shape
    Y = (Y * 255.0).clip(0, 255).astype(np.uint8)
    Y = Image.fromarray(Y)
    Y = Y.resize((n_org, m_org), Image.BICUBIC)
    
    return np.array(Y)

def im2block8(X):
    """
    Esta función convierte una matriz X de tamaño m x n en una matriz Y de
    tamaño 64 x mn/64, donde cada columna de Y vienen de un bloque
    vectorizado de tamaño 8 x 8 de X
    
    :param X: una matriz de tamaño m x n
    :return: una matriz Y de tamaño 64 x mn/64
    
    """
    # Obtener las dimensiones de la matriz X
    m, n = X.shape

    # Obtener el número de bloques en la dirección horizontal y vertical
    b1 = m // 8
    b2 = n // 8

    # Inicializar la matriz Y
    Y = np.empty((64, b1 * b2))

    # Iterar sobre cada bloque y vectorizarlo
    for i in range(b1):
        for j in range(b2):
            block = X[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            vector = block.flatten(order='F')  # orden Fortran (column-major)
            Y[:, i * b2 + j] = vector

    return Y


def OMP(y, A, k):
    """
    Algoritmo "Orthogonal Matching Pursuit"
    
    :param y: vector de observación
    :param A: matriz de diseño
    :param k: número máximo de elementos no nulos en x
    
    :return: vector solución
    """
    
    n = A.shape[1]
    r = y
    T = []
    x = np.zeros(n)
    
    for i in range(k):
        g = A.T @ r
        t = argmax_OMP(g, A)
        T.append(t)
        T.sort()
        A_T = mtx_colt(T, A)
        xaux = np.linalg.pinv(A_T) @ y
        for j, index in enumerate(T):
            x[index] = xaux[j]
        r = y - A @ x
        
    return x

def argmax_OMP(g, A):
    """
    Encuentra el índice del máximo valor de una lista de valores obtenidos a partir de g y A
    
    :param g: lista de valores
    :param A: matriz de diseño
    
    :return: índice del valor máximo
    """
    
    x = []
    n = A.shape[1]
    for j in range(n):
        z = abs(g[j]) / (np.linalg.norm(A[:, j]))
        x_n = np.append(x, z)
        x = x_n
    
    t = np.argmax(x)
    
    return t

def mtx_colt(T, A):
    """
    Selecciona las columnas de la matriz A especificadas por T
    
    :param T: lista de índices de columnas
    :param A: matriz de diseño
    
    :return: submatriz de A formada por las columnas seleccionadas
    """
    
    m = A.shape[0]
    n1 = len(T)
    B = np.zeros((m, n1))
    
    for i, index in enumerate(T):
        B[:, i] = A[:, index]
        
    return B


def decim_mat(D0, y):
    n = y.size
    D = D0.copy()
    m = D0.shape[1]
    for i in range(n-1):
        if y[i] == 0:
            D[i,:] = np.zeros((m))
    return D

def block_img_8(X):
    m, n = X.shape
    s1, s2 = m % 8, n % 8
    if s1 == 0 and s2 == 0:
        m1, n1 = m, n
    elif s1 == 0 and s2 != 0:
        if s2 <= 4:
            m1, n1 = m, n - s2
        else:
            m1, n1 = m, n + 8 - s2
    elif s1 != 0 and s2 == 0:
        if s1 <= 4:
            m1, n1 = m - s1, n
        else:
            m1, n1 = m + 8 - s1, n
    elif s1 != 0 and s2 != 0:
        if s1 <= 4:
            m1 = m - s1
        else:
            m1 = m + 8 - s1
        if s2 <= 4:
            n1 = n - s2
        else:
            n1 = n + 8 - s2

    Y = np.array(ndimage.zoom(X, (m1 / m, n1 / n), order=1))
    return Y.astype('float64')