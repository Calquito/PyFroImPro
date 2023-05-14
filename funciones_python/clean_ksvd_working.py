import numpy as np
from scipy.io import loadmat
from scipy import ndimage
import cv2

def clean_ksvd2(TextPath, D):
    I = cv2.imread(TextPath, cv2.IMREAD_GRAYSCALE)
    m_org, n_org = I.shape
    Z = block_img_8(I.astype(float)/255.0)
    m_z, n_z = Z.shape
    mBlock, nBlock = m_z//8, n_z//8
    VecBlock = im2block8(Z)
    t, iter = 10, 0
    blockImg = np.empty((0, n_org))
    for i in range(mBlock):
        Aux = np.empty((8, 0))
        for j in range(nBlock):
            iter += 1
            y = VecBlock[:, iter-1]
            D_d = decim_mat(D, y)
            x = OMP(y, D_d, t)
            Aux = np.hstack((Aux, np.reshape(np.matmul(D, x), (8, 8))))
        blockImg = np.vstack((blockImg, Aux))
    Y = (blockImg * 255.0).clip(0, 255).astype(np.uint8)
    Y = ndimage.zoom(Y, (m_org/Y.shape[0], n_org/Y.shape[1]), order=1)
    return Y

def im2block8(X):
    m, n = X.shape
    b1, b2 = m//8, n//8
    Y = np.empty((64, b1*b2))
    for i in range(b1):
        for j in range(b2):
            Aux = X[i*8:(i+1)*8, j*8:(j+1)*8]
            Y[:, i*b2+j] = np.reshape(Aux, (64,))
    return Y

def decim_mat(D0, y):
    n, m = D0.shape
    D = np.copy(D0)
    for i in range(n):
        if y[i] == 0:
            D[i, :] = np.zeros((1, m))
    return D

def block_img_8(X):
    m, n = X.shape
    s1, s2 = m % 8, n % 8
    if s1 == s2 == 0:
        m1, n1 = m, n
    elif s1 == 0 and s2 != 0:
        if s2 <= 4:
            m1, n1 = m, n-s2
        else:
            m1, n1 = m, n+8-s2
    elif s1 != 0 and s2 == 0:
        if s1 <= 4:
            m1, n1 = m-s1, n
        else:
            m1, n1 = m+8-s1, n
    elif s1 != 0 and s2 != 0:
        if s1 <= 4:
            m1 = m-s1
        else:
            m1 = m+8-s1
        if s2 <= 4:
            n1 = n-s2
        else:
            n1 = n+8-s2
    Y = ndimage.zoom(X, (m1/m, n1/n), order=1)
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