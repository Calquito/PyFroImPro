import cv2
import numpy as np
from skimage.transform import resize

def clean_ksvd1(TextPath, D):
    # Lee la imagen y la convierte en una matriz Z
    I = cv2.imread(TextPath, cv2.IMREAD_GRAYSCALE)
    [m_org, n_org] = I.shape
    Z = cv2.normalize(block_img_8(I.astype('double')), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # Divide la matriz Z en bloques de tamaño 8x8 y los coloca en una matriz VecBlock
    [m_z, n_z] = Z.shape
    mBlock = int(m_z/8)
    nBlock = int(n_z/8)
    VecBlock = im2block8(Z)
    for i in range(mBlock):
        for j in range(nBlock):
            y = np.reshape(Z[i*8:(i+1)*8, j*8:(j+1)*8], (64, 1), order="F")
            D_d = decim_mat(D, y)
            x = OMP(y, D_d, 10)
            Aux = np.reshape(D @ x, (8, 8), order="F")
            VecBlock[:, i*nBlock+j] = np.reshape(Aux, (64,))
    
    # Reconstruye la imagen a partir de los bloques limpios
    blockImg = np.zeros((mBlock*8, nBlock*8))
    for i in range(mBlock):
        for j in range(nBlock):
            Aux = np.reshape(VecBlock[:, i*nBlock+j], (8, 8), order="F")
            blockImg[i*8:(i+1)*8, j*8:(j+1)*8] = Aux
    Y = cv2.normalize(blockImg, None, 0, 255, cv2.NORM_MINMAX)
    Y = cv2.resize(Y.astype("uint8"), (n_org, m_org))
    return Y

def im2block8(X):
    # Convierte una matriz X de tamaño m x n en una matriz Y de tamaño 64 x mn/64,
    # donde cada columna de Y vienen de un bloque vectorizado de tamaño 8 x 8 de X
    [m, n] = X.shape
    b1 = int(m/8)
    b2 = int(n/8)
    Y = np.zeros((64, b1*b2))
    for i in range(b1):
        for j in range(b2):
            Aux = X[i*8:(i+1)*8, j*8:(j+1)*8]
            Y[:, i*b2+j] = np.reshape(Aux, (64,), order="F")
    return Y

def decim_mat(D0, y):
    """
    Esta función recibe una matriz D0 de tamaño (nxm) y un vector y de tamaño n, y devuelve una matriz D 
    del mismo tamaño que D0, donde las filas cuyo valor correspondiente en el vector y es cero, se reemplazan 
    por una fila de ceros en la matriz D.
    """
    n = len(y)
    D = D0.copy()
    m = D0.shape[1]
    for i in range(n):
        if y[i] == 0:
            D[i, :] = np.zeros(m)
    return D

def block_img_8(X):
    """
    Esta función toma una matriz X de tamaño m x n y devuelve una matriz Y
    de tamaño m1 x n1 donde cada bloque de Y es de tamaño 8 x 8 de X.

    Parámetros
    ----------
    X : numpy.ndarray
        La matriz de entrada de tamaño m x n.

    Returns
    -------
    Y : numpy.ndarray
        La matriz de salida de tamaño m1 x n1 donde cada bloque de Y es de tamaño 8 x 8 de X.
    """

    m, n = X.shape
    s1, s2 = m % 8, n % 8

    if s1 == 0 and s2 == 0:
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

    Y = resize(X, (m1, n1), anti_aliasing=True)

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

        try:
            for j, index in enumerate(T):
                x[index] = xaux[j]
        except:
            None
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
    print("asd")
    m = A.shape[0]
    n1 = len(T)
    B = np.zeros((m, n1))
    try:
        for i in range(n1):
            B[:,i] = A[:,T[i]]
    except:
        None
    return B