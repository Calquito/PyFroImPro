import numpy as np
from scipy.sparse.linalg import svds
from imgs2blocks import imgs2blocks

def ksvd(TextPath, r, c, Extension='.jpg', IteraMax=2000, Tol=1e-4):
    # Validaciones
    expectedExtensions = ['.jpg', '.pgm', '.bmp', '.png', '.tif']
    if Extension not in expectedExtensions:
        raise ValueError(f"Invalid extension. It must be one of: {', '.join(expectedExtensions)}")
    if not isinstance(r, int) or r < 1:
        raise ValueError("Invalid value for r. It must be a positive integer.")
    if not isinstance(c, int) or c < 1:
        raise ValueError("Invalid value for c. It must be a positive integer.")
    
    # Cargar imágenes en bloques
    Y = imgs2blocks(TextPath, Extension)
    m = Y.shape[0]
    if not (isinstance(c, int) and c >= 1 and c <= m):
        raise ValueError(f"The value of c is invalid. It must be integer and satisfy 1 <= c <= {m}")
    
    # Inicializar D y otras variables
    D = np.random.rand(Y.shape[0], r)
    Err = np.inf
    e_old = np.inf
    
    # Algoritmo K-SVD
    for it in range(IteraMax):
        # Actualizar X
        X = sparce_matrix(Y, D, c)
        
        # Actualizar D
        R = Y - np.dot(D, X)
        for k in range(r):
            I = np.nonzero(X[k, :])[0]
            if len(I) == 0:
                continue
            Ri = R[:, I] + np.outer(D[:, k], X[k, I])
            U, S, V = svds(Ri, k=1, which='LM')
            D[:, k] = U.flatten()
            X[k, I] = S * V.flatten()
            R[:, I] = Ri - np.outer(D[:, k], X[k, I])
        
        # Calcular error y verificar convergencia
        e_new = np.linalg.norm(Y - np.dot(D, X), 'fro')
        Err = abs(e_old - e_new) / abs(e_new)
        if it > 0 and Err < Tol:
            break
        e_old = e_new
    
    return D, X, Err


def sparce_matrix(Y, D0, c):
    """
    Calcula la matriz de esparcidad X
    
    Args:
    Y: matriz de tamaño MxN
    D0: diccionario de tamaño MxK
    c: parámetro de esparsidad
    
    Returns:
    X: matriz de esparcidad de tamaño KxN
    """
    N = Y.shape[1]
    K = D0.shape[1]
    X = np.zeros((K, N))
    for i in range(N):
        X[:, i] = OMP(Y[:, i], D0, c)
    return X


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