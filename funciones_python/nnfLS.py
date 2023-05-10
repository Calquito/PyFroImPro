import numpy as np
from numpy.linalg import norm

def nnfLS(A, r=None, IteraMax=None, Tol=None):
    mA, nA = A.shape  # Obtener las dimensiones de la matriz A
    Min = min(mA, nA)  # Valor mínimo entre las dimensiones
    r = r or (Min // 2)  # Si r no se proporciona, se establece como la mitad del valor mínimo
    IteraMax = IteraMax or (2 * nA)  # Si IteraMax no se proporciona, se establece como 2 veces el número de columnas de A
    Tol = Tol or (10 ** -6)  # Si Tol no se proporciona, se establece como 10^-6
    
    # Validar los argumentos
    assert isinstance(A, np.ndarray) and A.ndim == 2 and np.issubdtype(A.dtype, np.number), "A debe ser una matriz numérica de dos dimensiones"
    assert isinstance(r, int) and (1 <= r <= Min), "r debe ser un entero entre 1 y la dimensión más pequeña de A"
    assert isinstance(IteraMax, int) and (IteraMax >= 1), "IteraMax debe ser un entero positivo"
    assert isinstance(Tol, (float, int)) and (Tol > 0), "Tol debe ser un número positivo"
    
    # Código de la función
    W = np.random.rand(mA, r)
    H = np.random.rand(r, nA)
    VectorError = []
    contador = 0
    Error = np.inf
    while (contador <= IteraMax) and (Error > Tol):
        contador += 1
        H=H*(W.T @ A)/(W.T @ W @H)
        W=W*(A @ H.T)/(W @ H @H.T)
        Error = norm(A - W @ H, 'fro') / np.sqrt(mA * nA)
        VectorError.append(Error)
        
    return W, H, VectorError