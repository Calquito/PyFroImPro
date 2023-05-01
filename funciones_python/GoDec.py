import numpy as np
from scipy.sparse import csc_matrix
from LowRankMatrixBRP import LowRankMatrixBRP

import numpy as np
from scipy.sparse import coo_matrix
from LowRankMatrixBRP import LowRankMatrixBRP

def GoDec(XMatrix, r, k, Opcion):
    Bandera = False
    m, n = XMatrix.shape
    if m < n:
        XMatrix = XMatrix.T
        m, n = XMatrix.shape
        Bandera = True

    S = np.zeros((m, n))
    t = 0
    L = XMatrix
    Errors = [np.inf]
    NormaX2 = np.linalg.norm(XMatrix, 'fro')**2
    ValorError = np.inf
    Tol = 1e-6
    IteraMax = 500

    if Opcion == 'BRP':
        while ValorError > Tol and t < IteraMax:
            t += 1
            A, B = LowRankMatrixBRP(XMatrix - S)
            L = np.dot(A, B)

            T = XMatrix - L  # Inicia la actualización de S
            idx = np.argsort(np.abs(T.ravel()))[::-1]
            S = np.zeros((m, n))
            S.ravel()[idx[:k]] = T.ravel()[idx[:k]]

            T = XMatrix - L - S  # Inicia el cálculo del nuevo error
            ErrorF = np.linalg.norm(T, 'fro')**2 / NormaX2
            T.ravel()[idx[:k]] = 0

            Errors.append(ErrorF)
            ValorError = abs((Errors[-1] - Errors[-2]))

    elif Opcion == 'SVD':
        while ValorError > Tol and t < IteraMax:
            t += 1
            U, Sigma, VT = np.linalg.svd(XMatrix - S, full_matrices=False)
            U = U[:, :r]
            VT = VT[:r, :]
            Sigma = np.diag(Sigma[:r])
            L = np.dot(np.dot(U, Sigma), VT)

            T = XMatrix - L  # Inicia la actualización de S
            idx = np.argsort(np.abs(T.ravel()))[::-1]
            S = np.zeros((m, n))
            S.ravel()[idx[:k]] = T.ravel()[idx[:k]]

            T = XMatrix - L - S  # Inicia el cálculo del nuevo error
            ErrorF = np.linalg.norm(T, 'fro')**2 / NormaX2
            T.ravel()[idx[:k]] = 0

            Errors.append(ErrorF)
            ValorError = abs((Errors[-1] - Errors[-2]))

    if Bandera:
        LF = L.T
        SF = S.T
    else:
        LF = L
        SF = S

    Errors = Errors[1:]

    return LF, SF, Errors


def LowRankMatrixBRPSinValidacion(L, r, c):
    n = L.shape[1]
    Y2 = np.random.randn(n, r)

    for i in range(c+1):
        Y1 = L @ Y2
        Y2 = L.T @ Y1

    Qr, _ = np.linalg.qr(Y2)
    A = L @ Qr
    B = Qr.T

    return A, B