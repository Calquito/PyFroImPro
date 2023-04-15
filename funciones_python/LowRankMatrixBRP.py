import numpy as np
import math

def LowRankMatrixBRP(L,r=None,c=3):
    m,n = L.shape
    if r is None:
        r=math.floor(min(m, n) / 2)

    #validar argumentos
    ValidC = lambda x: isinstance(x, int) and (x >= 2) and (x <= 10)
    ValidMatrix = lambda x: isinstance(x, np.ndarray) and np.isnumeric(x)
    ValidR = lambda x: isinstance(x, int) and (x >= 1) and (x <= min(L.shape))

    Y2 = np.random.randn(n, r)  # Generar matriz aleatoria Y2 de tamaño n x r con valores normalmente distribuidos

    for i in range(c + 1):
        Y1 = L @ Y2  # Multiplicar L por Y2
        Y2 = L.T @ Y1  # Multiplicar la transpuesta de L por Y1

    Qr, _ = np.linalg.qr(Y2, mode='reduced')  # Realizar la descomposición QR reducida de Y2

    A = L @ Qr  # Calcular la matriz A como el producto de L y Qr
    B = Qr.T  # Calcular la matriz B como la transpuesta de Qr

    return A, B
