import numpy as np

def RankConstrainedFilterX(A, C, k):
    # Obtener dimensiones de la matriz A
    mA, nA = A.shape

    # Calcular la pseudoinversa de C
    pinvC = np.linalg.pinv(C)

    # Multiplicar A por C y por su pseudoinversa
    Matrix = A @ pinvC @ C

    # Calcular el rango de la matriz resultante
    RankMatrix = np.linalg.matrix_rank(Matrix)

    if k <= RankMatrix:
        # Si el rango es mayor o igual que k, aplicar la descomposición en valores singulares
        U, S, V = np.linalg.svd(Matrix)
        Matrixk = np.zeros((mA, nA))
        for i in range(k):
            Matrixk += S[i] * np.outer(U[:, i], V[i, :])
    else:
        # Si el rango es menor que k, simplemente asignar la matriz resultante
        Matrixk = Matrix

    # Multiplicar la matriz resultante por la pseudoinversa de C
    X = Matrixk @ pinvC

    return X