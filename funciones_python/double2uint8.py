import numpy as np

def double2uint8(X):
    # Encontrar el valor mínimo y máximo de la matriz X
    mini = np.min(X)
    maxi = np.max(X)
    
    # Escalar los valores de la matriz X al rango de 0 a 255 de forma lineal
    # utilizando la fórmula: (X - min(X)) * 255 / (max(X) - min(X))
    # y luego convertir los valores escalados a uint8
    OutX = np.uint8(255 / (maxi - mini) * (X - mini))
    
    # Retornar la matriz resultante de tipo uint8
    return OutX