import os
import numpy as np
import cv2
from double2uint8 import double2uint8

import os
import numpy as np
import cv2

def Matrix2Video(X, m, n, PathNameVideo):
    posUltiLinea = max([i for i in range(len(PathNameVideo)) if PathNameVideo.startswith('\\', i)])
    # Encuentra la posición de la última barra invertida en el nombre del archivo
    if posUltiLinea > 1:
        # Si la posición de la última barra invertida es mayor que 1, significa que hay una ruta de acceso en el nombre del archivo
        if not os.path.exists(PathNameVideo[0:posUltiLinea]):
            # Comprueba si la carpeta de la ruta de acceso existe. Si no, la crea.
            os.makedirs(PathNameVideo[0:posUltiLinea])
    # Crea un objeto VideoWriter y abre el archivo de video para escribir
    v = cv2.VideoWriter(PathNameVideo, cv2.VideoWriter_fourcc(*'mp4v'), 30, (n, m), isColor=False)
    # Define el tamaño de cada fotograma del video
    isize = (m, n)
    # Escribe cada fotograma de la matriz en el archivo de video
    for ii in range(X.shape[1]):
        frame = double2uint8(np.reshape(X[:, ii], isize))
        v.write(frame)
    # Cierra el archivo de video
    v.release()