import cv2
import numpy as np

def Video2Matrix(PathTextVideo, scale=1, numFrames=None):
    # Carga el vídeo y determina el número de frame que posee
    VideoAProcesar = cv2.VideoCapture(PathTextVideo)
    FrameTotales = int(VideoAProcesar.get(cv2.CAP_PROP_FRAME_COUNT))


    
    # Factor para definir el número de frames
    if numFrames is not None:
        FactorFrame = FrameTotales / numFrames
    else:
        FactorFrame = 1
        numFrames = FrameTotales
    
    # Lee el primer frame y lo convierte a escala de grises.
    #El primer valor de VideoAProcesar.read() retorna un booleano para indicar si la operación fue exitosa o no
    _, Ima = VideoAProcesar.read()
    Ima = cv2.resize(cv2.cvtColor(Ima, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    
    # Obtiene las dimensiones de la matriz Ima y redefine la matriz como una columna.
    m1, n1 = Ima.shape
    XO = Ima.flatten().reshape((-1, 1))
    
    # Lee el resto de los frames y los convierte a escala de grises.
    for i in range(2, numFrames):
        # Establece la posición del cuadro de video para leerlo
        VideoAProcesar.set(cv2.CAP_PROP_POS_FRAMES, i*FactorFrame)
        # Lee el cuadro de video y aplica la transformación
        _, Ima = VideoAProcesar.read()
        Ima = cv2.resize(cv2.cvtColor(Ima, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        # Añade la imagen procesada a la matriz de salida
        XO = np.hstack([XO, Ima.flatten().reshape((-1, 1))])
    
    # Retorna la matriz XO y las dimensiones del primer frame.
    return XO, m1, n1