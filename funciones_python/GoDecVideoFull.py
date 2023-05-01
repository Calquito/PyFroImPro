from Video2Matrix import Video2Matrix
from Matrix2Video import Matrix2Video
from GoDec import GoDec
import numpy as np
import cv2
import os
from double2uint8 import double2uint8


def GoDecVideoFull(TextPath,OutputPath,r, k, Opcion,c=3,IteraMax=100,Tol=2e-6):

    X, m1, n1 = Video2Matrix(TextPath)
    #k = fix(0.075*n1*m1*200);
    LRF, SRF, _ = GoDec(X, r, k, Opcion,c,IteraMax,Tol)
    Matrix2Video(SRF, m1, n1, OutputPath+'/S.mp4')
    Matrix2Video(LRF, m1, n1, OutputPath+'/L.mp4')
    #En vez de utilizar matrices, esta funcion concatena los videos directamente
    MatrixXLS2Video(TextPath,OutputPath+'/S.mp4',OutputPath+'/L.mp4',OutputPath+'/XLS.mp4')


def MatrixXLS2Video(nombrevideo1, nombrevideo2, nombrevideo3,nombreOutput ):
    # Leemos los videos
    video1 = cv2.VideoCapture(nombrevideo1)
    video2 = cv2.VideoCapture(nombrevideo2)
    video3 = cv2.VideoCapture(nombrevideo3)

    # Verificamos que los videos se hayan abierto correctamente
    if not video1.isOpened():
        print("Error al abrir el video 1")
        exit()
    if not video2.isOpened():
        print("Error al abrir el video 2")
        exit()
    if not video3.isOpened():
        print("Error al abrir el video 3")
        exit()

    # Obtenemos la resolución de los videos
    width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Creamos un nuevo video con los frames concatenados
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(nombreOutput, fourcc, 30.0, (width*3, height))

    while True:
        # Leemos un frame de cada video
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        ret3, frame3 = video3.read()

        # Si alguno de los videos llegó al final, salimos del loop
        if not ret1 or not ret2 or not ret3:
            break

        # Concatenamos los frames horizontalmente
        concatenated_frame = cv2.hconcat([frame1, frame2, frame3])

        # Escribimos el frame en el nuevo video
        out.write(concatenated_frame)

    # Liberamos los videos y el nuevo video
    video1.release()
    video2.release()
    video3.release()
    out.release()

