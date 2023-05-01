import os
import numpy as np
from ReadImageDataBase import ReadImageDataBase
from Matrix2Image import Matrix2Image
from double2uint8 import double2uint8
from Video2Matrix import Video2Matrix
from Matrix2Video import Matrix2Video
from img_miss_pix import missing_pixels
from RankConstrainedFilterX import RankConstrainedFilterX
from NoiseFunction import NoiseFunction
from LowRankMatrixBRP import LowRankMatrixBRP
from ImageCompression import ImageCompression
from GoDecVideoFull import GoDecVideoFull


#Llamado de prueba a la función ReadImageDataBase
#A_ReadImageDataBase,m_ReadImageDataBase,n_ReadImageDataBase = ReadImageDataBase('ImageDatabase')
#np.set_printoptions(precision=5, threshold=np.inf)
#print(A_ReadImageDataBase)

#Llamado de prueba a la función double2uint8
#A_double2uint8 = double2uint8(A_ReadImageDataBase)

#Llamado de prueba a la función Matrix2Image
#Matrix2Image(A_ReadImageDataBase,m_ReadImageDataBase,n_ReadImageDataBase,'Matrix2ImageResults','Matrix2ImageResult_')

#Llamado de prueba a la función video2Matrix
#XO_video2Matrix, m1, n1 = Video2Matrix('Video2MatrixExample/video.mp4')

#Llamado de prueba a la función video2Matrix con scale y numFrames
#XO_video2Matrix, m1, n1 = Video2Matrix('Video2MatrixExample/video.mp4',0.1,100)

#np.set_printoptions(precision=6, threshold=np.inf)
#print(XO_video2Matrix)

#Llamado de prueba a la función Matrix2Video, misma carpeta
#Matrix2Video(XO_video2Matrix,m1,n1,'./Matrix2VideoResult.mp4')

#Llamado de prueba a la función Matrix2Video, distinta carpeta
#Matrix2Video(XO_video2Matrix,m1,n1,'Matrix2VideoResults/Matrix2VideoResult.mp4')

#Llamado de prueba a la función missing_pixels
#missing_pixels('ImageDatabase/tree.jpg',0.15)

#Llamado de prueba a la función missing_pixels
#Matriz A
'''A = np.array([[-1, 1, 2],
              [3, 0, -1],
              [2, -2, 3],
              [4, 0, 3]])

# Matriz C
C = np.array([[-2, 1, 6],
              [-2, 4, 5]])
k=2
print(RankConstrainedFilterX(A,C,k))'''

#Llamado de prueba a la función NoiseFunction

#A,C,m,n=NoiseFunction("ImageDatabaseSmall")
#np.set_printoptions(precision=3, threshold=np.inf)
#print(C)

#Llamado de prueba a la función LowRankMatrixBRP
#L = np.array([[-1, 1, 2],[3, 0, -1],[2, -2, 3],[4, 0, 3]])
#A,B = LowRankMatrixBRP(L)

#Llamado de prueba a la función ImageCompression
#ImageCompression('ImageDatabase/tree.jpg',0.1,'BRP')

r=2
s=2150000
GoDecVideoFull('video.mp4',r,s,'BRP')


