from ReadImageDataBase import ReadImageDataBase
from Matrix2Image import Matrix2Image
from double2uint8 import double2uint8

#Llamado de prueba a la función ReadImageDataBase
A_ReadImageDataBase,m_ReadImageDataBase,n_ReadImageDataBase = ReadImageDataBase('image database')

#Llamado de prueba a la función double2uint8
A_double2uint8 = double2uint8(A_ReadImageDataBase)

#Llamado de prueba a la función Matrix2Image
Matrix2Image(A_ReadImageDataBase,m_ReadImageDataBase,n_ReadImageDataBase,'Matrix2ImageResults','Matrix2ImageResult_')







