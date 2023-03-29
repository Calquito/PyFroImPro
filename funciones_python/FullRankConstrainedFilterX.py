import numpy as np
from ReadImageDataBase import ReadImageDataBase
from NoiseFunction import NoiseFunction
from RankConstrainedFilterX import RankConstrainedFilterX

def FullRankConstrainedFilterX(Textpath,k=None,NoiseOption='gaussian', sigmagauss=0.01, meangauss=0, d=0.05, sigmaspeckle=0.05):
    #obtiene la matriz desde la funcion ReadImageDatabase
    A,m,n = ReadImageDataBase(Textpath)
    #cálculo correcto del parámetro k
    mA,nA = np.shape(A)[:2]
    if(k is None):
        k=min(mA,nA)
    elif(k>min(mA,nA)):
        raise ValueError('The value of "k" is invalid. For your dataset it must satisfy the condition 0<k<='+str(min(mA,nA)))
    #obtener C y X con las funciones respectivas
    C = NoiseFunction(Textpath,NoiseOption,sigmagauss,meangauss,d,sigmaspeckle)
    X = RankConstrainedFilterX(A,C,k)
    return A,C,X,m,n