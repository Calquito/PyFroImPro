from ReadImageDataBase import ReadImageDataBase
from GoDec import GoDec
from Matrix2Image import Matrix2Image
import numpy as np

def GoDecImageFull(TextPath,r,k,opcion,c=3,IteraMax=100,Tol=1e-6):
    X,m,n=ReadImageDataBase(TextPath)
    L,S,_=GoDec(X,r,k,opcion,c,IteraMax,Tol)
    Matrix2Image(S,m,n,'ImagesForS','S')
    Matrix2Image(L,m,n,'ImagesForL','L')



