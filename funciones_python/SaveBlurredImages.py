import os
import cv2
import numpy as np
from NoiseFunction import NoiseFunction
from Matrix2Image import Matrix2Image


def SaveBlurredImages(Textpath,pathWriteNoise, NoiseOption='gaussian', sigmagauss=0.01, meangauss=0, d=0.05, sigmaspeckle=0.05):
    C,_,m,n=NoiseFunction(Textpath, NoiseOption, sigmagauss, meangauss, d, sigmaspeckle)
    _, num_img = np.shape(C)[:2]
    Matrix2Image(C,m,n,pathWriteNoise,"BlurredImage")
