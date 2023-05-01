import sys
 
#setting path
sys.path.append('../funciones_python')

from NoiseFunction import NoiseFunction
from Matrix2Image import Matrix2Image

A,C,m,n=NoiseFunction("ejemplos/2_Eliminacion_ruido/trainingData")
Matrix2Image(C,m,n,"ejemplos/2_Eliminacion_ruido/NoisyImg","SatNoise")
