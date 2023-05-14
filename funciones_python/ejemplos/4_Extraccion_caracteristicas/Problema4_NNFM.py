import sys
 
#setting path
sys.path.append('../funciones_python')

from ConstructionNNFM import ConstructionNNFM
from ReconstructionExternalFace import ReconstructionExternalFace

# Ejemplo de Extracción de Características de caras usando 
# el problema NNMF con el  toolbox FroImPro


# ENTRADA: La función ConstructionNNFM recibe un path de una carpeta con imágenes en
# escala de grises, del mismo tamaño, y una restricción para el rango r<=min{m,n}. 

# SALIDA: Matrices X1 y X2, donde A\approx X1*X2 y A es la matriz original
# de imágenes vectorizadas. Además, X1 representa la base de caras.
# EJEMPLO PARTICULAR COMPLEMENTARIO AL ARTÍCULO:

X1,X2=ConstructionNNFM('ejemplos/4_Extraccion_caracteristicas/database','ejemplos/4_Extraccion_caracteristicas',285)

ReconstructionExternalFace('ejemplos/4_Extraccion_caracteristicas/f1.jpg','ejemplos/4_Extraccion_caracteristicas',X1,'f2.jpg')