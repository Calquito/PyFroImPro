import sys
 
#setting path
sys.path.append('../funciones_python')

from FullRankConstrainedFilterX import FullRankConstrainedFilterX
from SaveFilteredImages import SaveFilteredImages

#Problema2_GLRMA_ImageDenoising
# Ejemplo de Eliminación de Ruido usando el problema GLRMA 
# con el  toolbox FroImPro

# ENTRADA: La función FullRankConstrainedFilterX recibe un path de una
# carpeta que contiene imágenes en escala de grises, todas del mismo tamaño. 

# SALIDA: Matriz A cuyas columnas son vectorizaciones de las imágenes recibidas
# y matriz X que corresponde al filtro que será utilizado para limpiar
# ruido de otra imagen.

# EJEMPLO PARTICULAR 

# Parte A del ejemplo: Generar el filtro X.

_,_,X,_,_=FullRankConstrainedFilterX('ejemplos/2_Eliminacion_ruido/trainingData')

# Parte B del ejemplo: Usar el filtro X para limpiar las imágenes almacenadas
# en la carpeta 'NoisyImg' y en la carpeta 'Results' (creada por el
# usuario) la función SaveFilteredImages guarda la imagen con ruido y la 
# imagen limpiada al aplicar el filtro X.

SaveFilteredImages('ejemplos/2_Eliminacion_ruido/NoisyImg','ejemplos/2_Eliminacion_ruido/Results',X)