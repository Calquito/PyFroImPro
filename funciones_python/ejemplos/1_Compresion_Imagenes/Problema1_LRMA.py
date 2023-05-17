

import sys
 
#setting path
sys.path.append('../funciones_python')

from ImageCompression  import ImageCompression
#from ImageCompression import ImageCompression

# Ejemplo de Compresión de Imágenes usando el problema LRMA 
# con el  toolbox FroImPro

# ENTRADA: La función ImageCompression recibe un path de una imagen, un 
# radio de comprensión y una etiqueta 'BRP' o 'SVD', para que la función 
# se ejecute con la SVD o con el método MBRP.

# SALIDA: La función crea una carpeta 'funciones_python/Results' que contiene la imagen
# recibida en escala de grises y su representación matricial almacenada en
# el archivo L.txt, la imagen comprimida, las matrices A y B (corresponden 
# a D y C en el artículo, respectivamente) almacenadas en archivos de texto. 

# EJEMPLO PARTICULAR

ImageCompression('ejemplos/1_Compresion_Imagenes/montanas.jpg',0.1,'BRP','ejemplos/1_Compresion_Imagenes/results/')


