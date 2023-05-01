import sys
 
#setting path
sys.path.append('../funciones_python')

from GoDecVideoFull import GoDecVideoFull

# Ejemplo de Modelado de Fondo usando el algoritmo GoDec 
# con el  toolbox FroImPro

# ENTRADA: La función GoDecVideoFull recibe un path de un video en formato 
# .mp4, una restricción para el rango r<=min{m,n} y 's' una constante de 
# esparcidad. 

# SALIDA: Tres videos que se generan en la carpeta de trabajo. El video L
# modelo el fondo y el video S modela los objetos en movimiento. El video
# XLS muestra los tres videos juntos para mejor visualización.

# EJEMPLO PARTICULAR

r=2
s=2150000
GoDecVideoFull('ejemplos/3_Modelado_Fondo/cut_video.mp4','ejemplos/3_Modelado_Fondo',r,s,'BRP')
