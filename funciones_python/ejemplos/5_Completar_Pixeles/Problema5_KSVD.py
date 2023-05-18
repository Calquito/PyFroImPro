import sys
 
#setting path
sys.path.append('../funciones_python')

from ksvd import ksvd
from clean_ksvd import clean_ksvd
from PIL import Image
from double2uint8 import double2uint8
import numpy as np

#EJEMPLO PARTICULAR
X1,X2,Err=ksvd('ejemplos/5_Completar_Pixeles/dataset_jpg',15,2,'.jpg')

#RECONTRUCCIÓN IMAGEN CON PIXELES PERDIDOS
Img_Rec = clean_ksvd('ejemplos/5_Completar_Pixeles/img_pix.pgm',X1)

#GENERAR IMAGEN
im = Image.fromarray(double2uint8(Img_Rec))
im.save('ejemplos/5_Completar_Pixeles/img_pix_filled.jpg')

#1 minuto 28 segundos