from PIL import Image
import os

def convert_pgm_to_jpg(input_folder, output_folder):
    # Crea la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recorre todos los archivos de la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith('.pgm'):
            # Carga la imagen PGM
            img_path = os.path.join(input_folder, filename)
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img.load()

            # Guarda la imagen en formato JPG
            new_filename = filename[:-4] + '.jpg'
            new_img_path = os.path.join(output_folder, new_filename)
            with open(new_img_path, 'wb') as f:
                img.convert('RGB').save(f, 'JPEG')


input_folder = 'ejemplos/5_Completar_Pixeles'
output_folder = 'ruta'
convert_pgm_to_jpg(input_folder, output_folder)