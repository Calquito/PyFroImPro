import os
from typing import List
from scipy.io import loadmat
from skimage import io
import argparse
from ReconstructionExternalFace import ReconstructionExternalFace

def ReconstructionExternalFacesDirectory(Textpath, Results_Texthpath, W , Extension='.jpg', NameOutput='Reconstructed'):
    expectedExtensions = ['.jpg','.pgm','.bmp','.png','.tif']
    if Extension.lower().strip() not in expectedExtensions:
        raise ValueError(f"Invalid extension '{Extension}'")

    if not os.path.isdir(Textpath):
        raise ValueError(f"Invalid directory '{Textpath}'")

    i = 0
    for file in os.listdir(Textpath):
        file_path = os.path.join(Textpath, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file)
            if ext.lower() == Extension.lower() or Extension == '':
                i += 1
                ReconstructionExternalFace(file_path,Results_Texthpath,W, f"{NameOutput}{i}")
    
    if i == 0:
        raise ValueError(f"There are no images in the data base with extension '{Extension}'")

