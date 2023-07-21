import numpy as np
import pydicom 
from pathlib import Path
import os



filenames = os.listdir('.')

dicom = pydicom.dcmread('I71')

for filename in filenames:
    if '.npy' in filename:
        new_pixels = np.load(filename) 
        dicom.PixelData = np.array(new_pixels[:, :, 0], np.int16).tobytes()
        dicom.Rows = int(new_pixels.shape[0])
        dicom.Columns = int(new_pixels.shape[1])
        dicom.save_as(filename[:-4])

