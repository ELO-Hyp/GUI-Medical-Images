import pydicom
import numpy as np

dicom = pydicom.dcmread('test_imgs\\I14')

pixels_old = dicom.pixel_array

new_rows = 1000
new_cols = 1000
dicom.Rows = new_rows
dicom.Columns = new_cols


dicom.PixelData = np.zeros((new_rows, new_cols), np.int16)

dicom.save_as("test.dcm")


new_dicom = pydicom.dcmread('test.dcm')
print(new_dicom.pixel_array.shape)