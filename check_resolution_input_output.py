import os
import matplotlib.pyplot as plt
import pydicom
import cv2 as cv
import numpy as np

output_folder = 'output'

filenames = os.listdir(output_folder)


def read_ct(path_):
    dicom = pydicom.dcmread(path_)
    pixels = dicom.pixel_array
    return pixels


for filename in filenames:
    # input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # input_pixels = read_ct(input_path)
    output_pixels = read_ct(output_path)
    # output_pixels = output_pixels[:320, :400]
    # output_pixels = (output_pixels / output_pixels.max()) * 255
    # output_pixels = np.dstack((output_pixels, output_pixels, output_pixels))
    #
    # cv.rectangle(output_pixels, (0, 0), (120, 80), (255, 255, 255), -1)
    # cv.rectangle(output_pixels, (5, 5), (15, 15), (255, 0, 0), -1)
    # cv.putText(output_pixels, 'Stomach', (16, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # cv.rectangle(output_pixels, (5, 30), (15, 40), (0, 0, 255), -1)
    # cv.putText(output_pixels, 'Large bowel', (16, 41), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # cv.rectangle(output_pixels, (5, 55), (15, 65), (0, 255, 0), -1)
    # cv.putText(output_pixels, 'Small bowel', (16, 66), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # plt.subplot(1, 2, 1)
    # plt.imshow(input_pixels, cmap='gray')
    plt.imshow(output_pixels, cmap='gray')
    plt.show()




