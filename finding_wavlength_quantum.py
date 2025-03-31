#Finding the wavelength of light using rgb intensity values and the quantum efficency

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

file_name  = "camera_qe_response.txt"
im = cv.imread("Lenna_(test_image).png")
image_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
plt.subplot(111)
plt.imshow(image_rgb)
plt.show()