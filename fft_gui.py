#%%
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image_name = "kodim17.png" 
mask_name = "fhouse edited.png"

def fast_fourier_transform(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def inverse_fourier_transform(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.ifft2(ft)
    return np.fft.fftshift(ft)

def normalise_data(input):
    max_val  = np.amax(input)
    return(input/max_val)


image_data = cv.imread(image_name)
image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
image_data_gray = image_data[:,:,:3].mean(axis=2) #forpng
image_data_gray = normalise_data(image_data_gray)

mask_data = cv.imread(mask_name)
mask_data = mask_data[:,:,:3].mean(axis=2)
plt.set_cmap("gray")


ft = fast_fourier_transform(image_data_gray)
ift = inverse_fourier_transform(ft)
plt.subplot(151)
plt.title("Original")
plt.axis("off")
plt.imshow(image_data)
plt.subplot(152)
plt.imshow(image_data_gray,vmin=0)
plt.axis("off")
plt.title("Grayscale")

plt.subplot(153)
plt.axis("off")
plt.title("Mask")
plt.imshow(np.zeros((20,20))) #For maskless
plt.subplot(154)
plt.imshow(normalise_data(np.log(abs(ft))),vmin=0)
plt.axis("off")
plt.title("FT")
plt.subplot(155)
plt.title("IFT")
plt.imshow((abs(ift)),vmin=0)
print(np.median(abs(ift),axis=None))
#plt.imshow(np.clip(abs(ift)*5,0,1))
plt.axis("off")
plt.savefig("hi",dpi=1000,bbox_inches='tight')

plt.show()


