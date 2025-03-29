#%%
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import cmath

image_name = "Lenna_(test_image).png" 

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


image_data = plt.imread(image_name)
image_data = image_data[:,:,:3].mean(axis=2) #forpng

plt.set_cmap("gray")


ft = fast_fourier_transform(image_data)
ift = inverse_fourier_transform(ft)

plt.subplot(131)
plt.imshow(image_data,vmin=0,vmax=1)
plt.axis("off")
plt.title("Original Image")
plt.subplot(132)
plt.imshow(normalise_data(np.log(abs(ft))),vmin=0,vmax=1)
plt.axis("off")
plt.title("FT")
plt.subplot(133)
plt.title("IFT")
plt.imshow(normalise_data(abs(ift)),vmin=0,vmax=1)
#plt.imshow(np.clip(abs(ift)*5,0,1))
plt.axis("off")
plt.savefig("hi",dpi=1000,bbox_inches='tight')

plt.show()



# %%
