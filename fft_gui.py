#%%
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

image_name = "fhous.png" 

def fast_fourier_transform(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

image_data = plt.imread(image_name)
image_data = image_data[:,:,:3].mean(axis=2) #forpng

plt.set_cmap("gray")

ft = fast_fourier_transform(image_data)
plt.subplot(121)
plt.imshow(image_data,vmin=0,vmax=1)
plt.colorbar()
plt.axis("off")
plt.subplot(122)
plt.imshow(np.log(abs(ft)))
plt.axis("off")
plt.savefig("hi",dpi=600,bbox_inches='tight')
plt.show()



# %%
