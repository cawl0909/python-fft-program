import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if not file_path:
        return
    
    global img, img_cv
    img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img_cv)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

def draw_mask(event):
    x, y = event.x, event.y
    cv2.circle(mask, (x, y), 10, 255, -1)
    update_mask_display()

def update_mask_display():
    mask_img = Image.fromarray(mask)
    mask_img = ImageTk.PhotoImage(mask_img)
    mask_panel.config(image=mask_img)
    mask_panel.image = mask_img

def create_mask():
    global mask
    if img_cv is None:
        messagebox.showerror("Error", "No image loaded!")
        return
    mask = np.zeros_like(img_cv, dtype=np.uint8)
    update_mask_display()

def apply_fourier_transform():
    if img_cv is None:
        messagebox.showerror("Error", "No image loaded!")
        return
    
    f = np.fft.fft2(img_cv)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    if mask is not None:
        fshift = fshift * (mask / 255)
        f_ishift = np.fft.ifftshift(fshift)
        img_reconstructed = np.fft.ifft2(f_ishift)
        img_reconstructed = np.abs(img_reconstructed)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(131), plt.imshow(img_cv, cmap='gray')
    plt.title("Original Image"), plt.axis("off")
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum"), plt.axis("off")
    if mask is not None:
        plt.subplot(133), plt.imshow(img_reconstructed, cmap='gray')
        plt.title("Filtered Image"), plt.axis("off")
    plt.show()

root = tk.Tk()
root.title("2D Fourier Transform GUI")

frame = tk.Frame(root)
frame.pack(pady=10)

btn_open = tk.Button(frame, text="Open Image", command=open_image)
btn_open.pack(side=tk.LEFT, padx=10)

btn_mask = tk.Button(frame, text="Create Mask", command=create_mask)
btn_mask.pack(side=tk.LEFT, padx=10)

btn_transform = tk.Button(frame, text="Apply Fourier Transform", command=apply_fourier_transform)
btn_transform.pack(side=tk.LEFT, padx=10)

panel = tk.Label(root)
panel.pack(pady=10)

mask_panel = tk.Label(root)
mask_panel.pack(pady=10)
mask_panel.bind("<B1-Motion>", draw_mask)

img = None
img_cv = None
mask = None

root.mainloop()