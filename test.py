import numpy as np
import matplotlib.pyplot as plt

def visualize_ifft(N=1000, freq_x=20, freq_y=20):
    """
    Visualizes the spatial pattern created by a single DFT pixel.
    :param N: Size of the image (NxN)
    :param freq_x: Frequency component in the x-direction
    :param freq_y: Frequency component in the y-direction
    """
    # Create an empty DFT image
    dft_image = np.zeros((N, N), dtype=np.complex64)
    
    # Set a single frequency component
    dft_image[freq_y, freq_x] = 1
    
    # Compute the inverse DFT
    ifft_image = np.fft.ifft2(dft_image)
    ifft_image = np.fft.fftshift(ifft_image)  # Shift for visualization
    ifft_magnitude = np.abs(ifft_image)  # Get magnitude
    
    # Normalize for display
    ifft_magnitude /= np.max(ifft_magnitude)
    
    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.imshow(ifft_magnitude, cmap='gray')
    plt.title(f"Spatial Pattern from Single DFT Pixel at ({freq_x}, {freq_y})")
    plt.axis("off")
    plt.show()

# Run visualization with example values
visualize_ifft(N=1000, freq_x=10, freq_y=0)