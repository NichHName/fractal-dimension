import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from noise import pnoise2

# ==========================================
# 1. PERLIN NOISE & IMAGE GENERATION
# ==========================================

def generate_perlin_2d(shape, scale=100.0, octaves=1, persistence=0.5, lacunarity=2.0, seed=67):
    """Generates a 2D array of Perlin noise."""
    arr = np.zeros(shape)
    # Using an offset to simulate seeding
    offset = seed * 100 
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i][j] = pnoise2(
                (i + offset) / scale, 
                (j + offset) / scale, 
                octaves=octaves, 
                persistence=persistence, 
                lacunarity=lacunarity, 
                repeatx=1024, 
                repeaty=1024, 
                base=0
            )
            
    # Normalized return
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    # # Non-normalized return:
    # return arr

def array_to_image(array, filename="perlin_noise.png", cmap="magma"):
    """Saves a 2D array as an image."""
    plt.imsave(filename, array, cmap=cmap)
    print(f"Image successfully saved as '{filename}'")

def array_to_custom_color_image(array, filename="colored_terrain.png"):
    """
    Maps normalized 2D array values (0.0 to 1.0) to specific RGB colors.
    """
    # Get the dimensions of your 2D array
    h, w = array.shape
    
    # Create an empty 3D array for the RGB channels (height, width, 3)
    rgb_image = np.zeros((h, w, 3))
    
    # Define your color thresholds and RGB values (0.0 to 1.0 format)
    # Examples:
    # Water: Deep blue to light blue
    # Sand: Tan/Yellow
    # Grass: Green
    # Mountain: Gray
    # Snow: White
    
    # Apply colors based on magnitude conditions
    # Water
    rgb_image[array < 0.35] = [0.1, 0.3, 0.6]  # Deep Water
    rgb_image[(array >= 0.35) & (array < 0.45)] = [0.2, 0.5, 0.8]  # Shallow Water
    
    # Sand
    rgb_image[(array >= 0.45) & (array < 0.50)] = [0.9, 0.8, 0.5]  # Sand
    
    # Land
    rgb_image[(array >= 0.50) & (array < 0.70)] = [0.2, 0.6, 0.2]  # Grass
    rgb_image[(array >= 0.70) & (array < 0.85)] = [0.4, 0.4, 0.4]  # Rock/Mountain
    
    # Peaks
    rgb_image[array >= 0.85] = [0.95, 0.95, 0.95]  # Snow
    
    # Save the resulting RGB array as an image
    plt.imsave(filename, rgb_image)
    print(f"Colored image successfully saved as '{filename}'")

def array_to_purple_black_image(array, filename="purple_black_noise.png"):
    """
    Maps a normalized 2D array to a continuous gradient from Black (0.0) to Purple (1.0).
    """
    # Define our maximum purple color in RGB (0.0 to 1.0 format)
    # A standard deep, vibrant purple is roughly [0.5, 0.0, 0.5]
    # Matplotlib's 'Purples' maxes out closer to [0.29, 0.08, 0.53]
    max_purple = np.array([0.3, 0.0, 0.5]) 
    
    # Use NumPy broadcasting to calculate the gradient instantly.
    # array[:, :, np.newaxis] turns your (h, w) array into (h, w, 1).
    # Multiplying it by the (3,) max_purple array automatically scales the color
    # perfectly for every single pixel, creating an (h, w, 3) RGB image.
    rgb_image = array[:, :, np.newaxis] * max_purple
    
    # Note: If you want to REVERSE the fade (so 0.0 is Purple and 1.0 is Black),
    # change the line above to: rgb_image = (1.0 - array[:, :, np.newaxis]) * max_purple
    
    # Save the resulting RGB array as an image
    plt.imsave(filename, rgb_image)
    print(f"Gradient image successfully saved as '{filename}'")