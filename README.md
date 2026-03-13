# Fractal Dimension Repository
Repository of code to estimate the fractal dimension of 1D and 2D noise.

## fracdim.ipynb

A Jupyter Notebook serving as the main test file for all fractal dimension calculations.

### signal_1d.py

Python file containing $\texttt{box_count_dimension()}$ and $\texttt{hurst_rs()}$, methods for estimating the fractal dimension of a 1D signal (array, list, etc.).

### noise_2d.py

Python file containing $\texttt{power_spectrum()}$ and $\texttt{variogram()}$, methods for estimating the fractal dimension of a 2D image/array.

### perlin_gen.py

Python file containing $\texttt{generate_perlin_2d()}$, a method for generating perlin noise of the given size, octaves, and frequency; $\texttt{array_to_image()}$, a method for saving an array of float values to an image w/ color based on magnitude; and $\texttt{array_to_custom_color_image()}$, a custom-color version of $\texttt{array_to_image()}$.

(Note: yes, the last method bases colors off of Minecraft colors, as Perlin noise is the main generator of Minecraft terrain.)