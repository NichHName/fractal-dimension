# Fractal Dimension Repository
Repository of code to estimate the fractal dimension of 1D and 2D noise.

## main.py
The main engine for tests. Uses functions defined in experiments.py.

## experiments.py
Contains the functions used in main.py.
1. `run_1d_convergence_test(Ns=None)`: This function shows that as the length of the signal increases, the average error between the R/S analysis and box-counting methods compared to the theoretical fractal dimension diminishes towards zero.
    - `Ns`: list of signal lengths
2. `run_2d_parameter_sweep(param, shape=(512, 512), num_samples=30)`: Choose a parameter p (persistence), s (scale), or o (octave count) to test how estimated fractal dimension changes with respect to change in the specified parameter.
    - `param`: Parameter to alter
3. `run_2d_convergence_test(target_d=2.5, shape=(512, 512), n=30)`: Give a theoretical dimension (between 2.0 and 3.0) and number of trials to test numerical convergence to the theoretical fractal dimension. Note: Variogram is quite bad at this one.
4. `generate_test_image(shape=(512, 512), scale=100.0, octaves=6, persistence=0.5, color="default")`: Generate a Perlin-noise-generated image with a specified color palette (default, two custom palettes, and any other supported by the noise library)

### signal\_1d.py

Python file containing `box_count_dimension()` and `hurst_rs()`, methods for estimating the fractal dimension of a 1D signal (array, list, etc.).

### noise\_2d.py

Python file containing `power_spectrum()` and `variogram()`, methods for estimating the fractal dimension of a 2D image/array.

### perlin\_gen.py

Python file containing `generate_perlin_2d()`, a method for generating perlin noise of the given size, octaves, and frequency; `array_to_image()`, a method for saving an array of float values to an image w/ color based on magnitude; and `array_to_custom_color_image()`, a custom-color version of `array_to_image()`.

(Note: yes, the last method bases colors off of Minecraft colors, as Perlin noise is the main generator of Minecraft terrain.)