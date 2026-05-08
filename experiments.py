import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from signal_1d import hurst_rs, box_count_dimension
from noise_2d import power_spectrum, variogram
from perlin_gen import (
    generate_perlin_2d, 
    array_to_image, 
    array_to_custom_color_image, 
    array_to_purple_black_image
)

def run_1d_convergence_test(Ns=None):
    """Runs the 1D Box-Counting and R/S analysis across varying signal lengths."""
    if Ns is None:
        Ns = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    boxerrs, rserrs, avgerrs = [], [], []

    for N in tqdm(Ns, desc="1D Signal Testing"):
        signal = np.random.normal(0, 10, N)
        motion = np.cumsum(signal)

        boxdim, _, _ = box_count_dimension(motion)
        hurstrs = hurst_rs(signal)
        
        boxerr = 1.5 - boxdim
        rserr = 0.5 - hurstrs

        boxerrs.append(boxerr)
        rserrs.append(rserr)
        avgerrs.append((boxerr + rserr) / 2)

        print(f"\n[N={N}] Box-count err: {boxerr:.4f} | R/S err: {rserr:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, boxerrs, label='Box-Count Error', marker='o', linestyle='-', color='blue')
    plt.plot(Ns, rserrs, label='R/S Error', marker='s', linestyle='--', color='red')
    plt.plot(Ns, avgerrs, label='Average Error', marker='x', color='green')
    plt.xscale('log', base=2) # Highly recommended for Ns testing!
    plt.xlabel('Number of Points (N)')
    plt.ylabel('Error Margin')
    plt.title('1D Fractal Dimension Estimation Error vs. Signal Length')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

def run_2d_parameter_sweep(param, shape=(512, 512), num_samples=30):
    """Runs the 2D Power Spectrum and Variogram tests across varying parameters."""
    x_vals, psvals, vvals = [], [], []
    param_names = {'p': 'Persistence', 's': 'Scale', 'o': 'Octaves'}

    if param == 'p':
        x_vals = np.linspace(0, 1, 11)
        desc = "Varying Persistence"
    elif param == 's':
        x_vals = np.linspace(10, 100, 10)
        desc = "Varying Scale"
    elif param == 'o':
        x_vals = list(range(6, 16))
        desc = "Varying Octaves"
    else:
        print("Invalid sweep parameter.")
        return

    print(f"\n--- Estimating FD ({param_names[param]}), {num_samples} samples/val ---")
    
    for val in tqdm(x_vals, desc=desc):
        ps_temp, v_temp = [], []
        for _ in range(num_samples):
            # Assign parameters dynamically based on the current sweep
            p = val if param == 'p' else 0.5
            s = val if param == 's' else 50.0
            o = int(val) if param == 'o' else 6
            
            im = generate_perlin_2d(shape, scale=s, octaves=o, persistence=p)
            ps_temp.append(power_spectrum(im)[0])
            v_temp.append(variogram(im))
            
        psvals.append(np.mean(ps_temp))
        vvals.append(np.mean(v_temp))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, psvals, marker='o', linestyle='-', color='b', label='Power Spectrum')
    plt.plot(x_vals, vvals, marker='s', linestyle='--', color='r', label='Variogram')
    plt.title(f'Estimated Fractal Dimension vs. {param_names[param]} (Avg of {num_samples} runs)', fontsize=14)
    plt.xlabel(param_names[param], fontsize=12)
    plt.ylabel('Estimated Fractal Dimension (D)', fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=12)
    plt.show()

def run_2d_convergence_test(target_d=2.5, shape=(512, 512), n=30):
    """Tests how the estimates stabilize as sample size increases for a specific target dimension."""
    
    # 1. The Mathematical Conversion
    H = 3.0 - target_d
    p = 2.0 ** (-H)
    
    print(f"\n--- Convergence Test: Target D = {target_d:.2f} ---")
    print(f"Calculated Hurst Exponent (H) = {H:.3f}")
    print(f"Calculated Persistence (p)  = {p:.4f}")
    print(f"-------------------------------------------")
    
    ps_temp, v_temp = [], []
    
    for _ in tqdm(range(n), desc=f"Collecting {n} samples"):
        im = generate_perlin_2d(shape, scale=50, octaves=6, persistence=p)
        ps_temp.append(power_spectrum(im)[0])
        v_temp.append(variogram(im))
        
    ps_mean = np.mean(ps_temp)
    v_mean = np.mean(v_temp)
        
    # 2. Output the results alongside the absolute error
    print(f"\nTheoretical Target D: {target_d:.5f}")
    print(f"Power Spectrum Mean : {ps_mean:.5f} (Error: {abs(target_d - ps_mean):.5f})")
    print(f"Variogram Mean      : {v_mean:.5f} (Error: {abs(target_d - v_mean):.5f})")

def generate_test_image(shape=(512, 512), scale=100.0, octaves=6, persistence=0.5, color="default"):
    """Generates and saves a Perlin noise image with selectable color schemes."""
    im = generate_perlin_2d(shape, scale=scale, octaves=octaves, persistence=persistence)
    
    filename = f"perlin_{color}_s{scale}_o{octaves}_p{persistence:.2f}.png"
    
    if color == "minecraft":
        array_to_custom_color_image(im, filename)
    elif color == "purple":
        array_to_purple_black_image(im, filename)
    else:
        array_to_image(im, filename)
        
    print(f"Image saved successfully as {filename}")