import scipy
import numpy as np
from scipy.stats import linregress

def fd_power_spectrum(array):
    """
    Estimates fractal dimension using the Power Spectrum (Fourier) method.
    """
    # 1. Compute 2D Fast Fourier Transform
    f_transform = np.fft.fft2(array)
    f_shift = np.fft.fftshift(f_transform) # Shift zero frequency to center
    
    # 2. Calculate Power Spectrum
    power_spectrum = np.abs(f_shift)**2
    
    # 3. Radially average the power spectrum
    center = (power_spectrum.shape[0] // 2, power_spectrum.shape[1] // 2)
    y, x = np.indices(power_spectrum.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    # Sum the power in each radial bin and divide by the number of pixels in that bin
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1) # avoid division by zero
    
    # 4. Fit a line in log-log space (ignore r=0 to avoid log(0))
    r_valid = np.arange(1, len(radial_profile))
    p_valid = radial_profile[1:]
    
    # Filter out zero values before taking log
    mask = p_valid > 0
    r_valid, p_valid = r_valid[mask], p_valid[mask]
    
    log_r = np.log(r_valid)
    log_p = np.log(p_valid)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_p)
    
    # Calculate D (Beta is the negative of the slope)
    beta = -slope
    D = (8 - beta) / 2
    
    # Clip to theoretical bounds for 2D surface
    return np.clip(D, 2.0, 3.0)

def fd_variogram(array, num_samples=5000):
    """
    Estimates fractal dimension using the Variogram (Hurst Exponent) method.
    Uses random sampling for performance on large arrays.
    """
    h, w = array.shape
    distances = []
    variances = []
    
    # Sample random pairs of points
    for _ in range(num_samples):
        # Pick two random coordinates
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        
        # Euclidean distance
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if dist == 0: continue
            
        # Variance (squared difference)
        diff_sq = (array[y1, x1] - array[y2, x2])**2
        
        distances.append(dist)
        variances.append(diff_sq)
        
    distances = np.array(distances)
    variances = np.array(variances)
    
    # Bin the distances to compute average variance per distance bin
    bins = np.logspace(np.log10(1), np.log10(min(w, h)/2), num=20)
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(distances, variances, statistic='mean', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter NaNs (empty bins)
    valid = ~np.isnan(bin_means) & (bin_means > 0)
    log_dist = np.log(bin_centers[valid])
    log_var = np.log(bin_means[valid])
    
    slope, intercept, _, _, _ = linregress(log_dist, log_var)
    
    # Slope = 2H, D = 3 - H
    H = slope / 2
    D = 3 - H
    
    return np.clip(D, 2.0, 3.0)