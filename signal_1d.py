import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def box_count_dimension(signal):
    """
    Calculates the box-counting fractal dimension of a 1D signal.

    :param signal: 1D numpy array representing the signal
    :return tuple (slope, log_eps_inv, log_counts): Estimated box-counting dimension, log(1/epsilon) values, log(N(epsilon)) values
    """
    n = len(signal)
    
    # 1. Normalize the signal to a [0, 1] x [0, 1] box
    signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    # 2. Define box sizes (epsilon)
    # We use powers of 2 for grid divisions: 2, 4, 8, 16... up to a sensible limit
    max_k = int(np.floor(np.log2(n/10))) # Keep at least 10 points per box at the smallest scale
    epsilons = 1.0 / (2 ** np.arange(1, max_k + 1))
    
    counts = []
    
    # 3. Count boxes for each epsilon
    # This function runs in O(n) for each epsilon; box_count_dimension runs in O(n log n), thanks to logarithmic num of epsilons
    for eps in epsilons:
        num_boxes = 0
        num_intervals = int(1 / eps)
        points_per_interval = n // num_intervals
        
        for i in range(num_intervals):
            start_idx = i * points_per_interval
            # Handle the last interval safely
            end_idx = (i + 1) * points_per_interval if i < num_intervals - 1 else n
            
            segment = signal_norm[start_idx:end_idx]
            
            if len(segment) > 0:
                y_min = np.min(segment)
                y_max = np.max(segment)
                # Number of vertical boxes covering the signal in this interval
                vertical_boxes = np.ceil((y_max - y_min) / eps)
                # At least 1 box is needed to cover the points
                num_boxes += max(1, vertical_boxes)
                
        counts.append(num_boxes)

    # 4. Fit the line to log(N(epsilon)) vs log(1/epsilon)
    log_eps_inv = np.log(1.0 / epsilons)
    log_counts = np.log(counts)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_eps_inv, log_counts)
    
    return slope, log_eps_inv, log_counts

def hurst_rs(signal, min_chunk=8):
    """
    Estimates the Hurst exponent of a 1D signal using Rescaled Range (R/S) analysis.
    
    :param signal: 1D array-like representing the signal.
    :param min_chunk: Minimum chunk size to consider.
    :return: Estimated Hurst exponent (slope of the log-log plot).
    """
    signal = np.array(signal)
    n = len(signal)
    
    # 1. Generate logarithmically spaced chunk sizes (k)
    max_chunk = n // 2
    if max_chunk < min_chunk:
        raise ValueError("Signal is too short for the specified minimum chunk size.")
        
    # Create ~20 evenly spaced points on a log scale
    k_values = np.logspace(np.log10(min_chunk), np.log10(max_chunk), num=20, dtype=int)
    k_values = np.unique(k_values) # Remove any duplicates from rounding
    
    rs_averages = []
    valid_k = []
    
    for k in k_values:
        # Calculate how many full chunks of size k we can fit
        num_chunks = n // k
        if num_chunks == 0:
            continue
            
        # Truncate the signal to fit exactly 'num_chunks'
        truncated_signal = signal[:num_chunks * k]
        
        # Reshape into a matrix where each row is a chunk of size k
        # This allows us to vectorize the math operations
        chunks = truncated_signal.reshape(num_chunks, k)
        
        # Calculate mean and standard deviation for each chunk
        means = np.mean(chunks, axis=1, keepdims=True)
        stds = np.std(chunks, axis=1, ddof=1)
        
        # Mean-centered cumulative deviate (Y)
        Y = np.cumsum(chunks - means, axis=1)
        
        # Calculate Range (R) for each chunk
        R = np.max(Y, axis=1) - np.min(Y, axis=1)
        
        # Calculate R/S, avoiding division by zero
        valid_chunks = stds > 0
        if not np.any(valid_chunks):
            continue
            
        rs_values = R[valid_chunks] / stds[valid_chunks]
        
        # Average the R/S values for this specific k
        rs_averages.append(np.mean(rs_values))
        valid_k.append(k)
        
    
    
    # Perform log-log linear regression
    log_k = np.log(valid_k)
    log_rs = np.log(rs_averages)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_k, log_rs)
    
    return slope