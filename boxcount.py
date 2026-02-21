"""
Program: boxcount.py
Author: Nicholas Harsell
Date: 2026-02-21
Purpose: This program calculates the box-counting fractal dimension of a 1D signal.
         Imports a 1D numpy array (the signal), and outputs the approximate box-counting dimension.
         Ex. uniform noise show have a box-counting dimension of 2.
"""

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