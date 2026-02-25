import numpy as np
from scipy.stats import linregress

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

# Old function; doesn't work efficiently at all
# def rs(signal):
#     """
#     Placeholder function for R/S analysis to estimate the Hurst exponent of a 1D signal.
#     This function is not yet implemented and serves as a template for future development.

#     :param signal: 1D numpy array representing the signal
#     :return: Estimated Hurst exponent (currently returns None)
#     """

#     n = len(signal)
#     rs_values = []
#     valid_k = [] # Keep track of valid k values to ensure arrays match later

#     # Start from k=2 to avoid standard deviation of a single element
#     for k in range(2, n + 1):
#         # 1. Slice the sub-series for the current step
#         sub_signal = signal[:k]
        
#         # 2. Calculate mean and standard deviation for this specific slice
#         mu = np.mean(sub_signal)
#         S = np.std(sub_signal, ddof=1)
        
#         # Safety check: if standard deviation is 0, we can't divide by it
#         if S == 0:
#             continue
            
#         # 3. Calculate the mean-centered cumulative deviate (Y)
#         Y = np.cumsum(sub_signal - mu)
        
#         # 4. Calculate the Range (R) for this specific slice
#         R = np.max(Y) - np.min(Y)
        
#         # 5. Calculate R/S and store it along with its corresponding k
#         rs_values.append(R / S)
#         valid_k.append(k)

#     rsSeries = np.array(rs_values)

#     # Create log series for linear regression using the valid_k values
#     log_k = np.log(valid_k)
#     log_rs = np.log(rsSeries)

#     # Perform linear regression to find the slope (Hurst Exponent)
#     slope, intercept, r_value, p_value, std_err = linregress(log_k, log_rs)

#     return slope