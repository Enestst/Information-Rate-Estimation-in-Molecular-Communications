import numpy as np
import pandas as pd
import time
import os
from scipy.special import erfc
from itertools import product as iproduct

# --- Configuration ---
BATCH_SIZE = 10000        
PHYSICS_DIR = "data_physics"   
RANDOM_DIR = "data_random"     
PHYSICS_MAX_MEM_LEN = 14          
ARRIVAL_COVERAGE = 0.70

if not os.path.exists(PHYSICS_DIR):
    os.makedirs(PHYSICS_DIR)
if not os.path.exists(RANDOM_DIR):
    os.makedirs(RANDOM_DIR)

# --- Core Functions ---

def Fhit_function(radius, distance, diffusionCoef, t):
    """Calculates cumulative hitting probability up to time t."""
    if t <= 0: return 0.0
    return (radius / (distance + radius)) * erfc(distance / np.sqrt(4 * diffusionCoef * t))

def calculate_hitting_probabilities(mem_len, radius, distance, diffusionCoef, Ts):
    """Calculates interval hitting probabilities P[i] for each symbol slot."""
    P = np.zeros(mem_len)
    for i in range(mem_len):
        t_end = (i + 1) * Ts
        t_start = i * Ts
        P[i] = Fhit_function(radius, distance, diffusionCoef, t_end) - Fhit_function(radius, distance, diffusionCoef, t_start)
    return P

def calculate_ber_vectorized(mem_len, threshold, P, variances):
    """
    Vectorized BER calculation using Gaussian approximation.
    Calculates error probability across all 2^mem_len bit sequences.
    """
    P_arr = np.asarray(P, dtype=float)[:mem_len]
    vars_arr = np.asarray(variances, dtype=float)[:mem_len]
    
    # Generate bit patterns where column 0 represents the current bit
    seqs = np.array(list(iproduct([0, 1], repeat=mem_len)), dtype=np.float64)[:, ::-1]
    c_bit = seqs[:, 0] 

    mu = (seqs * P_arr).sum(axis=1)
    var_total = (seqs * vars_arr).sum(axis=1)
    std = np.sqrt(np.maximum(var_total, 0.0))

    pe = np.empty_like(mu)
    
    # Deterministic edge case for zero variance
    zero_std = (std == 0)
    if np.any(zero_std):
        pe[zero_std & (c_bit == 1)] = np.where(mu[zero_std & (c_bit == 1)] < threshold, 1.0, 0.0)
        pe[zero_std & (c_bit == 0)] = np.where(mu[zero_std & (c_bit == 0)] >= threshold, 1.0, 0.0)

    # Gaussian Q-function equivalent for non-zero variance
    nz = ~zero_std
    if np.any(nz):
        pe[nz & (c_bit == 1)] = 0.5 * erfc((mu[nz & (c_bit == 1)] - threshold) / (std[nz & (c_bit == 1)] * np.sqrt(2)))
        pe[nz & (c_bit == 0)] = 0.5 * erfc((threshold - mu[nz & (c_bit == 0)]) / (std[nz & (c_bit == 0)] * np.sqrt(2)))

    return float(np.mean(pe))

# --- Generation Logic ---

def generate_physics_sample(rng):
    """Generates data from physical molecular communication parameters."""
    radius = rng.uniform(3.0, 7.0)
    distance = rng.uniform(3.0, 15.0)
    diff = rng.uniform(50.0, 120.0)
    Ts = rng.uniform(0.5, 3.0)
    N = int(10 ** rng.uniform(3.0, 6.0)) 
    
    # Determine memory length to reach specified arrival coverage
    f_inf = radius / (radius + distance)
    target = ARRIVAL_COVERAGE * f_inf 
    
    cumsum, k = 0.0, 0
    while k < PHYSICS_MAX_MEM_LEN:
        pk = Fhit_function(radius, distance, diff, (k+1)*Ts) - Fhit_function(radius, distance, diff, k*Ts)
        cumsum += pk
        k += 1
        if cumsum >= target: break
        
    # Calculate channel taps including one extra tap for residual ISI analysis
    P_ext = calculate_hitting_probabilities(k + 1, radius, distance, diff, Ts)
    P_main = P_ext[:k]
    P_extra = float(P_ext[k]) 
    
    P_scaled = P_main * N
    variances = N * P_main * (1.0 - P_main) # Binomial variance property
    
    # Threshold sampling range based on first-tap expected value
    lambda_max = max(1.0, 1.5 * P_scaled[0])
    threshold = float(rng.uniform(1.0, lambda_max))
    
    ber = calculate_ber_vectorized(k, threshold, P_scaled, variances)
    
    return {
        "radius": radius, "distance": distance, "diffusion": diff, "Ts": Ts, "N": N, 
        "mem_len": k, "threshold": threshold, "BER": ber,
        "P_mem_len_extra": P_extra, 
        "P_mem_len_extra_var": P_extra * (1.0 - P_extra)
    }

def generate_random_sample(rng):
    """Generates synthetic data using random monotone-decreasing sequences."""
    mem_len = int(rng.integers(3, 10)) 
    
    # Generate monotone-decreasing probability taps
    P = np.empty(mem_len)
    P[0] = rng.uniform(0.05, 0.95)
    for i in range(1, mem_len):
        P[i] = P[i-1] * rng.uniform(0.35, 0.98)
    P = np.clip(P, 1e-6, 1 - 1e-6) 
    
    # Assign random independent variances
    scales = rng.uniform(0.05, 1.00, size=mem_len)
    variances = np.maximum(scales * P, 1e-9) 
    
    # Wide-range threshold sampling
    threshold = float(rng.uniform(1e-3, max(2e-3, 2.0 * P[0])))
    
    ber = calculate_ber_vectorized(mem_len, threshold, P, variances)
    return {"mem_len": mem_len, "threshold": threshold, "BER": ber, "p0": P[0]}

# --- Main Runtime ---

if __name__ == "__main__":
    rng = np.random.default_rng()
    p_buffer, r_buffer = [], []
    
    print("Data generation in progress. Folders: /data_physics and /data_random")
    try:
        while True:
            p_buffer.append(generate_physics_sample(rng))
            r_buffer.append(generate_random_sample(rng))

            if len(p_buffer) >= BATCH_SIZE:
                pd.DataFrame(p_buffer).to_csv(f"{PHYSICS_DIR}/p_{int(time.time())}.csv", index=False)
                p_buffer = []
                print(f"[{time.strftime('%H:%M:%S')}] Physics batch saved.")

            if len(r_buffer) >= BATCH_SIZE:
                pd.DataFrame(r_buffer).to_csv(f"{RANDOM_DIR}/r_{int(time.time())}.csv", index=False)
                r_buffer = []
                print(f"[{time.strftime('%H:%M:%S')}] Random batch saved.")

    except KeyboardInterrupt:
        print("\nStopping and flushing buffers...")
        if p_buffer: pd.DataFrame(p_buffer).to_csv(f"{PHYSICS_DIR}/p_final.csv", index=False)
        if r_buffer: pd.DataFrame(r_buffer).to_csv(f"{RANDOM_DIR}/r_final.csv", index=False)