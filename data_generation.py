import numpy as np
import pandas as pd
import time
import os
from scipy.special import erfc
from itertools import product as iproduct

# --- Configuration ---
LOG_INTERVAL = 10000        # Log progress every 10,000 rows
PHYSICS_CSV = "data_physics_total.csv"   
RANDOM_CSV = "data_random_total.csv"     
PHYSICS_MAX_MEM_LEN = 14          
ARRIVAL_COVERAGE = 0.70

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

def calculate_ber_vectorized(mem_len, threshold, P_scaled, variances):
    """
    Vectorized BER calculation using Gaussian approximation.
    """
    P_arr = np.asarray(P_scaled, dtype=float)[:mem_len]
    vars_arr = np.asarray(variances, dtype=float)[:mem_len]
    
    # Generate all sequences: shape (2^mem_len, mem_len)
    seqs = np.array(list(iproduct([0, 1], repeat=mem_len)), dtype=np.float64)[:, ::-1]
    c_bit = seqs[:, 0] 

    # Calculate Mean and Variance for every sequence simultaneously
    mu = (seqs * P_arr).sum(axis=1)
    var_total = (seqs * vars_arr).sum(axis=1)
    std = np.sqrt(np.maximum(var_total, 0.0))

    pe = np.empty_like(mu)
    
    # Deterministic edge case (zero variance)
    zero_std = (std == 0)
    if np.any(zero_std):
        pe[zero_std & (c_bit == 1)] = np.where(mu[zero_std & (c_bit == 1)] < threshold, 1.0, 0.0)
        pe[zero_std & (c_bit == 0)] = np.where(mu[zero_std & (c_bit == 0)] >= threshold, 1.0, 0.0)

    # Gaussian approximation case
    nz = ~zero_std
    if np.any(nz):
        # Using 0.5 * erfc(x / sqrt(2)) which is equivalent to the Q-function
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
    
    f_inf = radius / (radius + distance)
    target = ARRIVAL_COVERAGE * f_inf 
    
    cumsum, k = 0.0, 0
    while k < PHYSICS_MAX_MEM_LEN:
        pk = Fhit_function(radius, distance, diff, (k+1)*Ts) - Fhit_function(radius, distance, diff, k*Ts)
        cumsum += pk
        k += 1
        if cumsum >= target: break
        
    P_ext = calculate_hitting_probabilities(k + 1, radius, distance, diff, Ts)
    P_main = P_ext[:k]
    P_extra = float(P_ext[k]) 
    
    # Physics-based mean and variance
    P_scaled = P_main * N
    variances = N * P_main * (1.0 - P_main) 
    
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
    """
    Generates synthetic data using random monotone-decreasing sequences.
    """
    mem_len = int(rng.integers(3, 11)) 
    N = int(10 ** rng.uniform(3.0, 6.0)) # Randomize N for the synthetic sample
    
    P = np.empty(mem_len)
    P[0] = rng.uniform(0.05, 0.95)
    for i in range(1, mem_len):
        P[i] = P[i-1] * rng.uniform(0.35, 0.98)
    P = np.clip(P, 1e-6, 1 - 1e-6) 
    
    P_scaled = P * N
    variances = N * P * (1.0 - P) 
    
    # Random threshold based on the first slot's expected arrivals
    threshold = float(rng.uniform(1.0, max(2.0, 1.5 * P_scaled[0])))
    
    ber = calculate_ber_vectorized(mem_len, threshold, P_scaled, variances)
    return {"mem_len": mem_len, "threshold": threshold, "BER": ber, "p0": P[0], "N": N}

# --- Main Runtime ---

def save_data(data_list, filename):
    """Appends data to the CSV and includes a header only if the file is new."""
    df = pd.DataFrame(data_list)
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)

if __name__ == "__main__":
    rng = np.random.default_rng()
    p_buffer, r_buffer = [], []
    total_generated = 0
    
    print(f"Generation Active. Logging every {LOG_INTERVAL} rows.")
    print(f"Files: {PHYSICS_CSV}, {RANDOM_CSV}")
    
    try:
        while True:
            p_buffer.append(generate_physics_sample(rng))
            r_buffer.append(generate_random_sample(rng))
            total_generated += 1

            # To prevent data loss, we append to the file frequently
            # We use a small buffer (e.g., 100 rows) to balance safety and speed
            if len(p_buffer) >= 100:
                save_data(p_buffer, PHYSICS_CSV)
                save_data(r_buffer, RANDOM_CSV)
                p_buffer, r_buffer = [], []

            if total_generated % LOG_INTERVAL == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Total rows generated and saved: {total_generated}")

    except KeyboardInterrupt:
        print("\nStopping... flushing final buffer.")
        if p_buffer:
            save_data(p_buffer, PHYSICS_CSV)
            save_data(r_buffer, RANDOM_CSV)
        print("Data saved successfully.")