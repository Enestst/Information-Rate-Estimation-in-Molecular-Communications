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
MAX_MEM_LEN = 12          

# Create separate output directories
for folder in [PHYSICS_DIR, RANDOM_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Core Physics & BER Functions ---

def Fhit_function(radius, distance, diffusionCoef, t):
    """Calculates the cumulative hitting probability up to time t."""
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
    Calculates the Bit Error Rate (BER) using a Gaussian approximation.
    Optimized via NumPy vectorization to handle all 2^mem_len sequences simultaneously.
    """
    P_arr = np.asarray(P, dtype=float)[:mem_len]
    vars_arr = np.asarray(variances, dtype=float)[:mem_len]
    
    # Generate all possible bit sequences of length mem_len
    seqs = np.array(list(iproduct([0, 1], repeat=mem_len)), dtype=np.float64)[:, ::-1]
    c_bit = seqs[:, 0] # Current bit is the first column after reversal

    # Calculate Mean and Variance for each sequence
    mu = (seqs * P_arr).sum(axis=1)
    var_total = (seqs * vars_arr).sum(axis=1)
    std = np.sqrt(np.maximum(var_total, 1e-15))

    pe = np.empty_like(mu)
    idx_1 = (c_bit == 1)
    idx_0 = (c_bit == 0)
    
    # Calculate conditional error probability using Gaussian Q-function equivalent
    pe[idx_1] = 0.5 * erfc((mu[idx_1] - threshold) / (std[idx_1] * np.sqrt(2)))
    pe[idx_0] = 0.5 * erfc((threshold - mu[idx_0]) / (std[idx_0] * np.sqrt(2)))

    return float(np.mean(pe))

# --- Data Generation Modes ---

def generate_physics_sample(rng):
    """Generates data based on physical molecular communication parameters."""
    radius = rng.uniform(3.0, 7.0)
    distance = rng.uniform(3.0, 15.0)
    diff = rng.uniform(50.0, 120.0)
    Ts = rng.uniform(0.5, 3.0)
    N = int(10 ** rng.uniform(3.0, 6.0))
    
    # Determine memory length based on 70% cumulative arrival coverage
    f_inf = radius / (radius + distance)
    target = 0.70 * f_inf
    cumsum, k = 0.0, 0
    while k < MAX_MEM_LEN:
        pk = Fhit_function(radius, distance, diff, (k+1)*Ts) - Fhit_function(radius, distance, diff, k*Ts)
        cumsum += pk
        k += 1
        if cumsum >= target: break
        
    P = calculate_hitting_probabilities(k, radius, distance, diff, Ts)
    P_scaled = P * N
    # Variance of a sum of independent Binomials: N*p*(1-p)
    variances = N * P * (1.0 - P) 
    threshold = rng.uniform(1.0, max(2.0, 1.5 * P_scaled[0]))
    
    ber = calculate_ber_vectorized(k, threshold, P_scaled, variances)
    return {
        "radius": radius, "distance": distance, "diffusion": diff, 
        "Ts": Ts, "N": N, "mem_len": k, "threshold": threshold, "BER": ber
    }

def generate_random_sample(rng):
    """Generates synthetic data with random monotone-decreasing probabilities."""
    mem_len = int(rng.integers(3, 10))
    
    # Generate random monotone-decreasing P[i] sequence in (0, 1)
    P = np.empty(mem_len)
    P[0] = rng.uniform(0.05, 0.95)
    for i in range(1, mem_len):
        P[i] = P[i-1] * rng.uniform(0.35, 0.98)
    
    # Generate random variances for each tap, independent from P(1-P) rule
    variances = rng.uniform(0.05, 1.00, size=mem_len) * P
    threshold = rng.uniform(1e-3, max(2e-3, 2.0 * P[0]))
    
    ber = calculate_ber_vectorized(mem_len, threshold, P, np.maximum(variances, 1e-9))
    return {"mem_len": mem_len, "threshold": threshold, "BER": ber, "p0": P[0]}

# --- Main Loop ---

if __name__ == "__main__":
    rng = np.random.default_rng()
    physics_buffer = []
    random_buffer = []
    
    print("Starting continuous generation...")
    print(f"Physics data saved to: {PHYSICS_DIR}/")
    print(f"Random data saved to:  {RANDOM_DIR}/")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # Generate samples for both modes
            physics_buffer.append(generate_physics_sample(rng))
            random_buffer.append(generate_random_sample(rng))

            # Save Physics batch when limit is reached
            if len(physics_buffer) >= BATCH_SIZE:
                ts = int(time.time())
                pd.DataFrame(physics_buffer).to_csv(f"{PHYSICS_DIR}/physics_{ts}.csv", index=False)
                physics_buffer = []
                print(f"[{time.strftime('%H:%M:%S')}] Saved 10,000 Physics samples.")

            # Save Random batch when limit is reached
            if len(random_buffer) >= BATCH_SIZE:
                ts = int(time.time())
                pd.DataFrame(random_buffer).to_csv(f"{RANDOM_DIR}/random_{ts}.csv", index=False)
                random_buffer = []
                print(f"[{time.strftime('%H:%M:%S')}] Saved 10,000 Random samples.")

    except KeyboardInterrupt:
        print("\nStopping and saving remaining progress...")
        if physics_buffer:
            pd.DataFrame(physics_buffer).to_csv(f"{PHYSICS_DIR}/physics_final_{int(time.time())}.csv", index=False)
        if random_buffer:
            pd.DataFrame(random_buffer).to_csv(f"{RANDOM_DIR}/random_final_{int(time.time())}.csv", index=False)
        print("Done.")