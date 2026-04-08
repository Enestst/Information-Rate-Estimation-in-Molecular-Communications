# Data Generation for BER Estimation

## Overview
Generates training data for CNN-based BER (Bit Error Rate) estimation in molecular communication systems. Produces two datasets: physics-based and random synthetic data.

## Output Files
- `data_physics_total.csv`: Physics-based molecular communication scenarios
- `data_random_total.csv`: Synthetic random scenarios

## Data Generation Methods

### 1. Physics-Based Generation (`generate_physics_sample`)
Simulates realistic molecular communication channel using physical parameters:

**Parameters:**
- `radius`: Receiver radius (3-7 μm)
- `distance`: Transmitter-receiver distance (3-15 μm)
- `diffusion`: Diffusion coefficient (50-120 μm²/s)
- `Ts`: Symbol duration (0.5-3 s)
- `N`: Number of molecules released (10³-10⁶)

**Process:**
1. Calculate hitting probabilities using `Fhit_function` (complementary error function)
2. Determine memory length `mem_len` where cumulative probability reaches 70% of `f_inf`
3. Generate 25 threshold values uniformly distributed from 0 to 2×sum(means)
4. Compute BER for each threshold using Gaussian approximation

### 2. Random Synthetic Generation (`generate_random_sample`)
Creates synthetic scenarios without physical constraints:

**Process:**
1. Generate random memory length (3-15)
2. Create monotone-decreasing probability sequence: `P[i] = P[i-1] × uniform(0.35, 0.98)`
3. **Normalize**: Scale probabilities to ensure `sum(P) < 1` (mimics physical constraint)
4. Generate 25 thresholds from 0 to 2×sum(means)
5. Compute BER for each threshold

## BER Calculation
Uses vectorized Gaussian approximation:
- Generates all 2^mem_len possible bit sequences
- Calculates mean `μ = Σ(sequence × P_scaled)`
- Calculates variance `σ² = Σ(sequence × variances)`
- Computes error probability using complementary error function
- Returns average error probability across all sequences

## Key Features

### Threshold Sampling Strategy
For each scenario, generates **25 systematic threshold values** instead of random:
- Range: `[0, 2×sum(means)]`
- Purpose: Model learns threshold-BER relationship
- Benefit: 25× more data points from same expensive physical calculations

### Normalization (Random Data)
Ensures `sum(P) < uniform(0.6, 0.95)` to maintain physical validity:
- Hitting probabilities must sum to less than 1
- Preserves monotone-decreasing structure
- Prevents physically impossible scenarios

## Runtime Behavior
- Runs indefinitely generating samples
- Saves every 100 rows to prevent data loss
- Logs progress every 10,000 rows
- Graceful shutdown on keyboard interrupt

## Output Format
Each row contains:
- Physical/scenario parameters (radius, distance, N, Ts, etc.)
- `mem_len`: ISI memory length
- `threshold`: Decision threshold value
- `BER`: Calculated bit error rate
- `tap_1` to `tap_15`: Hitting probability taps (P values)
- Additional metadata (P_mem_len_extra, variances)
