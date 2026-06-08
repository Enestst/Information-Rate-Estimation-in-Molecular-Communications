import numpy as np
from scipy.special import erfc
from itertools import product as iproduct

print("="*80)
print("COMPLETE NUMERICAL SIMULATION EXAMPLE")
print("="*80)

# Step 1: Random Physical Parameters
print("\n### STEP 1: Physical Parameters ###")
radius = 5.0        # μm
distance = 10.0     # μm
diffusion = 80.0    # μm²/s
Ts = 1.5            # seconds (symbol duration)
N = 50000           # molecules released per bit=1

print(f"Receiver radius (r):        {radius} μm")
print(f"Distance (d):               {distance} μm")
print(f"Diffusion coefficient (D):  {diffusion} μm²/s")
print(f"Symbol duration (Ts):       {Ts} s")
print(f"Molecules released (N):     {N}")

# Step 2: Calculate Ultimate Hitting Probability
print("\n### STEP 2: Ultimate Arrival Probability ###")
f_inf = radius / (radius + distance)
print(f"f_∞ = r / (r + d) = {radius} / {radius + distance} = {f_inf:.4f}")
print(f"→ {f_inf*100:.2f}% of molecules will EVENTUALLY arrive (t → ∞)")

# Step 3: Determine Memory Length (70% coverage)
print("\n### STEP 3: Determine Memory Length (70% coverage) ###")
ARRIVAL_COVERAGE = 0.70
target = ARRIVAL_COVERAGE * f_inf
print(f"Target coverage: {ARRIVAL_COVERAGE*100}% × {f_inf:.4f} = {target:.4f}")

def Fhit_function(radius, distance, diffusionCoef, t):
    """Calculates cumulative hitting probability up to time t."""
    if t <= 0:
        return 0.0
    return (radius / (distance + radius)) * erfc(distance / np.sqrt(4 * diffusionCoef * t))

# Find memory length
MAX_MEM_LEN = 15
cumsum = 0.0
k = 0
P_list = []

print(f"\nSlot | Time (s) | P[i]     | Cumulative | Target Met?")
print("-" * 60)

while k < MAX_MEM_LEN:
    t_start = k * Ts
    t_end = (k + 1) * Ts
    pk = Fhit_function(radius, distance, diffusion, t_end) - Fhit_function(radius, distance, diffusion, t_start)
    cumsum += pk
    P_list.append(pk)

    met = "✓ STOP" if cumsum >= target else ""
    print(f"{k:4d} | {t_end:8.2f} | {pk:8.5f} | {cumsum:10.5f} | {met}")

    k += 1
    if cumsum >= target:
        break

mem_len = k
P_main = np.array(P_list[:mem_len])

print(f"\n→ Memory length determined: k = {mem_len}")
print(f"→ Captured {cumsum:.5f} out of {f_inf:.5f} total ({cumsum/f_inf*100:.2f}%)")

# Step 4: Calculate Mean and Variance for Each Tap
print("\n### STEP 4: Calculate Means and Variances ###")
print(f"\nFor each slot i: μᵢ = N × P[i],  σᵢ² = N × P[i] × (1 - P[i])")
print(f"\nSlot | P[i]     | μᵢ (mean)  | σᵢ² (variance) | σᵢ (std)")
print("-" * 70)

P_scaled = P_main * N  # Means
variances = N * P_main * (1.0 - P_main)
std_devs = np.sqrt(variances)

for i in range(mem_len):
    print(f"{i:4d} | {P_main[i]:8.5f} | {P_scaled[i]:10.2f} | {variances[i]:14.2f} | {std_devs[i]:10.2f}")

# Step 5: Calculate Σμᵢ
print("\n### STEP 5: Calculate Σμᵢ (Sum of Means) ###")
sum_means = np.sum(P_scaled)
print(f"Σμᵢ = {' + '.join([f'{P_scaled[i]:.2f}' for i in range(mem_len)])}")
print(f"Σμᵢ = {sum_means:.2f} molecules")
print(f"\n→ When transmitting bit=1, we expect ~{sum_means:.0f} molecules to arrive")

# Step 6: Generate Threshold Range
print("\n### STEP 6: Generate Thresholds ###")
NUM_THRESHOLDS = 25
threshold_max = 2.0 * sum_means
thresholds = np.linspace(0, threshold_max, NUM_THRESHOLDS)

print(f"Threshold range: [0, 2 × Σμᵢ] = [0, {threshold_max:.2f}]")
print(f"Number of thresholds: {NUM_THRESHOLDS}")
print(f"\nFirst 5 thresholds: {thresholds[:5]}")
print(f"Last 5 thresholds:  {thresholds[-5:]}")

# Step 7: Calculate BER for Specific Threshold (70% of Σμᵢ)
print("\n### STEP 7: Calculate BER for Threshold = 0.70 × Σμᵢ ###")
threshold = 0.70 * sum_means
print(f"Selected threshold: θ = 0.70 × {sum_means:.2f} = {threshold:.2f} molecules")

def calculate_ber_with_details(mem_len, threshold, P_scaled, variances):
    """
    Calculate BER with detailed output for educational purposes.
    """
    P_arr = np.asarray(P_scaled, dtype=float)[:mem_len]
    vars_arr = np.asarray(variances, dtype=float)[:mem_len]

    # Generate all bit sequences (reversed for c_bit indexing)
    seqs = np.array(list(iproduct([0, 1], repeat=mem_len)), dtype=np.float64)[:, ::-1]
    c_bit = seqs[:, 0]  # Current bit being transmitted

    # Calculate statistics for each sequence
    mu = (seqs * P_arr).sum(axis=1)
    var_total = (seqs * vars_arr).sum(axis=1)
    std = np.sqrt(np.maximum(var_total, 0.0))

    # Calculate error probability for each sequence
    pe = np.empty_like(mu)

    # For bit=1: error if received < threshold
    # For bit=0: error if received >= threshold

    for idx in range(len(seqs)):
        if std[idx] == 0:
            # Deterministic case
            if c_bit[idx] == 1:
                pe[idx] = 1.0 if mu[idx] < threshold else 0.0
            else:
                pe[idx] = 1.0 if mu[idx] >= threshold else 0.0
        else:
            # Gaussian approximation
            if c_bit[idx] == 1:
                pe[idx] = 0.5 * erfc((mu[idx] - threshold) / (std[idx] * np.sqrt(2)))
            else:
                pe[idx] = 0.5 * erfc((threshold - mu[idx]) / (std[idx] * np.sqrt(2)))

    return seqs, c_bit, mu, std, pe

seqs, c_bits, mus, stds, pes = calculate_ber_with_details(mem_len, threshold, P_scaled, variances)

print(f"\nTotal bit sequences to evaluate: 2^{mem_len} = {len(seqs)}")
print(f"\nShowing first 10 sequences:")
print(f"{'Seq':3s} | {'Bits (b₀ b₁ ...)':20s} | {'Current':7s} | {'μ':10s} | {'σ':10s} | {'P(error)':10s}")
print("-" * 80)

for i in range(min(10, len(seqs))):
    bits_str = ''.join([str(int(b)) for b in seqs[i]])
    print(f"{i:3d} | {bits_str:20s} | {int(c_bits[i]):7d} | {mus[i]:10.2f} | {stds[i]:10.2f} | {pes[i]:10.6f}")

print(f"\n... ({len(seqs)-10} more sequences)")

# Calculate average BER
ber = np.mean(pes)
print(f"\n{'='*80}")
print(f"FINAL BER = average of all P(error) = {ber:.6f} ({ber*100:.4f}%)")
print(f"{'='*80}")

# Step 8: Show how this varies with threshold
print("\n### STEP 8: BER vs Threshold (Sample) ###")
print(f"\nShowing BER for 5 different thresholds around Σμᵢ:")
print(f"{'Threshold':12s} | {'Relative to Σμᵢ':20s} | {'BER':12s}")
print("-" * 50)

sample_thresholds = [0.5 * sum_means, 0.7 * sum_means, 1.0 * sum_means, 1.3 * sum_means, 1.5 * sum_means]
for th in sample_thresholds:
    seqs_temp, c_bits_temp, mus_temp, stds_temp, pes_temp = calculate_ber_with_details(mem_len, th, P_scaled, variances)
    ber_temp = np.mean(pes_temp)
    relative = th / sum_means
    print(f"{th:12.2f} | {relative:5.2f} × Σμᵢ         | {ber_temp:.8f}")

print(f"\n→ Optimal threshold minimizes BER (usually near Σμᵢ)")
print(f"→ The CNN learns to predict this optimal threshold for any channel!")
