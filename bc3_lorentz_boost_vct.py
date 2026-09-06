import numpy as np

def crossing_dev(delta_p, beta=0.6, L_rest=1.0):
    """Single-observer crossing-time deviation. delta_p is an independent,
    externally fixed lattice tick -- NOT derived from beta -- so beta
    actually survives into the result."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    L_lab_tr = L_rest / gamma
    dt_true = L_lab_tr / beta
    n_ticks_true = dt_true / delta_p        # real-valued, not yet rounded
    dt_meas = np.round(n_ticks_true) * delta_p
    return ( beta * dt_meas / L_lab_tr - 1.0 )

def rms_dev(delta_p, beta=0.6, n_phases=500, seed=0):
    """RMS over random rest lengths, sampling the phase between the
    continuum crossing time and the fixed lattice grid."""
    rng = np.random.default_rng(seed)
    L_rests = rng.uniform(0.5, 1.5, n_phases)
    devs = crossing_dev(delta_p, beta, L_rests)  # Fully vectorized
    return np.sqrt(np.mean(devs**2))

print("BC3 Kinematic Sub-Criterion: single-observer crossing-time protocol\n")
print("Sweep 1: resolution delta_p -> 0 at fixed beta = 0.6")
print(f"{'delta_p':>10} | {'rms dev':>12}\n" + "-"*28)
for dp in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    print(f"{dp:10.1e} | {rms_dev(dp):12.3e}")

print("\nSweep 2: beta -> 1 at fixed delta_p = 1e-4")
print(f"{'beta':>8} | {'gamma':>10} | {'rms dev':>12}\n" + "-"*36)
for beta in [0.5, 0.9, 0.99, 0.999, 0.9999]:
    g = 1.0 / np.sqrt(1.0 - beta**2)
    print(f"{beta:8.4f} | {g:10.2f} | {rms_dev(1e-4, beta):12.3e}")

print("\nPass criterion: rms deviation falls as delta_p -> 0 (Sweep 1) and")
print("remains bounded (does not grow with gamma) as beta -> 1 (Sweep 2).")