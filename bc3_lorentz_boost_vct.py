import numpy as np

def crossing_dev(delta_p, beta=0.6, L_rest=1.0):
    """Single-observer crossing-time deviation for a rod of rest length
    L_rest (which may be an array, for vectorized phase-sampling)."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    L_lab_tr = L_rest / gamma
    dt_true = L_lab_tr / beta
    n_ticks_true = dt_true / delta_p        # real-valued, not yet rounded
    dt_meas = np.round(n_ticks_true) * delta_p
    return ( beta * dt_meas / L_lab_tr - 1.0 )

def rms_dev(n_ticks, beta=0.6, n_phases=500, seed=0):
    """ Fix delta_p via reference L_rest=1.0 to preserve grid resolution across beta sweeps; sampling delta_p per sampled length trivializes phase-averaging error to zero."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    dt_true_ref = (1.0 / gamma) / beta      # reference interval, L_rest=1.0
    delta_p = dt_true_ref / n_ticks
    rng = np.random.default_rng(seed)
    L_rests = rng.uniform(0.5, 1.5, n_phases)
    devs = crossing_dev(delta_p, beta, L_rests)  # Fully vectorized
    return np.sqrt(np.mean(devs**2))

print("BC3 Kinematic Sub-Criterion: single-observer crossing-time protocol\n")
print("Sweep 1: resolution n_ticks -> infinity at fixed beta = 0.6")
print(f"{'n_ticks':>10} | {'rms dev':>12}\n" + "-"*28)
for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
    print(f"{n:10d} | {rms_dev(n):12.3e}")

print("\nSweep 2: beta -> 1 at fixed resolution n_ticks = 10000")
print(f"{'beta':>8} | {'gamma':>10} | {'rms dev':>12}\n" + "-"*36)
for beta in [0.5, 0.9, 0.99, 0.999, 0.9999]:
    g = 1.0 / np.sqrt(1.0 - beta**2)
    print(f"{beta:8.4f} | {g:10.2f} | {rms_dev(10_000, beta):12.3e}")

print("\nPass criterion: rms deviation falls as n_ticks -> infinity (Sweep 1) and")
print("remains bounded (does not grow with gamma) as beta -> 1 (Sweep 2).")
