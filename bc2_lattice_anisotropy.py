# ``` Python

import numpy as np

# 1. Candidate lattice nearest-neighbor unit vectors
LATTICES = {
    "Simple Cubic": np.vstack([np.eye(3), -np.eye(3)]),
    "BCC": np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1)
           for sz in (1, -1)], dtype=float),
    "FCC": np.array([[a, b, 0] for a in (1, -1) for b in (1, -1)] +
                    [[a, 0, b] for a in (1, -1) for b in (1, -1)] +
                    [[0, a, b] for a in (1, -1) for b in (1, -1)],
                    dtype=float),
    "Diamond": np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
               dtype=float),
}
LATTICES = {k: v / np.linalg.norm(v[0]) for k, v in LATTICES.items()}

# 2. Vectorised Spherical Grid (60 theta x 30 phi)
th = np.linspace(0, np.pi, 60)[:, None]
ph = np.linspace(0, 2 * np.pi, 30, endpoint=False)[None, :]
K_HAT = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                  np.cos(th) * np.ones_like(ph)], axis=-1).reshape(-1, 3)

def compute_anisotropy(vecs, k_mag):
    """Fully vectorised c(theta, phi) sampling and fractional spread calculation."""
    phases = (K_HAT * k_mag) @ vecs.T
    omega2 = np.sum(2.0 * (1.0 - np.cos(phases)), axis=1)
    speeds = np.sqrt(omega2) / k_mag
    return ( (speeds.max() - speeds.min()) / speeds.mean() )

print(f"{'Lattice':<14}{'Leading Dev':>14}{'k*dp=0.1':>12}{'k*dp=0.5':>12}{'k*dp=1.0':>12}\n" + "-" * 64)

for name, vecs in LATTICES.items():
    T2 = vecs.T @ vecs
    eigs = np.linalg.eigvalsh(T2)
    leading_dev = (eigs.max() - eigs.min()) / eigs.mean()

    res = [compute_anisotropy(vecs, k) for k in (0.1, 0.5, 1.0)]

    print(f"{name:<14}{leading_dev:>16.2e}{res[0]:>14.2e}"\
              f"{res[1]:>14.2e}{res[2]:>14.2e}")
print("-" * 64 + "\nBC2 Pass Criterion: Exact quadratic isotropy; " +
                 "quartic LIV onset.")

# ```