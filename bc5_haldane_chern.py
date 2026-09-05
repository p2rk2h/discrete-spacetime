# ``` Python

import numpy as np

# --- Honeycomb lattice and reciprocal basis ---------------------------
d = np.array([[0.0, 1.0], [np.sqrt(3)/2, -0.5], [-np.sqrt(3)/2, -0.5]])
b = np.array([d[1] - d[2], d[2] - d[0], d[0] - d[1]])
G = 2 * np.pi * np.linalg.inv(np.array([b[0], -b[1]])).T


# --- Bloch Hamiltonian & Lattice Chern Number -------------------------
def H_k(k, t1, t2, phi, M):
    """2x2 Haldane Bloch Hamiltonian omitting trace shift."""
    hx = t1 * np.sum(np.cos(k @ d.T), axis = -1)
    hy = t1 * np.sum(np.sin(k @ d.T), axis = -1)
    hz = M - 2 * t2 * np.sin(phi) * np.sum(np.sin(k @ b.T), axis = -1)

    H = np.zeros(k.shape[ : -1] + (2, 2), dtype = complex)
    H[..., 0, 0], H[..., 1, 1] = hz, -hz
    H[..., 0, 1], H[..., 1, 0] = hx - 1j * hy, hx + 1j * hy
    return ( H )

def chern_number(t1 = 1.0, t2 = 0.2, phi = np.pi / 2, M = 0.0, Nk = 48):
    """Fukui-Hatsugai-Suzuki gauge-invariant lattice Chern number."""
    idx = np.arange(Nk) / Nk
    k_grid = idx[:, None, None] * G[0] + idx[None, :, None] * G[1]

    # Lower-band eigenvectors across the discretized Brillouin zone
    _, v = np.linalg.eigh(H_k(k_grid, t1, t2, phi, M))
    u = v[..., : , 0]

    # U(1) link variables along cell boundaries
    U1 = np.sum(u.conj() * np.roll(u, -1, axis = 0), axis = -1)
    U2 = np.sum(u.conj() * np.roll(u, -1, axis = 1), axis = -1)
    U1 /= np.abs(U1)
    U2 /= np.abs(U2)

    # Lattice field strength (plaquette flux)
    F = np.angle(U1 * np.roll(U2, -1, axis = 0) * np.roll(U1, -1,
                 axis = 1).conj() * U2.conj())
    return ( np.sum(F) / (2 * np.pi) )

t1, t2, phi = 1.0, 0.2, np.pi / 2
Mc = 3 * np.sqrt(3) * t2 * np.sin(phi)

print(f"Critical Mass (topological/trivial boundary): Mc = {Mc:+.6f}")
print(f"Topological Phase (M=0.0): C = "
      f"{chern_number(t1, t2, phi, 0.0):+.6f}")
print(f"Trivial Phase (M=2.0):     C = "
      f"{chern_number(t1, t2, phi, 2.0):+.6f}")
print(f"Resolution check (Nk=12):  C = "
      f"{chern_number(t1, t2, phi, 0.0, Nk=12):+.6f}")

# ```