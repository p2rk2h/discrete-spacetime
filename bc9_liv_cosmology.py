# ``` Python

import numpy as np
from scipy.integrate import quad
import scipy.constants as const

# Physical constants (CODATA 2022)
c, hbar, G_N, e = const.c, const.hbar, const.G, const.e
l_P = np.sqrt(hbar * G_N / c**3)              # Planck length (m)

# Flat Lambda-CDM parameters (Planck 2018)
H0 = (67.4 * 1e3) / 3.085677581491367e22    # Hubble constant (1/s)
Omega_M, Omega_L = 0.315, 0.685

def k2_integral(z: float) -> float:
    """Cosmological redshift integral K_2(z) for quadratic LIV."""
    integrand = lambda zp: (1 + zp)**2 / np.sqrt(Omega_M * (1 + zp)**3 + Omega_L)
    return ( quad(integrand, 0.0, z)[0] )

def delta_p_bound(z: float, delta_E_GeV: float, delta_t_max_s: float) -> tuple[float, float]:
    """Upper bound on grid spacing delta_p from an observed arrival-time-delay limit."""
    K2 = k2_integral(z)
    delta_E_J = delta_E_GeV * 1e9 * e
    delta_p_sq = (2 * c * (hbar * c)**2 * delta_t_max_s * H0) / (delta_E_J**2 * K2)
    delta_p_m = np.sqrt(delta_p_sq)
    return ( delta_p_m, delta_p_m / l_P )

# Fermi-LAT GRB 090510 observational inputs
z_grb, E_max_GeV, delta_t_bound = 0.903, 31.0, 0.82

delta_p_m, delta_p_lP = delta_p_bound(z_grb, E_max_GeV, delta_t_bound)

print(f"K_2({z_grb}) = {k2_integral(z_grb):.6f}")
print(f"Max grid spacing (SI):           delta_p <= {delta_p_m:.3e} m")
print(f"Max grid spacing (Planck units): delta_p <= {delta_p_lP:.3e} l_P")

# ```