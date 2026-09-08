# discrete-spacetime
symbolic derivations, arbitrary-precision numerical audits, and lattice dispersion solvers supporting "Sub-Planckian Discrete Spacetime Calibrated by Dirac's Large-Number Ratio: Effective Field Theory Bounds, Scale-Free Invariants, and a Ten-Condition Falsifiability Roadmap"

### Numerical Verification & Symbolic Audit Suite

This repository contains the symbolic derivations, arbitrary-precision numerical audits, and lattice dispersion solvers supporting the manuscript *"Sub-Planckian Discrete Spacetime Calibrated by Dirac's Large-Number Ratio: Effective Field Theory Bounds, Scale-Free Invariants, and a Ten-Condition Falsifiability Roadmap."*

The verification scripts evaluate key boundary conditions (BCs) established in the ten-condition falsifiability roadmap (Appendices A and B). All symbolic derivations are implemented in Python using *SymPy*, high-precision numerical identities are audited using *mpmath* (set to 50 decimal places of precision), and lattice dispersion/topological solvers utilize vectorized *NumPy* routines.


### Script Summary & Roadmap Mapping

The table below outlines the computational scripts available in the `discrete-spacetime/` directory, detailing their corresponding manuscript's boundary conditions, mathematical tasks, and analytical targets:

| Python Filename | Target Boundary Condition | Computational Role | Primary Output / Physical Result |
|:---|:---|:---|:---|
| `bc2_lattice_anisotropy.py` | **BC2** (Isotropy and Dispersion) | *Lattice Dispersion Anisotropy Evaluator:* Vectorized dispersion analysis evaluating wave-speed directional spread across SC, BCC, FCC, and Diamond geometries. | Confirms exact quadratic isotropy and quantifies quartic LIV onset across candidate lattices. |
| bc3_lorentz_boost.py | BC3 (Local Lorentz Covariance) | Quantized Crossing-Time Verifier: Single-observer crossing-time protocol evaluating length-contraction convergence at fixed relative resolution ($n_\mathrm{ticks}$) across boost factor $\beta$. | Confirms RMS deviation from $\ell_\mathrm{obs} = \ell_\mathrm{rest}\sqrt{1-\beta^2}$ falls as $\mathcal{O}(1/n_\mathrm{ticks})$ and remains flat, with no growth in $\gamma$, from $\beta = 0.5$ up to $\gamma > 10^7$. |
| `bc4_scale_hierarchy.py` | **BC4** (IR Scale Matching) | *Infrared Scale Hierarchy Auditor ($G_N / G_\mathrm{raw}$):* Symbolic dimension checker and arbitrary-precision ($50$-dps) arbitrary-precision evaluation of $G_N / G_\mathrm{raw} = \mathcal{Q}^2$. | Verifies symbolic dimensional consistency and evaluates the exact calibration ratio $inv_Q2 / \mathcal{Q}^2 \equiv 1.000000$. |
| `bc5_haldane_chern.py` | **BC5** (Topological Invariants) | *Topological Chern Number Evaluator:* Gauge-invariant Fukui–Hatsugai–Suzuki lattice discretization computing $C$ for Haldane honeycomb phases. | Evaluates phase transitions across critical mass $M_c$ <br> ($C = +1$ in topological phase; $C = 0$ in trivial phase). |
| bc8_dimensional_stability.py | BC8 (Dimensional Stability) | Substrate Relaxation Stability Auditor: Stochastic diffusion-relaxation simulation on a periodic lattice, classifying variance growth via 95%-confidence-interval significance testing across diffusion coefficient $D$. | Confirms no statistically significant variance growth for $D \leq 0.24$; correctly identifies divergence at $D \geq 0.26$, consistent with the von Neumann stability bound $D \leq 0.25$. |
| `bc9_liv_cosmology.py` | **BC9** (LIV Observational Bounds) | *Observational LIV Limit Evaluator:* Cosmological $K_2(z)$ redshift integrator under $\Lambda\mathrm{CDM}$ using GRB 090510 arrival-time limits. | Calculates observational upper bound on spatial grid spacing ($\delta_p \le 1.73 \times 10^{-22} \text{ m} \approx 1.07 \times 10^{13} \ \ell_P$). |
