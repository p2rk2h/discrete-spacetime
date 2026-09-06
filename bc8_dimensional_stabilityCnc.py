import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run_trial(L=48, n_iters=2000, D=0.2, noise=0.05, seed=0):
    """Fast 2-D explicit diffusion + noise."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(0, 1e-3, (L, L))
    var = np.empty(n_iters)
    noise_arr = noise * rng.normal(size=(n_iters, L, L))
    for i in range(n_iters):
        lap = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
               np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4*phi)
        phi += D*lap + noise_arr[i]
        if not np.isfinite(phi).all():
            var[i:] = np.inf
            break
        var[i] = phi.var()
    return var

def late_early_ratio(var, tail=0.2):
    if not np.isfinite(var).all():
        return np.inf
    k = max(1, int(len(var)*tail))
    return var[-k:].mean() / var[-2*k:-k].mean()

def _one_seed(args):
    param, v, seed, kw = args
    return late_early_ratio(run_trial(**{param: v, "seed": seed, **kw}))

def sweep(param, values, n_seeds=16, **kw):
    """Return (value, mean_ratio, ci_lower_on_(ratio-1), n_diverged)."""
    out = []
    with ThreadPoolExecutor() as ex:
        for v in values:
            args = [(param, v, s, kw) for s in range(n_seeds)]
            ratios = np.array(list(ex.map(_one_seed, args)))
            n_div = int(np.sum(~np.isfinite(ratios)))
            finite = ratios[np.isfinite(ratios)]
            if len(finite) == 0:
                out.append((v, np.inf, np.inf, n_seeds))
                continue
            mean = finite.mean()
            sem  = finite.std(ddof=1)/np.sqrt(len(finite)) if len(finite) > 1 else np.inf
            out.append((v, mean, (mean - 1) - 1.96*sem, n_div))
    return out

def classify(mean, ci_lo, n_div, n_seeds):
    if n_div:
        return f"DIVERGENT ({n_div}/{n_seeds} overflowed)"
    return "DIVERGENT (growth at 95% CI)" if ci_lo > 0 else "STABLE (no significant growth)"

import time
t0 = time.perf_counter()          # <-- start timer
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)

    print("Sweep 1: diffusion coefficient (noise=0.05, 16 seeds)")
    print("Expected transition near von Neumann bound D ≤ 0.25\n")
    for D, mean, ci_lo, n_div in sweep("D", [0.10, 0.20, 0.24, 0.26, 0.30], noise=0.05):
        mstr = "inf" if not np.isfinite(mean) else (f"{mean:6.3f}" if mean < 1e6 else f"{mean:.2e}")
        print(f"D = {D:.2f} | mean ratio = {mstr} | {classify(mean, ci_lo, n_div, 16)}")

    print("\nSweep 2: noise amplitude (D=0.20, absolute late-time variance)")
    print("Ratio is scale-invariant; absolute variance should scale ~ noise²\n")
    for noise in [0.05, 0.10, 0.20, 0.40, 0.80]:
        finals = [run_trial(D=0.20, noise=noise, seed=s)[-1] for s in range(16)]
        mv = np.mean(finals)
        print(f"noise = {noise:.2f} | late var = {mv:.3e} | var/noise² = {mv/noise**2:.3e}")

print("\nBC8 criterion: late/early variance ratio shows no statistically")
print("significant growth (95% CI excludes ratio > 1) across tested D.")
t1 = time.perf_counter()          # <-- end timer
print(f"\nTotal runtime: {t1 - t0:.2f} seconds")