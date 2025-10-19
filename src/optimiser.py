from __future__ import annotations
import numpy as np

def _power_iteration_top_eigval(A: np.ndarray, iters: int = 50) -> float:
    n = A.shape[0]
    v = np.ones(n) / np.sqrt(n)
    for _ in range(iters):
        Av = A @ v
        nrm = np.linalg.norm(Av)
        if nrm == 0:
            return 1.0
        v = Av / nrm
    return float(v @ (A @ v))

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1:
        v = v.ravel()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1, u.size + 1)) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

def project_to_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Projection onto {w >= 0, sum w = 1, w_i <= cap} via bisection on theta."""
    v = v.ravel()
    lo, hi = v.min() - cap, v.max()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = np.clip(v - mid, 0.0, cap).sum()
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(v - hi, 0.0, cap)

def mean_variance_long_only(mu, cov, gamma: float = 5.0, max_iter: int = 500, tol: float = 1e-7, cap: float | None = 0.6):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = mu.shape[0]

    lam_max = _power_iteration_top_eigval(cov)
    step = 1.0 / (gamma * lam_max + 1e-12)

    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        grad = -mu + gamma * (cov @ w)
        w_new = w - step * grad
        if cap is None:
            w_new = project_to_simplex(w_new)
        else:
            # ensure cap * n >= 1 to keep feasibility
            eff_cap = max(cap, 1.0 / n)
            w_new = project_to_capped_simplex(w_new, cap=eff_cap)
        if np.linalg.norm(w_new - w, 1) <= tol:
            return w_new
        w = w_new
    return w

def per_regime_weights(regime_stats: dict, gamma: float = 5.0, cap: float | None = 0.6, shrink_alpha: float = 0.1):
    """Compute long-only MV weights per regime with light diagonal shrinkage."""
    weights = {}
    for k, d in regime_stats.items():
        mu = d["mu"].values
        S = d["cov"].values
        # simple shrinkage towards diag for stability
        S = (1.0 - shrink_alpha) * S + shrink_alpha * np.diag(np.diag(S))
        S = S + 1e-6 * np.eye(S.shape[0])
        weights[int(k)] = mean_variance_long_only(mu, S, gamma=gamma, cap=cap)
    return weights