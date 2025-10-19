import numpy as np
from src.optimiser import mean_variance_long_only

def test_mean_variance_long_only_shapes_and_simplex():
    # Simple 3-asset problem
    mu = np.array([0.10, 0.05, 0.02])
    cov = np.diag([0.02, 0.01, 0.03]).astype(float)
    w = mean_variance_long_only(mu, cov, gamma=5.0, cap=1.0)
    # Shape
    assert w.shape == (3,)
    # Long-only and sums to 1
    assert (w >= -1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-6
