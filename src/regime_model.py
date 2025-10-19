from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def fit_hmm(returns: pd.DataFrame, n_states: int = 3, random_state: int = 42) -> GaussianHMM:
    #HMM expects a 2D numpy array
    x = np.asarray(returns.values, dtype=float)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        verbose=False,
    )
    model.fit(x)
    return model

def posterior_probs(model: GaussianHMM, returns: pd.DataFrame) -> pd.DataFrame:
    x = np.asarray(returns.values, dtype=float)
    _, post = model.score_samples(x)  #(T, n_states)
    return pd.DataFrame(post, index=returns.index, columns=[f"state_{i}" for i in range(model.n_components)])

def viterbi_path(model: GaussianHMM, returns: pd.DataFrame) -> pd.Series:
    x = np.asarray(returns.values, dtype=float)
    states = model.predict(x)
    return pd.Series(states, index=returns.index, name="state")

def regime_stats_by_label(returns: pd.DataFrame, labels: pd.Series):
    stats = {}
    
    for k in np.unique(labels.values):
        mask = (labels.values == k)
        r = returns.iloc[mask]
        mu = r.mean()
        cov = r.cov()
        stats[int(k)] = {"mu": mu, "cov": cov}
    return stats