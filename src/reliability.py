"""How sure is the model, and should we trust it? (lessons 07 and 08)

Unknown scores (higher = more likely an unknown ship):
  msp_score     1 - highest softmax probability (the baseline from the presentation)
  energy_score  -T * logsumexp(logits / T), uses the raw logits before softmax squashes them
  entropy / mutual_information over MC-dropout samples or ensemble members
Calibration:
  fit_temperature  one number T fitted on validation so that softmax(logits / T) is honest
"""
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def softmax(logits, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def msp_score(logits, T: float = 1.0) -> np.ndarray:
    return 1.0 - softmax(logits, T).max(axis=-1)


def energy_score(logits, T: float = 1.0) -> np.ndarray:
    return -T * logsumexp(np.asarray(logits, np.float64) / T, axis=-1)


def entropy(probs) -> np.ndarray:
    p = np.clip(np.asarray(probs, np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def mutual_information(prob_samples) -> np.ndarray:
    """Disagreement between samples (S, N, C): entropy of the average minus the average entropy."""
    s = np.asarray(prob_samples, np.float64)
    return entropy(s.mean(axis=0)) - entropy(s).mean(axis=0)


def ensemble_probs(list_of_probs) -> np.ndarray:
    return np.mean(np.stack(list_of_probs), axis=0)


def fit_temperature(logits, y, weights=None) -> float:
    """weights: optional per-clip weights, e.g. metrics.class_balanced_weights so no class dominates."""
    logits, y = np.asarray(logits, np.float64), np.asarray(y)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, np.float64)

    def nll(log_t):
        z = logits / np.exp(log_t)
        return float(np.average(logsumexp(z, axis=1) - z[np.arange(len(y)), y], weights=w))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


@torch.no_grad()
def mc_dropout_logits(model, loader, n_samples: int, device) -> np.ndarray:
    """Run the whole loader n_samples times with only the dropout layers active -> (S, N, C) logits."""
    model.eval()
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in dropouts:
        m.train()
    try:
        samples = []
        for _ in range(n_samples):
            out = [model(b["x"].to(device))["logits"].float().cpu() for b in loader]
            samples.append(torch.cat(out).numpy())
        return np.stack(samples)
    finally:
        for m in dropouts:
            m.eval()
