"""Scores for every experiment. Plain numpy in, plain numbers out (lessons 03, 07, 08, 09)."""
import numpy as np
from sklearn.metrics import roc_auc_score


# ---------- closed-set classification ----------
def accuracy(y, yhat) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(yhat)))


def confusion(y, yhat, n_classes) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (np.asarray(y), np.asarray(yhat)), 1)
    return m


def per_class_recall(y, yhat, n_classes) -> np.ndarray:
    m = confusion(y, yhat, n_classes)
    support = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, np.diag(m) / support, np.nan)


def macro_f1(y, yhat, n_classes) -> float:
    """Average F1 over the classes present in y (a class missing from the test set can't be scored)."""
    m = confusion(y, yhat, n_classes)
    f1s = []
    for c in np.flatnonzero(m.sum(1) > 0):
        tp, fp, fn = m[c, c], m[:, c].sum() - m[c, c], m[c, :].sum() - m[c, c]
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


# ---------- calibration ----------
def reliability_bins(probs, y, n_bins=15) -> dict:
    conf = probs.max(1)
    correct = (probs.argmax(1) == np.asarray(y)).astype(float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        mean_conf = np.bincount(idx, conf, n_bins) / count
        mean_acc = np.bincount(idx, correct, n_bins) / count
    return {"conf": mean_conf, "acc": mean_acc, "count": count}


def ece(probs, y, n_bins=15) -> float:
    """Expected calibration error: average gap between confidence and accuracy, weighted by bin size."""
    b = reliability_bins(probs, y, n_bins)
    used = b["count"] > 0
    return float(np.sum(b["count"][used] / b["count"].sum() * np.abs(b["acc"][used] - b["conf"][used])))


def nll(probs, y) -> float:
    p = probs[np.arange(len(y)), np.asarray(y)]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))


def brier(probs, y) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), np.asarray(y)] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def class_balanced_weights(y) -> np.ndarray:
    """Per-clip weights giving every class the same total weight (mean weight 1).
    Fold 1's validation split is 38 % dredger; without this, calibration would mostly fit dredger clips."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    per_class = len(y) / (len(classes) * counts)
    return per_class[np.searchsorted(classes, y)]


def weighted_percentile(values, weights, q) -> float:
    """Smallest value at which the cumulative weight reaches q % of the total."""
    order = np.argsort(values)
    v, cw = np.asarray(values, float)[order], np.cumsum(np.asarray(weights, float)[order])
    return float(v[np.searchsorted(cw, q / 100.0 * cw[-1] - 1e-12)])


# ---------- open set ("unknown" detection); score: higher = more likely unknown ----------
def open_set_auroc(score, is_unknown) -> float:
    is_unknown = np.asarray(is_unknown, bool)
    if is_unknown.all() or not is_unknown.any():
        return float("nan")
    return float(roc_auc_score(is_unknown, score))


def fpr_at_95_tpr(score, is_unknown) -> float:
    """Fraction of unknown clips wrongly accepted as known when the threshold keeps 95 % of known clips."""
    score, is_unknown = np.asarray(score), np.asarray(is_unknown, bool)
    t = np.percentile(score[~is_unknown], 95)
    return float(np.mean(score[is_unknown] <= t))


def open_set_accuracy(pred_class, score, threshold, y_true_known_idx) -> float:
    """Correct = unknown clip flagged as unknown, or known clip accepted AND given the right class."""
    y, flagged = np.asarray(y_true_known_idx), np.asarray(score) > threshold
    ok = np.where(y == -1, flagged, (~flagged) & (np.asarray(pred_class) == y))
    return float(np.mean(ok))


# ---------- regression ----------
def regression_report(y_true, y_pred, bands: list[float]) -> dict:
    err = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    same_band = np.digitize(y_true, bands) == np.digitize(y_pred, bands)
    return {"mae": float(err.mean()), "median_ae": float(np.median(err)),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "band_accuracy": float(same_band.mean())}
