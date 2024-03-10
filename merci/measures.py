import numpy as np

import merci.exceptions


def to_probability_distribution(y: np.ndarray) -> np.ndarray:
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(y[0].shape) < 2:
        num_targets = max(np.max(y), 1) + 1
        y = np.eye(num_targets)[y.flatten()]
    return y


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if p.shape != q.shape:
        raise merci.exceptions.MerciException(
            "Input and target data must have the same length"
        )

    kl_divergence = np.where((p != 0) & (q != 0), p * np.log(p / q), 0).sum(axis=1)
    return kl_divergence


def symmetrict_kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if p.shape != q.shape:
        raise merci.exceptions.MerciException(
            "Input and target data must have the same length"
        )

    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2


def reliability_estimation(p: np.ndarray, q: np.ndarray) -> float:
    p = to_probability_distribution(p)
    q = to_probability_distribution(q)

    # for non-degenerate distributions
    divergence_sym = symmetrict_kl_divergence(p, q)

    # for degenerate distributions
    degenerate_score = (p != q).any(axis=1)

    agg_score = np.where(
        (divergence_sym == 0) & (degenerate_score), degenerate_score, divergence_sym
    )

    reliability = (2 ** (-agg_score)).mean()

    return reliability


# [1, 2, 3, 4, 5]
# [[0.2, 0.8], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.1, 0.9]
# [[1], [0], [1], [0], [1]]
