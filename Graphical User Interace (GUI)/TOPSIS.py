import numpy as np


def topsis(data, weights):
    if len(data) == 0:
        return np.array([])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    norm_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    weighted_data = norm_data * weights
    ideal_best = np.min(weighted_data, axis=0)
    ideal_worst = np.max(weighted_data, axis=0)
    dist_best = np.sqrt(np.sum((weighted_data - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_data - ideal_worst) ** 2, axis=1))
    scores = dist_worst / (dist_best + dist_worst + 1e-10)
    return scores
    