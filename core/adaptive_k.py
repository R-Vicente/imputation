import numpy as np

def adaptive_k_from_distances(distances: np.ndarray, min_friends: int = 3, max_friends: int = 15) -> int:
    valid = np.isfinite(distances)
    if not valid.any():
        return min_friends
    valid_dist = distances[valid]
    median_dist = np.median(valid_dist)
    max_dist = np.max(valid_dist)
    if max_dist < 1e-10:
        return min_friends
    trust = median_dist / max_dist
    k = int(min_friends + (max_friends - min_friends) * (1 - trust))
    return max(min_friends, min(max_friends, k, len(valid_dist)))
