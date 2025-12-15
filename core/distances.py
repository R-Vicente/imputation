import numpy as np
from numba import jit, prange
import warnings

@jit(nopython=True, parallel=True, cache=True)
def weighted_euclidean_batch(sample, reference_matrix, weights):
    n_ref = reference_matrix.shape[0]
    distances = np.empty(n_ref)
    for i in prange(n_ref):
        diff = np.abs(reference_matrix[i] - sample)
        weighted_diff = diff * weights
        distances[i] = np.sqrt(np.sum(weighted_diff ** 2))
    return distances

@jit(nopython=True, parallel=True, cache=True)
def mixed_distance_batch(sample, reference_matrix, 
                         sample_original, reference_original,
                         numeric_mask, binary_mask, 
                         ordinal_mask, nominal_mask, 
                         weights):
    n_ref = reference_matrix.shape[0]
    n_features = sample.shape[0]
    distances = np.empty(n_ref)
    for i in prange(n_ref):
        weighted_dist = 0.0
        total_weight = 0.0
        for j in range(n_features):
            w = weights[j]
            if numeric_mask[j]:
                contrib = np.abs(sample[j] - reference_matrix[i, j]) * w
            elif ordinal_mask[j]:
                contrib = np.abs(sample[j] - reference_matrix[i, j]) * w
            elif binary_mask[j]:
                contrib = 0.0 if sample[j] == reference_matrix[i, j] else w
            elif nominal_mask[j]:
                contrib = 0.0 if sample_original[j] == reference_original[i, j] else w
            else:
                contrib = 0.0
            weighted_dist += contrib
            total_weight += w
        distances[i] = weighted_dist / total_weight if total_weight > 0 else 0.0
    return distances

@jit(nopython=True, parallel=True, cache=True)
def range_normalized_mixed_distance(sample, reference_matrix, 
                                    sample_original, reference_original,
                                    numeric_mask, binary_mask, 
                                    ordinal_mask, nominal_mask, 
                                    weights, range_factors):
    n_ref = reference_matrix.shape[0]
    n_features = sample.shape[0]
    distances = np.empty(n_ref)
    for i in prange(n_ref):
        weighted_dist = 0.0
        total_weight = 0.0
        for j in range(n_features):
            w = weights[j]
            if numeric_mask[j]:
                raw_diff = np.abs(sample[j] - reference_matrix[i, j])
                normalized_diff = raw_diff * range_factors[j]
                if normalized_diff > 1.0:
                    normalized_diff = 1.0
                contrib = normalized_diff * w
            elif ordinal_mask[j]:
                normalized_diff = np.abs(sample[j] - reference_matrix[i, j])
                contrib = normalized_diff * w
            elif binary_mask[j]:
                contrib = 0.0 if sample[j] == reference_matrix[i, j] else w
            elif nominal_mask[j]:
                contrib = 0.0 if sample_original[j] == reference_original[i, j] else w
            else:
                contrib = 0.0
            weighted_dist += contrib
            total_weight += w
        distances[i] = weighted_dist / total_weight if total_weight > 0 else 0.0
    return distances

@jit(nopython=True, cache=True)
def quantile_lookup_single(friend_val, friend_sorted, target_sorted):
    friend_idx = np.searchsorted(friend_sorted, friend_val)
    quantile_pct = 100.0 * friend_idx / len(friend_sorted)
    target_idx = int(quantile_pct * len(target_sorted) / 100.0)
    target_idx = min(target_idx, len(target_sorted) - 1)
    return target_sorted[target_idx]

@jit(nopython=True, parallel=True, cache=True)
def quantile_lookup_batch(friend_vals, friend_sorted, target_sorted):
    n = len(friend_vals)
    result = np.empty(n)
    for i in prange(n):
        friend_idx = np.searchsorted(friend_sorted, friend_vals[i])
        quantile_pct = 100.0 * friend_idx / len(friend_sorted)
        target_idx = int(quantile_pct * len(target_sorted) / 100.0)
        target_idx = min(target_idx, len(target_sorted) - 1)
        result[i] = target_sorted[target_idx]
    return result

def _warmup_numba():
    try:
        dummy_friend = np.array([1.0,2.0,3.0,4.0,5.0])
        dummy_target = np.array([10.0,20.0,30.0,40.0,50.0])
        dummy_batch = np.array([1.5,3.2,4.8])
        _ = quantile_lookup_single(2.5, dummy_friend, dummy_target)
        _ = quantile_lookup_batch(dummy_batch, dummy_friend, dummy_target)
        dummy_sample = np.array([0.5,0.3])
        dummy_ref = np.array([[0.1,0.2],[0.6,0.7],[0.4,0.5]])
        dummy_weights = np.array([0.6,0.4])
        _ = weighted_euclidean_batch(dummy_sample, dummy_ref, dummy_weights)
        dummy_sample_orig = np.array([1.0,0.0])
        dummy_ref_orig = np.array([[0.0,1.0],[1.0,0.0],[1.0,1.0]])
        dummy_numeric_mask = np.array([True,False])
        dummy_binary_mask = np.array([False,True])
        dummy_ordinal_mask = np.array([False,False])
        dummy_nominal_mask = np.array([False,False])
        _ = mixed_distance_batch(dummy_sample, dummy_ref,
                                 dummy_sample_orig, dummy_ref_orig,
                                 dummy_numeric_mask, dummy_binary_mask,
                                 dummy_ordinal_mask, dummy_nominal_mask,
                                 dummy_weights)
        dummy_range_factors = np.array([0.167,1.0])
        _ = range_normalized_mixed_distance(dummy_sample, dummy_ref,
                                           dummy_sample_orig, dummy_ref_orig,
                                           dummy_numeric_mask, dummy_binary_mask,
                                           dummy_ordinal_mask, dummy_nominal_mask,
                                           dummy_weights, dummy_range_factors)
    except Exception as e:
        warnings.warn(f"Numba warmup failed: {e}. Performance may be slower on first run.")
