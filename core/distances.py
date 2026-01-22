import numpy as np
from numba import jit, prange
import warnings


@jit(nopython=True, parallel=True, cache=True)
def weighted_euclidean_pds(sample, reference_matrix, weights, min_overlap=3):
    """
    Distância euclidiana ponderada com Partial Distance Strategy.

    Permite donors com overlap parcial - calcula distância só nas features
    onde AMBOS (sample e donor) têm valores.

    Args:
        sample: Array (n_features,) - pode ter NaN
        reference_matrix: Matriz (n_ref x n_features) - pode ter NaN
        weights: Pesos por feature
        min_overlap: Mínimo de features em comum

    Returns:
        distances: Array (n_ref,) - np.inf se overlap < min_overlap
        n_shared: Array (n_ref,) - número de features partilhadas
    """
    n_ref = reference_matrix.shape[0]
    n_features = len(sample)
    distances = np.empty(n_ref)
    n_shared = np.empty(n_ref, dtype=np.int64)

    for i in prange(n_ref):
        dist_sq = 0.0
        weight_sum = 0.0
        count = 0

        for j in range(n_features):
            # Só usa features onde AMBOS têm valores
            if not np.isnan(sample[j]) and not np.isnan(reference_matrix[i, j]):
                diff = sample[j] - reference_matrix[i, j]
                dist_sq += weights[j] * diff * diff
                weight_sum += weights[j]
                count += 1

        n_shared[i] = count

        if count >= min_overlap and weight_sum > 0:
            # Adaptive PDS scaling (threshold=0.5):
            # - overlap < 50%: linear scaling (penalização forte)
            # - overlap >= 50%: sqrt scaling (penalização suave)
            overlap_ratio = count / n_features
            if overlap_ratio < 0.5:
                scale_factor = n_features / count  # linear
            else:
                scale_factor = np.sqrt(n_features / count)  # sqrt
            distances[i] = np.sqrt(dist_sq * scale_factor)
        else:
            distances[i] = np.inf

    return distances, n_shared


@jit(nopython=True, parallel=True, cache=True)
def mixed_distance_pds(sample, reference_matrix,
                       sample_original, reference_original,
                       numeric_mask, binary_mask,
                       ordinal_mask, nominal_mask,
                       weights, range_factors, min_overlap=3):
    """
    Distância mista com Partial Distance Strategy.

    Permite donors com overlap parcial.
    """
    n_ref = reference_matrix.shape[0]
    n_features = len(sample)
    distances = np.empty(n_ref)
    n_shared = np.empty(n_ref, dtype=np.int64)

    for i in prange(n_ref):
        weighted_dist = 0.0
        total_weight = 0.0
        count = 0

        for j in range(n_features):
            # Só usa features onde AMBOS têm valores
            s_val = sample[j]
            r_val = reference_matrix[i, j]

            if np.isnan(s_val) or np.isnan(r_val):
                continue

            count += 1
            w = weights[j]

            if numeric_mask[j]:
                raw_diff = np.abs(s_val - r_val)
                normalized_diff = raw_diff * range_factors[j]
                # Soft clipping: permite valores até 1.5 para distinguir outliers
                if normalized_diff > 1.5:
                    normalized_diff = 1.5
                contrib = normalized_diff * w
            elif ordinal_mask[j]:
                contrib = np.abs(s_val - r_val) * w
            elif binary_mask[j]:
                contrib = 0.0 if s_val == r_val else w
            elif nominal_mask[j]:
                contrib = 0.0 if sample_original[j] == reference_original[i, j] else w
            else:
                contrib = 0.0

            weighted_dist += contrib
            total_weight += w

        n_shared[i] = count

        if count >= min_overlap and total_weight > 0:
            # Adaptive PDS scaling (threshold=0.5):
            # - overlap < 50%: linear scaling (penalização forte)
            # - overlap >= 50%: sqrt scaling (penalização suave)
            overlap_ratio = count / n_features
            if overlap_ratio < 0.5:
                scale_factor = n_features / count  # linear
            else:
                scale_factor = np.sqrt(n_features / count)  # sqrt
            distances[i] = (weighted_dist / total_weight) * scale_factor
        else:
            distances[i] = np.inf

    return distances, n_shared


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
                # Soft clipping: permite valores até 1.5 para distinguir outliers
                if normalized_diff > 1.5:
                    normalized_diff = 1.5
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

@jit(nopython=True, parallel=True, cache=True)
def weighted_euclidean_multi_query(queries, reference_matrix, weights):
    """
    Calcula distâncias euclidianas ponderadas para múltiplas queries em batch.

    Args:
        queries: Matriz (n_queries x n_features) de queries
        reference_matrix: Matriz (n_ref x n_features) de referências
        weights: Vetor de pesos por feature

    Returns:
        Matriz (n_queries x n_ref) de distâncias
    """
    n_queries = queries.shape[0]
    n_ref = reference_matrix.shape[0]
    distances = np.empty((n_queries, n_ref))

    for q in prange(n_queries):
        for i in range(n_ref):
            diff = np.abs(reference_matrix[i] - queries[q])
            weighted_diff = diff * weights
            distances[q, i] = np.sqrt(np.sum(weighted_diff ** 2))

    return distances


@jit(nopython=True, parallel=True, cache=True)
def mixed_distance_multi_query(queries, reference_matrix,
                               queries_original, reference_original,
                               numeric_mask, binary_mask,
                               ordinal_mask, nominal_mask,
                               weights):
    """
    Calcula distâncias mistas para múltiplas queries em batch.
    """
    n_queries = queries.shape[0]
    n_ref = reference_matrix.shape[0]
    n_features = queries.shape[1]
    distances = np.empty((n_queries, n_ref))

    for q in prange(n_queries):
        for i in range(n_ref):
            weighted_dist = 0.0
            total_weight = 0.0
            for j in range(n_features):
                w = weights[j]
                if numeric_mask[j]:
                    contrib = np.abs(queries[q, j] - reference_matrix[i, j]) * w
                elif ordinal_mask[j]:
                    contrib = np.abs(queries[q, j] - reference_matrix[i, j]) * w
                elif binary_mask[j]:
                    contrib = 0.0 if queries[q, j] == reference_matrix[i, j] else w
                elif nominal_mask[j]:
                    contrib = 0.0 if queries_original[q, j] == reference_original[i, j] else w
                else:
                    contrib = 0.0
                weighted_dist += contrib
                total_weight += w
            distances[q, i] = weighted_dist / total_weight if total_weight > 0 else 0.0

    return distances

def _warmup_numba():
    try:
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
        dummy_range_factors = np.array([0.167,1.0])
        _ = range_normalized_mixed_distance(dummy_sample, dummy_ref,
                                           dummy_sample_orig, dummy_ref_orig,
                                           dummy_numeric_mask, dummy_binary_mask,
                                           dummy_ordinal_mask, dummy_nominal_mask,
                                           dummy_weights, dummy_range_factors)
        # Warmup multi-query functions
        dummy_queries = np.array([[0.5,0.3],[0.2,0.8]])
        _ = weighted_euclidean_multi_query(dummy_queries, dummy_ref, dummy_weights)
        dummy_queries_orig = np.array([[1.0,0.0],[0.0,1.0]])
        _ = mixed_distance_multi_query(dummy_queries, dummy_ref,
                                       dummy_queries_orig, dummy_ref_orig,
                                       dummy_numeric_mask, dummy_binary_mask,
                                       dummy_ordinal_mask, dummy_nominal_mask,
                                       dummy_weights)
    except Exception as e:
        warnings.warn(f"Numba warmup failed: {e}. Performance may be slower on first run.")
