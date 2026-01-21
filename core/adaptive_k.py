import numpy as np


def adaptive_k_hybrid(distances: np.ndarray, neighbor_values: np.ndarray,
                      min_k: int = 3, max_k: int = 15,
                      alpha: float = 0.5, is_categorical: bool = False) -> int:
    """
    Seleção adaptativa de k baseada em densidade local E consistência dos vizinhos.

    Lógica:
    - Região densa + vizinhos consistentes → mais k é seguro (estimativa estável)
    - Região esparsa + vizinhos inconsistentes → menos k (não diluir informação boa)

    Args:
        distances: Distâncias a todos os donors potenciais
        neighbor_values: Valores target de todos os donors potenciais
        min_k: Número mínimo de vizinhos
        max_k: Número máximo de vizinhos
        alpha: Peso densidade vs consistência (0=só consistência, 1=só densidade)
        is_categorical: Se o target é categórico

    Returns:
        Valor óptimo de k
    """
    n_donors = len(distances)
    if n_donors <= min_k:
        return n_donors

    # Usar os max_k mais próximos para avaliar
    k_eval = min(max_k, n_donors)
    closest_idx = np.argpartition(distances, k_eval - 1)[:k_eval]
    closest_distances = distances[closest_idx]
    closest_values = neighbor_values[closest_idx]

    # === DENSIDADE LOCAL ===
    # Distâncias pequenas = região densa = density_trust alto
    valid_distances = closest_distances[np.isfinite(closest_distances)]
    if len(valid_distances) == 0:
        return min_k  # Fallback se não há distâncias válidas
    mean_dist = np.mean(valid_distances)
    density_trust = 1.0 / (1.0 + mean_dist) if np.isfinite(mean_dist) else 0.5

    # === CONSISTÊNCIA DOS VIZINHOS ===
    if is_categorical:
        # Para categóricos: frequência da moda
        # Se todos concordam → consistency_trust = 1
        # Se dispersos → consistency_trust baixo
        unique, counts = np.unique(closest_values[~np.isnan(closest_values)], return_counts=True)
        if len(counts) > 0:
            mode_freq = counts.max() / len(closest_values)
            consistency_trust = mode_freq
        else:
            consistency_trust = 0.5
    else:
        # Para numéricos: inverso do coeficiente de variação
        # CV baixo (valores similares) → consistency_trust alto
        valid_values = closest_values[~np.isnan(closest_values)]
        if len(valid_values) > 1:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            if abs(mean_val) > 1e-10:
                cv = std_val / abs(mean_val)
            else:
                # Se média ~0, usar std normalizado pelo range
                val_range = np.ptp(valid_values)
                cv = std_val / (val_range + 1e-10) if val_range > 0 else std_val
            consistency_trust = 1.0 / (1.0 + cv)
        else:
            consistency_trust = 0.5

    # === TRUST COMBINADO ===
    trust = alpha * density_trust + (1 - alpha) * consistency_trust

    # Garantir que trust é válido
    if not np.isfinite(trust):
        trust = 0.5  # Fallback para valor médio

    # Alta confiança (denso + consistente) → MAIS vizinhos é seguro
    # Baixa confiança (esparso + inconsistente) → MENOS vizinhos
    k = int(min_k + (max_k - min_k) * trust)
    k = max(min_k, min(max_k, k, n_donors))

    return k
