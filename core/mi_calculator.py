import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import warnings


def _compute_mi_pair(col_i, col_j, data_i, data_j, mi_neighbors, is_classification=False):
    """Calcula MI para um par de colunas (para paralelização)."""
    mask = ~(np.isnan(data_i) | np.isnan(data_j))
    n_samples = mask.sum()
    if n_samples < 10:
        return col_i, col_j, 0.0, n_samples

    X = data_i[mask].reshape(-1, 1)
    y = data_j[mask]

    try:
        if is_classification:
            mi = mutual_info_classif(X, y.astype(int), random_state=42)[0]
        else:
            mi = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]
        return col_i, col_j, mi, n_samples
    except:
        return col_i, col_j, 0.0, n_samples


def calculate_mi_fast(data: pd.DataFrame, columns: list = None):
    """
    Calcula matriz de 'MI' usando correlação de Spearman (muito mais rápido).
    Boa aproximação para relações monótonas.
    """
    if columns is None:
        columns = data.columns.tolist()

    n_cols = len(columns)
    mi_matrix = np.eye(n_cols)

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col_i, col_j = columns[i], columns[j]
            mask = ~(data[col_i].isna() | data[col_j].isna())
            if mask.sum() >= 10:
                try:
                    corr, _ = spearmanr(data.loc[mask, col_i], data.loc[mask, col_j])
                    # Converter correlação para "pseudo-MI" (0 a 1)
                    mi_approx = abs(corr) if np.isfinite(corr) else 0.0
                    mi_matrix[i, j] = mi_approx
                    mi_matrix[j, i] = mi_approx
                except:
                    pass

    return pd.DataFrame(mi_matrix, index=columns, columns=columns)


def calculate_mi_numeric(data: pd.DataFrame, numeric_cols: list, min_samples: int = 10,
                         mi_neighbors: int = 3, n_jobs: int = -1):
    """Versão paralelizada do cálculo de MI para dados numéricos."""
    n_cols = len(numeric_cols)
    mi_matrix = np.eye(n_cols)
    total_rows = len(data)

    # Preparar dados como arrays numpy para eficiência
    data_arrays = {col: data[col].values for col in numeric_cols}

    # Criar lista de pares para processar
    pairs = [(i, j, numeric_cols[i], numeric_cols[j])
             for i in range(n_cols) for j in range(i + 1, n_cols)]

    if len(pairs) == 0:
        return pd.DataFrame(mi_matrix, index=numeric_cols, columns=numeric_cols)

    # Processar em paralelo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_mi_pair)(
                col_i, col_j,
                data_arrays[col_i], data_arrays[col_j],
                mi_neighbors, False
            ) for i, j, col_i, col_j in pairs
        )

    # Preencher matriz
    col_to_idx = {col: idx for idx, col in enumerate(numeric_cols)}
    for col_i, col_j, mi_raw, n_samples in results:
        if n_samples >= min_samples:
            confidence_weight = min(n_samples / total_rows, 1.0)
            mi_weighted = mi_raw * confidence_weight
            i, j = col_to_idx[col_i], col_to_idx[col_j]
            mi_matrix[i, j] = mi_weighted
            mi_matrix[j, i] = mi_weighted

    return pd.DataFrame(mi_matrix, index=numeric_cols, columns=numeric_cols)

def _compute_mi_pair_mixed(col_i, col_j, data_i, data_j, scaled_i, scaled_j,
                            col_i_type, col_j_type, mi_neighbors, min_samples):
    """
    Calcula MI para um par de colunas de tipos mistos (para paralelização).
    col_type: 'numeric', 'binary', 'nominal', 'ordinal'
    """
    mask = ~(np.isnan(data_i) | np.isnan(data_j))
    n_samples = mask.sum()
    if n_samples < min_samples:
        return col_i, col_j, 0.0, n_samples

    try:
        # Ambos numéricos
        if col_i_type == 'numeric' and col_j_type == 'numeric':
            X = scaled_i[mask].reshape(-1, 1)
            y = scaled_j[mask]
            mi = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]

        # Ambos categóricos (nominal ou binary)
        elif col_i_type in ('nominal', 'binary') and col_j_type in ('nominal', 'binary'):
            X = data_i[mask].reshape(-1, 1)
            y = data_j[mask].astype(int)
            mi = mutual_info_classif(X, y, random_state=42)[0]

        # Ambos ordinais
        elif col_i_type == 'ordinal' and col_j_type == 'ordinal':
            X = data_i[mask].reshape(-1, 1)
            y = data_j[mask]
            mi = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]

        # Misto
        else:
            if col_i_type == 'numeric':
                X = scaled_i[mask].reshape(-1, 1)
            else:
                X = data_i[mask].reshape(-1, 1)

            if col_j_type == 'numeric':
                y = scaled_j[mask]
            else:
                y = data_j[mask]

            mi = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]

        return col_i, col_j, mi, n_samples
    except:
        return col_i, col_j, 0.0, n_samples


def calculate_mi_mixed(encoded_data: pd.DataFrame, scaled_data: pd.DataFrame,
                       numeric_cols: list, binary_cols: list, nominal_cols: list, ordinal_cols: list,
                       mi_neighbors: int = 3, min_samples: int = 10, n_jobs: int = -1,
                       fast_mode: bool = False):
    """
    Calcula matriz de MI para dados mistos.

    Args:
        fast_mode: Se True, usa correlação de Spearman (muito mais rápido)
    """
    columns = encoded_data.columns.tolist()
    n_cols = len(columns)
    total_rows = len(encoded_data)

    # Fast mode: usar Spearman
    if fast_mode:
        return calculate_mi_fast(encoded_data, columns)

    mi_matrix = np.eye(n_cols)

    # Mapear tipos
    col_types = {}
    for col in columns:
        if col in numeric_cols:
            col_types[col] = 'numeric'
        elif col in binary_cols:
            col_types[col] = 'binary'
        elif col in nominal_cols:
            col_types[col] = 'nominal'
        elif col in ordinal_cols:
            col_types[col] = 'ordinal'
        else:
            col_types[col] = 'numeric'  # default

    # Preparar arrays
    encoded_arrays = {col: encoded_data[col].values for col in columns}
    scaled_arrays = {col: scaled_data[col].values if col in scaled_data.columns
                     else encoded_data[col].values for col in columns}

    # Criar pares
    pairs = [(columns[i], columns[j]) for i in range(n_cols) for j in range(i + 1, n_cols)]

    if len(pairs) == 0:
        return pd.DataFrame(mi_matrix, index=columns, columns=columns)

    # Processar em paralelo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_mi_pair_mixed)(
                col_i, col_j,
                encoded_arrays[col_i], encoded_arrays[col_j],
                scaled_arrays[col_i], scaled_arrays[col_j],
                col_types[col_i], col_types[col_j],
                mi_neighbors, min_samples
            ) for col_i, col_j in pairs
        )

    # Preencher matriz
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    for col_i, col_j, mi_raw, n_samples in results:
        if n_samples >= min_samples:
            confidence_weight = min(n_samples / total_rows, 1.0)
            mi_weighted = mi_raw * confidence_weight
            i, j = col_to_idx[col_i], col_to_idx[col_j]
            mi_matrix[i, j] = mi_weighted
            mi_matrix[j, i] = mi_weighted

    return pd.DataFrame(mi_matrix, index=columns, columns=columns)
