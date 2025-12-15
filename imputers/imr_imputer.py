import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_selection import mutual_info_regression

from core.distances import quantile_lookup_batch

class IMRInitializer:
    def __init__(self, n_iterations: int = 3, min_samples: int = 10):
        self.n_iterations = n_iterations
        self.min_samples = min_samples

    def fit_transform(self, data: pd.DataFrame, numeric_cols: list, non_numeric_cols: list) -> pd.DataFrame:
        result = data.copy()
        for col in non_numeric_cols:
            if result[col].isna().any():
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col].fillna(mode_val[0], inplace=True)
                else:
                    result[col].fillna(0, inplace=True)
        for iteration in range(self.n_iterations):
            missings_before = result[numeric_cols].isna().sum().sum()
            if missings_before == 0:
                break
            mi_matrix = self._calculate_mi_numeric(result, numeric_cols)
            col_stats = {}
            for col in numeric_cols:
                available = result[col].dropna()
                if len(available) > 0:
                    col_stats[col] = {'values_sorted': np.sort(available.values), 'n': len(available)}
            for col in numeric_cols:
                missing_mask = result[col].isna()
                if not missing_mask.any():
                    continue
                missings_by_friend = defaultdict(list)
                friend_values_by_group = defaultdict(list)
                for idx in result[missing_mask].index:
                    available_features = [c for c in numeric_cols if c != col and not pd.isna(result.loc[idx, c])]
                    if len(available_features) == 0:
                        continue
                    mi_scores = mi_matrix.loc[available_features, col]
                    best_friend = mi_scores.idxmax()
                    friend_value = result.loc[idx, best_friend]
                    missings_by_friend[best_friend].append(idx)
                    friend_values_by_group[best_friend].append(friend_value)
                for best_friend, indices in missings_by_friend.items():
                    if best_friend not in col_stats or col not in col_stats:
                        continue
                    friend_vals = np.array(friend_values_by_group[best_friend])
                    friend_sorted = col_stats[best_friend]['values_sorted']
                    target_sorted = col_stats[col]['values_sorted']
                    imputed_vals = quantile_lookup_batch(friend_vals, friend_sorted, target_sorted)
                    result.loc[indices, col] = imputed_vals
            missings_after = result[numeric_cols].isna().sum().sum()
            if missings_after >= missings_before:
                break
        for col in numeric_cols:
            if result[col].isna().any():
                median_val = result[col].median()
                if not pd.isna(median_val):
                    result[col].fillna(median_val, inplace=True)
                else:
                    result[col].fillna(0, inplace=True)
        return result

    def _calculate_mi_numeric(self, data: pd.DataFrame, numeric_cols: list):
        n_cols = len(numeric_cols)
        mi_matrix = np.eye(n_cols)
        total_rows = len(data)
        for i in range(n_cols):
            for j in range(i+1, n_cols):
                col_i, col_j = numeric_cols[i], numeric_cols[j]
                mask = ~(data[col_i].isna() | data[col_j].isna())
                n_samples_used = mask.sum()
                if n_samples_used >= self.min_samples:
                    X = data.loc[mask, col_i].values.reshape(-1, 1)
                    y = data.loc[mask, col_j].values
                    try:
                        mi_raw = mutual_info_regression(X, y, n_neighbors=3, random_state=42)[0]
                        confidence_weight = min(n_samples_used / total_rows, 1.0)
                        mi_weighted = mi_raw * confidence_weight
                        mi_matrix[i, j] = mi_weighted
                        mi_matrix[j, i] = mi_weighted
                    except:
                        pass
        return pd.DataFrame(mi_matrix, index=numeric_cols, columns=numeric_cols)
