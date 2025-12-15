import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def calculate_mi_numeric(data: pd.DataFrame, numeric_cols: list, min_samples: int = 10, mi_neighbors: int = 3):
    n_cols = len(numeric_cols)
    mi_matrix = np.eye(n_cols)
    total_rows = len(data)
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            col_i, col_j = numeric_cols[i], numeric_cols[j]
            mask = ~(data[col_i].isna() | data[col_j].isna())
            n_samples_used = mask.sum()
            if n_samples_used >= min_samples:
                X = data.loc[mask, col_i].values.reshape(-1,1)
                y = data.loc[mask, col_j].values
                try:
                    mi_raw = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]
                    confidence_weight = min(n_samples_used / total_rows, 1.0)
                    mi_weighted = mi_raw * confidence_weight
                    mi_matrix[i, j] = mi_weighted
                    mi_matrix[j, i] = mi_weighted
                except:
                    pass
    return pd.DataFrame(mi_matrix, index=numeric_cols, columns=numeric_cols)

def calculate_mi_mixed(encoded_data: pd.DataFrame, scaled_data: pd.DataFrame,
                       numeric_cols: list, binary_cols: list, nominal_cols: list, ordinal_cols: list,
                       mi_neighbors: int = 3, min_samples: int = 10):
    columns = encoded_data.columns.tolist()
    n_cols = len(columns)
    mi_matrix = np.eye(n_cols)
    total_rows = len(encoded_data)
    numeric_set = set(numeric_cols)
    binary_set = set(binary_cols)
    nominal_set = set(nominal_cols)
    ordinal_set = set(ordinal_cols)
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            col_i, col_j = columns[i], columns[j]
            mask = ~(encoded_data[col_i].isna() | encoded_data[col_j].isna())
            if mask.sum() >= min_samples:
                try:
                    if col_i in numeric_set and col_j in numeric_set:
                        X = scaled_data.loc[mask, col_i].values.reshape(-1, 1)
                        y = scaled_data.loc[mask, col_j].values
                        mi = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]
                    elif ((col_i in nominal_set or col_i in binary_set) and 
                          (col_j in nominal_set or col_j in binary_set)):
                        X = encoded_data.loc[mask, col_i].values.reshape(-1, 1)
                        y = encoded_data.loc[mask, col_j].values.astype(int)
                        mi = mutual_info_classif(X, y, random_state=42)[0]
                    elif col_i in ordinal_set and col_j in ordinal_set:
                        X = encoded_data.loc[mask, col_i].values.reshape(-1, 1)
                        y = encoded_data.loc[mask, col_j].values
                        mi = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]
                    else:
                        if col_i in numeric_set:
                            X = scaled_data.loc[mask, col_i].values.reshape(-1, 1)
                            y = encoded_data.loc[mask, col_j].values
                        else:
                            X = encoded_data.loc[mask, col_i].values.reshape(-1, 1)
                            if col_j in numeric_set:
                                y = scaled_data.loc[mask, col_j].values
                            else:
                                y = encoded_data.loc[mask, col_j].values
                        mi = mutual_info_regression(X, y, n_neighbors=mi_neighbors, random_state=42)[0]
                    confidence_weight = min(mask.sum() / total_rows, 1.0)
                    mi_weighted = mi * confidence_weight
                    mi_matrix[i, j] = mi_weighted
                    mi_matrix[j, i] = mi_weighted
                except:
                    pass
    return pd.DataFrame(mi_matrix, index=columns, columns=columns)
