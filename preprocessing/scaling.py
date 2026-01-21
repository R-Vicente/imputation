import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_scaled_data(data: pd.DataFrame, mixed_handler, cache=None, force_refit: bool = False):
    scaler = StandardScaler()
    cache_key = (data.shape, id(data))
    if cache is None or cache.get("key") != cache_key or force_refit:
        scaled_data = data.copy()
        if len(mixed_handler.numeric_cols) > 0:
            scaled_values = scaler.fit_transform(data[mixed_handler.numeric_cols])
            scaled_data[mixed_handler.numeric_cols] = scaled_values
        for col in mixed_handler.nominal_cols:
            n_categories = mixed_handler.nominal_mappings[col]['n_categories']
            if n_categories > 1:
                codes = pd.Categorical(data[col]).codes
                scaled_data[col] = codes / (n_categories - 1)
            else:
                scaled_data[col] = 0.0
        if cache is not None:
            cache["data"] = scaled_data
            cache["key"] = cache_key
        return scaled_data
    return cache["data"]

def compute_range_factors(data: pd.DataFrame, scaled_data: pd.DataFrame, mixed_handler, verbose: bool = False):
    n_features = len(data.columns)
    range_factors = np.ones(n_features)
    if verbose:
        print("\\n=== DIAGNÓSTICO: RANGE FACTORS (EMPÍRICO) ===")
    for idx, col in enumerate(data.columns):
        if col in mixed_handler.numeric_cols:
            col_values = scaled_data[col].dropna()
            if len(col_values) > 1:
                min_val = col_values.min()
                max_val = col_values.max()
                empirical_range = max_val - min_val
                if empirical_range > 1e-6:
                    range_factors[idx] = 1.0 / empirical_range
                    if verbose:
                        print(f"  {col}: range=[{min_val:.2f}, {max_val:.2f}] = {empirical_range:.2f}")
                else:
                    # Coluna constante: não contribui para distância (range_factor=0)
                    range_factors[idx] = 0.0
                    if verbose:
                        print(f"  {col}: CONSTANTE (range_factor=0.0, ignorada)")
            else:
                # Apenas 1 valor: não contribui para distância
                range_factors[idx] = 0.0
                if verbose:
                    print(f"  {col}: APENAS 1 VALOR (range_factor=0.0, ignorada)")
        else:
            if verbose:
                tipo = 'binary' if col in mixed_handler.binary_cols else 'ordinal' if col in mixed_handler.ordinal_cols else 'nominal'
                print(f"  {col}: {tipo} (range_factor=1.0)")
    if verbose:
        print("=== FIM DIAGNÓSTICO ===\\n")
    return range_factors
