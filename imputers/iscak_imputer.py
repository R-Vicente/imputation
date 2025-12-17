"""
ISCA-k: Information-theoretic Smart Collaborative Approach with adaptive k

Método híbrido de imputação baseado em:
- Informação Mútua para ponderação de variáveis
- Fuzzy C-Means com PDS para acelerar busca de vizinhos
- Seleção dinâmica de "amigos" (vizinhos) baseada em densidade local e consistência
"""
import numpy as np
import pandas as pd
import time
from pathlib import Path

from preprocessing.type_detection import MixedDataHandler
from preprocessing.scaling import get_scaled_data, compute_range_factors
from core.mi_calculator import calculate_mi_mixed
from core.distances import (weighted_euclidean_batch, range_normalized_mixed_distance,
                            weighted_euclidean_multi_query, mixed_distance_multi_query,
                            weighted_euclidean_pds, mixed_distance_pds)
from core.adaptive_k import adaptive_k_hybrid
from core.fuzzy_clustering import FuzzyClusterIndex


class ISCAkCore:
    def __init__(self, min_friends: int = 3, max_friends: int = 15,
                 mi_neighbors: int = 3, n_jobs: int = -1, verbose: bool = True,
                 max_cycles: int = 3, categorical_threshold: int = 10,
                 adaptive_k_alpha: float = 0.5, fast_mode: bool = False,
                 use_fcm: bool = False, n_clusters: int = 8,
                 n_top_clusters: int = 3, fcm_membership_threshold: float = 0.05,
                 use_pds: bool = True, min_overlap: int = None):
        """
        Args:
            min_friends: Número mínimo de vizinhos (k_min)
            max_friends: Número máximo de vizinhos (k_max)
            mi_neighbors: Vizinhos para estimativa de MI
            n_jobs: Paralelização (-1 = todos os cores)
            verbose: Mostrar progresso
            max_cycles: Máximo de ciclos para residuais
            categorical_threshold: Limite para detectar categóricas
            adaptive_k_alpha: Peso densidade vs consistência (0=só consistência, 1=só densidade)
            fast_mode: Se True, usa Spearman em vez de MI (muito mais rápido)
            use_fcm: Se True, usa Fuzzy C-Means para acelerar busca de vizinhos
            n_clusters: Número de clusters para FCM
            n_top_clusters: Número de clusters a considerar na busca
            fcm_membership_threshold: Threshold mínimo de membership
            use_pds: Se True, usa Partial Distance Strategy (permite donors com overlap parcial)
            min_overlap: Mínimo de features em comum (default: max(3, n_features//3))
        """
        self.min_friends = min_friends
        self.max_friends = max_friends
        self.mi_neighbors = mi_neighbors
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_cycles = max_cycles
        self.adaptive_k_alpha = adaptive_k_alpha
        self.fast_mode = fast_mode
        self.use_fcm = use_fcm
        self.n_clusters = n_clusters
        self.n_top_clusters = n_top_clusters
        self.fcm_membership_threshold = fcm_membership_threshold
        self.use_pds = use_pds
        self._min_overlap_user = min_overlap  # Guardado para calcular depois
        self.min_overlap = min_overlap  # Será ajustado no impute()
        self.scaler = None
        self.mi_matrix = None
        self.fcm_index = None  # FuzzyClusterIndex
        self.execution_stats = {}
        self.mixed_handler = MixedDataHandler(categorical_threshold=categorical_threshold)
        self.encoding_info = None
        self._scaled_cache = {}
        self._cache_key = None

    def impute(self, data: pd.DataFrame,
               force_categorical: list = None,
               force_ordinal: dict = None,
               interactive: bool = True,
               column_types_config: str = None) -> pd.DataFrame:
        start_time = time.time()
        original_data = data.copy()
        if column_types_config and Path(column_types_config).exists():
            force_categorical, force_ordinal = MixedDataHandler.load_config(column_types_config)
        data_encoded, self.encoding_info = self.mixed_handler.fit_transform(
            original_data,
            force_categorical=force_categorical,
            force_ordinal=force_ordinal,
            interactive=interactive,
            verbose=self.verbose
        )

        # Calcular min_overlap automático se não especificado
        if self._min_overlap_user is None:
            n_features = data_encoded.shape[1]
            # Fórmula adaptativa:
            # - Poucos features (<=6): 50% mas min de 2 (menos restritivo para evitar fallback)
            # - Médio (7-15): ~40% (equilíbrio)
            # - Muitos features (>15): 50% (mais restritivo para manter qualidade)
            if n_features <= 6:
                self.min_overlap = max(2, n_features // 2)
            elif n_features <= 15:
                self.min_overlap = max(3, int(n_features * 0.4))
            else:
                self.min_overlap = max(5, int(n_features * 0.5))
        else:
            self.min_overlap = self._min_overlap_user

        missing_mask = data_encoded.isna()
        initial_missing = missing_mask.sum().sum()
        if self.verbose:
            self._print_header(data_encoded)
        complete_rows = (~missing_mask).all(axis=1).sum()
        pct_complete_rows = complete_rows / len(data) * 100
        if self.verbose:
            print(f"\nLinhas 100% completas: {complete_rows}/{len(data)} ({pct_complete_rows:.1f}%)")

        # Seleccionar e executar estratégia apropriada
        result_encoded = self._select_and_run_strategy(
            data_encoded, missing_mask, initial_missing,
            pct_complete_rows, start_time
        )

        result = self.mixed_handler.inverse_transform(result_encoded)
        return result

    def _select_and_run_strategy(self, data_encoded, missing_mask,
                                  initial_missing, pct_complete_rows, start_time):
        """
        Nova estratégia unificada:
        1. Sempre tentar ISCA-k+PDS primeiro (funciona mesmo com poucas linhas completas)
        2. Se restarem missings: aplicar IMR/Bootstrap como fallback
        3. Refinar com ISCA-k
        """
        from strategies.iscak_strategy import ISCAkStrategy

        n_categorical = (len(self.mixed_handler.binary_cols) +
                        len(self.mixed_handler.nominal_cols) +
                        len(self.mixed_handler.ordinal_cols))

        if self.verbose:
            if self.use_pds:
                print(f"Estratégia: ISCA-k+PDS primeiro, fallback se necessário")
            else:
                print(f"Estratégia: ISCA-k clássico")

        # FASE 1: Sempre tentar ISCA-k+PDS primeiro
        strategy = ISCAkStrategy()
        result = strategy.run(self, data_encoded, missing_mask, initial_missing, start_time)

        # FASE 2: Se restarem missings, usar fallback
        remaining = result.isna().sum().sum()
        if remaining > 0:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"FALLBACK: {remaining} missings restantes após ISCA-k+PDS")
                print(f"{'='*70}")

            if n_categorical == 0:
                # Dados numéricos: usar IMR
                result = self._apply_imr_fallback(result, data_encoded, start_time, initial_missing)
            else:
                # Dados mistos: usar mediana/moda
                result = self._apply_bootstrap_fallback(result, data_encoded, start_time, initial_missing)

        return result

    def _apply_imr_fallback(self, result, original_data, start_time, initial_missing):
        """Aplica IMR como fallback e refina com ISCA-k."""
        import time
        from imputers.imr_imputer import IMRInitializer

        if self.verbose:
            print("Aplicando IMR para preencher gaps...")

        imr = IMRInitializer(n_iterations=5)
        non_numeric = (self.mixed_handler.binary_cols +
                      self.mixed_handler.nominal_cols +
                      self.mixed_handler.ordinal_cols)
        result = imr.fit_transform(result, self.mixed_handler.numeric_cols, non_numeric)

        after_imr = result.isna().sum().sum()
        if self.verbose:
            print(f"Missings após IMR: {after_imr}")

        if after_imr == 0:
            # Refinar com ISCA-k
            if self.verbose:
                print("Refinando com ISCA-k...")
            scaled_result = self._get_scaled_data(result, force_refit=True)
            columns_ordered = self._rank_columns(original_data)
            residual_mask = original_data.isna() & ~result.isna()

            for col in columns_ordered:
                if residual_mask[col].any():
                    refined = self._refine_column_mixed(original_data, col, scaled_result, residual_mask[col])
                    result.loc[residual_mask[col], col] = refined[residual_mask[col]]

        end_time = time.time()
        self.execution_stats = {
            'initial_missing': initial_missing,
            'final_missing': result.isna().sum().sum(),
            'execution_time': end_time - start_time,
            'strategy': 'ISCA-k+PDS → IMR → Refinamento',
            'cycles': 1
        }
        if self.verbose:
            self._print_summary()

        return result

    def _apply_bootstrap_fallback(self, result, original_data, start_time, initial_missing):
        """Aplica mediana/moda como fallback e refina com ISCA-k."""
        import time

        if self.verbose:
            print("Aplicando mediana/moda para preencher gaps...")

        result = self._simple_bootstrap(result)

        after_bootstrap = result.isna().sum().sum()
        if self.verbose:
            print(f"Missings após bootstrap: {after_bootstrap}")

        if after_bootstrap == 0:
            # Refinar com ISCA-k
            if self.verbose:
                print("Refinando com ISCA-k...")
            scaled_result = self._get_scaled_data(result, force_refit=True)
            columns_ordered = self._rank_columns(original_data)
            residual_mask = original_data.isna() & ~result.isna()

            for col in columns_ordered:
                if residual_mask[col].any():
                    refined = self._refine_column_mixed(original_data, col, scaled_result, residual_mask[col])
                    result.loc[residual_mask[col], col] = refined[residual_mask[col]]

        end_time = time.time()
        self.execution_stats = {
            'initial_missing': initial_missing,
            'final_missing': result.isna().sum().sum(),
            'execution_time': end_time - start_time,
            'strategy': 'ISCA-k+PDS → Bootstrap → Refinamento',
            'cycles': 1
        }
        if self.verbose:
            self._print_summary()

        return result

    def _simple_bootstrap(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Bootstrap simples respeitando tipos de variáveis.

        USADO PARA: Dados mistos com < 5% linhas completas.
        ALTERNATIVA AO IMR que não funciona para categóricas.

        - Numéricas: Mediana
        - Binárias: Moda
        - Nominais: Moda
        - Ordinais: Mediana (em valores scaled [0,1])

        Returns:
            DataFrame com todos os missings preenchidos
        """
        result = data.copy()

        # Numéricas: mediana
        for col in self.mixed_handler.numeric_cols:
            if result[col].isna().any():
                median_val = result[col].median()
                if not pd.isna(median_val):
                    result.loc[:, col] = result[col].fillna(median_val)
                else:
                    result.loc[:, col] = result[col].fillna(0)

        # Binárias: moda
        for col in self.mixed_handler.binary_cols:
            if result[col].isna().any():
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result.loc[:, col] = result[col].fillna(mode_val[0])
                else:
                    result.loc[:, col] = result[col].fillna(0)

        # Nominais: moda
        for col in self.mixed_handler.nominal_cols:
            if result[col].isna().any():
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result.loc[:, col] = result[col].fillna(mode_val[0])
                else:
                    result.loc[:, col] = result[col].fillna(0)

        # Ordinais: mediana (já em escala [0,1])
        for col in self.mixed_handler.ordinal_cols:
            if result[col].isna().any():
                median_val = result[col].median()
                if not pd.isna(median_val):
                    result.loc[:, col] = result[col].fillna(median_val)
                else:
                    result.loc[:, col] = result[col].fillna(0.5)

        return result

    def _get_scaled_data(self, data: pd.DataFrame, force_refit: bool = False):
        return get_scaled_data(data, self.mixed_handler, cache=self._scaled_cache, force_refit=force_refit)

    def _compute_range_factors(self, data: pd.DataFrame, scaled_data: pd.DataFrame):
        return compute_range_factors(data, scaled_data, self.mixed_handler, verbose=False)

    def _handle_residuals_with_imr(self, result, remaining_missing, initial_missing,
                                   columns_ordered, original_data, start_time, prev_stats):
        from imputers.imr_imputer import IMRInitializer
        cycle = 0
        prev_progress = float('inf')
        non_numeric_cols = (self.mixed_handler.binary_cols +
                           self.mixed_handler.nominal_cols +
                           self.mixed_handler.ordinal_cols)
        while remaining_missing > 0 and cycle < self.max_cycles:
            cycle += 1
            imr = IMRInitializer(n_iterations=3)
            result = imr.fit_transform(
                result,
                self.mixed_handler.numeric_cols,
                non_numeric_cols
            )
            after_imr = result.isna().sum().sum()
            if after_imr > 0:
                break
            scaled_result = self._get_scaled_data(result, force_refit=True)
            residual_mask = original_data.isna() & ~result.isna()
            for col in columns_ordered:
                if residual_mask[col].any():
                    refined = self._refine_column_mixed(original_data, col, scaled_result, residual_mask[col])
                    result.loc[residual_mask[col], col] = refined[residual_mask[col]]
            new_remaining = result.isna().sum().sum()
            cycle_progress = remaining_missing - new_remaining
            if cycle_progress == 0 or (cycle > 1 and cycle_progress < prev_progress * 0.1):
                break
            prev_progress = cycle_progress
            remaining_missing = new_remaining
        end_time = time.time()
        self.execution_stats = {
            'initial_missing': initial_missing,
            'final_missing': remaining_missing,
            'execution_time': end_time - start_time,
            'cycles': cycle
        }
        if self.verbose:
            self._print_summary()
        return result

    def _rank_columns(self, data: pd.DataFrame) -> list:
        scores = []
        for col in data.columns:
            if not data[col].isna().any():
                continue
            pct_missing = data[col].isna().mean()
            mi_with_others = self.mi_matrix[col].drop(col)
            avg_mi = mi_with_others.mean()
            score = pct_missing / (avg_mi + 0.01)
            scores.append((col, score))
        scores.sort(key=lambda x: x[1])
        return [col for col, _ in scores]

    def _impute_column_mixed(self, data: pd.DataFrame, target_col: str, scaled_data: pd.DataFrame) -> pd.Series:
        """
        Imputa valores em falta para uma coluna.

        Quando use_pds=True: usa Partial Distance Strategy, permitindo donors
        com overlap parcial (mais robusto para datasets com muitos missings).

        Quando use_pds=False: requer que donors tenham todas as features do query.
        """
        result = data[target_col].copy()
        missing_mask = data[target_col].isna()
        complete_mask = ~missing_mask
        if complete_mask.sum() == 0:
            return result

        feature_cols = [c for c in data.columns if c != target_col]
        mi_scores = self.mi_matrix.loc[feature_cols, target_col]
        weights = mi_scores.values.astype(np.float64)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        range_factors_full = self._compute_range_factors(data, scaled_data)

        numeric_mask = np.array([col in self.mixed_handler.numeric_cols for col in feature_cols], dtype=np.bool_)
        binary_mask = np.array([col in self.mixed_handler.binary_cols for col in feature_cols], dtype=np.bool_)
        ordinal_mask = np.array([col in self.mixed_handler.ordinal_cols for col in feature_cols], dtype=np.bool_)
        nominal_mask = np.array([col in self.mixed_handler.nominal_cols for col in feature_cols], dtype=np.bool_)
        has_categorical = binary_mask.any() or nominal_mask.any() or ordinal_mask.any()

        # Converter para numpy ANTES do loop
        X_ref_scaled = scaled_data.loc[complete_mask, feature_cols].values.astype(np.float64)
        X_ref_original = data.loc[complete_mask, feature_cols].values.astype(np.float64)
        y_ref = data.loc[complete_mask, target_col].values
        ref_indices = np.where(complete_mask.values)[0]

        X_all_scaled = scaled_data[feature_cols].values.astype(np.float64)
        X_all_original = data[feature_cols].values.astype(np.float64)

        is_target_binary = target_col in self.mixed_handler.binary_cols
        is_target_nominal = target_col in self.mixed_handler.nominal_cols
        is_target_ordinal = target_col in self.mixed_handler.ordinal_cols
        is_target_categorical = is_target_binary or is_target_nominal or is_target_ordinal

        missing_row_indices = np.where(missing_mask.values)[0]
        if len(missing_row_indices) == 0:
            return result

        result_values = result.values.copy()
        range_factors = range_factors_full[[data.columns.get_loc(c) for c in feature_cols]].astype(np.float64)

        for row_idx in missing_row_indices:
            sample_scaled = X_all_scaled[row_idx]
            sample_original = X_all_original[row_idx]

            # === MODO PDS: Permite donors com overlap parcial ===
            if self.use_pds:
                if not has_categorical:
                    distances, n_shared = weighted_euclidean_pds(
                        sample_scaled, X_ref_scaled, weights, self.min_overlap
                    )
                else:
                    distances, n_shared = mixed_distance_pds(
                        sample_scaled, X_ref_scaled,
                        sample_original, X_ref_original,
                        numeric_mask, binary_mask,
                        ordinal_mask, nominal_mask,
                        weights, range_factors, self.min_overlap
                    )

                # Filtrar por overlap mínimo (distâncias infinitas)
                valid_mask = np.isfinite(distances)
                if valid_mask.sum() < self.min_friends:
                    continue

                distances_valid = distances[valid_mask]
                y_ref_valid = y_ref[valid_mask]

            # === MODO CLÁSSICO: Requer overlap completo ===
            else:
                avail_mask = ~np.isnan(sample_scaled)
                avail_indices = np.where(avail_mask)[0]
                if len(avail_indices) < self.min_overlap:
                    continue

                sample_scaled_sub = sample_scaled[avail_indices]
                sample_original_sub = sample_original[avail_indices]
                X_ref_scaled_sub = X_ref_scaled[:, avail_indices]
                X_ref_original_sub = X_ref_original[:, avail_indices]

                valid_donors_mask = ~np.isnan(X_ref_scaled_sub).any(axis=1)
                if valid_donors_mask.sum() < self.min_friends:
                    continue

                X_ref_valid = X_ref_scaled_sub[valid_donors_mask]
                X_ref_orig_valid = X_ref_original_sub[valid_donors_mask]
                y_ref_valid = y_ref[valid_donors_mask]

                weights_sub = weights[avail_indices].copy()
                if weights_sub.sum() > 0:
                    weights_sub = weights_sub / weights_sub.sum()
                else:
                    weights_sub = np.ones_like(weights_sub) / len(weights_sub)

                numeric_mask_sub = numeric_mask[avail_indices]
                binary_mask_sub = binary_mask[avail_indices]
                ordinal_mask_sub = ordinal_mask[avail_indices]
                nominal_mask_sub = nominal_mask[avail_indices]
                has_cat_sub = binary_mask_sub.any() or nominal_mask_sub.any() or ordinal_mask_sub.any()

                if not has_cat_sub:
                    distances_valid = weighted_euclidean_batch(sample_scaled_sub, X_ref_valid, weights_sub)
                else:
                    range_factors_sub = range_factors[avail_indices]
                    distances_valid = range_normalized_mixed_distance(
                        sample_scaled_sub, X_ref_valid,
                        sample_original_sub, X_ref_orig_valid,
                        numeric_mask_sub, binary_mask_sub,
                        ordinal_mask_sub, nominal_mask_sub,
                        weights_sub, range_factors_sub
                    )

            # Adaptive k e imputação (comum aos dois modos)
            k = adaptive_k_hybrid(
                distances_valid, y_ref_valid,
                min_k=self.min_friends, max_k=self.max_friends,
                alpha=self.adaptive_k_alpha, is_categorical=is_target_categorical
            )
            k = min(k, len(distances_valid))
            if k == 0:
                continue

            friend_idx = np.argpartition(distances_valid, k-1)[:k] if k < len(distances_valid) else np.arange(len(distances_valid))
            friend_values = y_ref_valid[friend_idx]
            friend_distances = distances_valid[friend_idx]

            if is_target_categorical:
                if len(friend_values) == 1:
                    result_values[row_idx] = friend_values[0]
                else:
                    weighted_votes = {}
                    for val, dist in zip(friend_values, friend_distances):
                        weight = 1 / (dist + 1e-6)
                        weighted_votes[val] = weighted_votes.get(val, 0) + weight
                    result_values[row_idx] = max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                if np.any(friend_distances < 1e-10):
                    exact_mask = friend_distances < 1e-10
                    result_values[row_idx] = np.mean(friend_values[exact_mask])
                else:
                    w = 1 / (friend_distances + 1e-6)
                    w = w / w.sum()
                    result_values[row_idx] = np.average(friend_values, weights=w)

        return pd.Series(result_values, index=result.index, name=result.name)

    def _refine_column_mixed(self, original_data: pd.DataFrame, target_col: str,
                             scaled_complete_df: pd.DataFrame, refine_mask_col: pd.Series) -> pd.Series:
        """
        Refina valores imputados.
        Optimizado: converte para numpy antes do loop para evitar overhead pandas.
        """
        original_complete_mask = ~original_data[target_col].isna()
        if original_complete_mask.sum() == 0:
            return pd.Series(np.nan, index=original_data.index)

        feature_cols = [c for c in original_data.columns if c != target_col]
        feature_col_indices = [original_data.columns.get_loc(c) for c in feature_cols]
        mi_scores = self.mi_matrix.loc[feature_cols, target_col]
        weights = mi_scores.values
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        range_factors_full = self._compute_range_factors(original_data, scaled_complete_df)

        numeric_mask = np.array([col in self.mixed_handler.numeric_cols for col in feature_cols], dtype=np.bool_)
        binary_mask = np.array([col in self.mixed_handler.binary_cols for col in feature_cols], dtype=np.bool_)
        ordinal_mask = np.array([col in self.mixed_handler.ordinal_cols for col in feature_cols], dtype=np.bool_)
        nominal_mask = np.array([col in self.mixed_handler.nominal_cols for col in feature_cols], dtype=np.bool_)

        # Converter para numpy ANTES do loop
        X_ref_scaled = scaled_complete_df.loc[original_complete_mask, feature_cols].values
        X_ref_original = original_data.loc[original_complete_mask, feature_cols].values
        y_ref = original_data.loc[original_complete_mask, target_col].values

        X_all_scaled = scaled_complete_df[feature_cols].values
        X_all_original = original_data[feature_cols].values

        is_target_categorical = (target_col in self.mixed_handler.binary_cols or
                                target_col in self.mixed_handler.nominal_cols or
                                target_col in self.mixed_handler.ordinal_cols)

        # Usar índices numéricos
        refine_row_indices = np.where(refine_mask_col.values)[0]
        if len(refine_row_indices) == 0:
            return pd.Series(np.nan, index=original_data.index)

        # Array para resultados
        refined_values = np.full(len(original_data), np.nan)

        for row_idx in refine_row_indices:
            row_scaled = X_all_scaled[row_idx]
            avail_mask = ~np.isnan(row_scaled)
            avail_indices = np.where(avail_mask)[0]
            if len(avail_indices) == 0:
                continue

            sample_scaled = row_scaled[avail_indices]
            sample_original = X_all_original[row_idx, avail_indices]

            X_ref_scaled_sub = X_ref_scaled[:, avail_indices]
            X_ref_original_sub = X_ref_original[:, avail_indices]

            valid_donors_mask = ~np.isnan(X_ref_scaled_sub).any(axis=1)
            if valid_donors_mask.sum() < self.min_friends:
                continue

            X_ref_valid = X_ref_scaled_sub[valid_donors_mask]
            X_ref_orig_valid = X_ref_original_sub[valid_donors_mask]
            y_ref_valid = y_ref[valid_donors_mask]

            weights_sub = weights[avail_indices].copy()
            if weights_sub.sum() > 0:
                weights_sub = weights_sub / weights_sub.sum()
            else:
                weights_sub = np.ones_like(weights_sub) / len(weights_sub)

            numeric_mask_sub = numeric_mask[avail_indices]
            binary_mask_sub = binary_mask[avail_indices]
            ordinal_mask_sub = ordinal_mask[avail_indices]
            nominal_mask_sub = nominal_mask[avail_indices]
            has_categorical = binary_mask_sub.any() or nominal_mask_sub.any() or ordinal_mask_sub.any()

            if not has_categorical:
                distances = weighted_euclidean_batch(sample_scaled, X_ref_valid, weights_sub)
            else:
                range_factors_sub = range_factors_full[feature_col_indices][avail_indices]
                distances = range_normalized_mixed_distance(
                    sample_scaled, X_ref_valid,
                    sample_original, X_ref_orig_valid,
                    numeric_mask_sub, binary_mask_sub,
                    ordinal_mask_sub, nominal_mask_sub,
                    weights_sub, range_factors_sub
                )

            k = adaptive_k_hybrid(
                distances, y_ref_valid,
                min_k=self.min_friends, max_k=self.max_friends,
                alpha=self.adaptive_k_alpha, is_categorical=is_target_categorical
            )
            k = min(k, len(distances))
            if k == 0:
                continue

            friend_idx = np.argpartition(distances, k-1)[:k] if k < len(distances) else np.arange(len(distances))
            friend_values = y_ref_valid[friend_idx]
            friend_distances = distances[friend_idx]

            if is_target_categorical:
                if len(friend_values) == 1:
                    refined_values[row_idx] = friend_values[0]
                else:
                    weighted_votes = {}
                    for val, dist in zip(friend_values, friend_distances):
                        weight = 1 / (dist + 1e-6)
                        weighted_votes[val] = weighted_votes.get(val, 0) + weight
                    refined_values[row_idx] = max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                if np.any(friend_distances < 1e-10):
                    exact_mask = friend_distances < 1e-10
                    refined_values[row_idx] = np.mean(friend_values[exact_mask])
                else:
                    w = 1 / (friend_distances + 1e-6)
                    w = w / w.sum()
                    refined_values[row_idx] = np.average(friend_values, weights=w)

        return pd.Series(refined_values, index=original_data.index)

    def _print_header(self, data: pd.DataFrame):
        print("\n" + "="*70)
        print("ISCA-k: Information-theoretic Smart Collaborative Approach".center(70))
        print("="*70)
        print(f"\nDataset: {data.shape[0]} x {data.shape[1]}")
        print(f"Missings: {data.isna().sum().sum()} ({data.isna().sum().sum()/data.size*100:.1f}%)")
        print(f"Parametros: min_friends={self.min_friends}, max_friends={self.max_friends}")
        print(f"MI neighbors: {self.mi_neighbors}")
        print(f"Adaptive k alpha: {self.adaptive_k_alpha}")
        print(f"Fast mode: {self.fast_mode}")
        print(f"FCM clustering: {self.use_fcm}")
        if self.use_fcm:
            print(f"  Clusters: {self.n_clusters}, Top: {self.n_top_clusters}")
        print(f"PDS (partial donors): {self.use_pds}")
        if self.use_pds:
            print(f"  Min overlap: {self.min_overlap} features")
        print(f"Max cycles: {self.max_cycles}")
        if self.mixed_handler.is_mixed:
            print(f"\nTipo dados: Misto")
            print(f"  Numericas: {len(self.mixed_handler.numeric_cols)}")
            print(f"  Binarias: {len(self.mixed_handler.binary_cols)}")
            print(f"  Nominais: {len(self.mixed_handler.nominal_cols)}")
            print(f"  Ordinais: {len(self.mixed_handler.ordinal_cols)}")

    def _print_summary(self):
        stats = self.execution_stats
        print("\n" + "="*70)
        print("RESULTADO")
        print("="*70)
        print(f"Estrategia: {stats.get('strategy', 'N/A')}")
        print(f"Inicial:  {stats['initial_missing']} missings")
        print(f"Final:    {stats['final_missing']} missings")
        if stats['final_missing'] == 0:
            print("SUCESSO: Dataset 100% completo")
        else:
            print(f"ATENCAO: {stats['final_missing']} missings NAO foram imputados")
        if stats['final_missing'] < stats['initial_missing']:
            taxa = (1 - stats['final_missing']/stats['initial_missing'])*100
            print(f"Taxa:     {taxa:.1f}%")
        print(f"Ciclos:   {stats.get('cycles', 0)}")
        print(f"Tempo:    {stats['execution_time']:.2f}s")
        print("="*70 + "\n")
