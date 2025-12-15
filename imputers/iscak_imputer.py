import numpy as np
import pandas as pd
import time
from pathlib import Path
from collections import defaultdict

from preprocessing.type_detection import MixedDataHandler
from preprocessing.scaling import get_scaled_data, compute_range_factors
from core.mi_calculator import calculate_mi_mixed
from core.distances import weighted_euclidean_batch, range_normalized_mixed_distance
from core.adaptive_k import adaptive_k_from_distances

class ISCAkCore:
    def __init__(self, min_friends: int = 3, max_friends: int = 15, 
                 mi_neighbors: int = 3, n_jobs: int = -1, verbose: bool = True,
                 max_cycles: int = 3, categorical_threshold: int = 10):
        self.min_friends = min_friends
        self.max_friends = max_friends
        self.mi_neighbors = mi_neighbors
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_cycles = max_cycles
        self.scaler = None
        self.mi_matrix = None
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
        missing_mask = data_encoded.isna()
        initial_missing = missing_mask.sum().sum()
        if self.verbose:
            self._print_header(data_encoded)
        complete_rows = (~missing_mask).all(axis=1).sum()
        pct_complete_rows = complete_rows / len(data) * 100
        if self.verbose:
            print(f"\\nLinhas 100% completas: {complete_rows}/{len(data)} ({pct_complete_rows:.1f}%)")
        # DECISÃO DE ESTRATÉGIA BASEADA NO TIPO DE DADOS
        n_numeric = len(self.mixed_handler.numeric_cols)
        n_categorical = (len(self.mixed_handler.binary_cols) + 
                        len(self.mixed_handler.nominal_cols) + 
                        len(self.mixed_handler.ordinal_cols))
        
        # DADOS PURAMENTE NUMÉRICOS
        if n_categorical == 0:
            if pct_complete_rows >= 5.0:
                if self.verbose:
                    print(f"Estratégia: ISCA-k puro (dados numéricos, >= 5% linhas completas)")
                result_encoded = self._strategy_iscak_first(data_encoded, missing_mask, 
                                                           initial_missing, start_time)
            else:
                if self.verbose:
                    print(f"Estratégia: IMR -> ISCA-k (dados numéricos, < 5% linhas completas)")
                result_encoded = self._strategy_imr_first(data_encoded, missing_mask, 
                                                         initial_missing, start_time)
        
        # DADOS MISTOS OU PURAMENTE CATEGÓRICOS
        else:
            if pct_complete_rows >= 5.0:
                if self.verbose:
                    print(f"Estratégia: ISCA-k puro (dados mistos, >= 5% linhas completas)")
                result_encoded = self._strategy_iscak_first(data_encoded, missing_mask, 
                                                           initial_missing, start_time)
            else:
                if self.verbose:
                    print(f"Estratégia: Mediana/Moda -> ISCA-k (dados mistos, < 5% linhas completas)")
                result_encoded = self._strategy_simple_bootstrap_first(data_encoded, missing_mask,
                                                                       initial_missing, start_time)
        result = self.mixed_handler.inverse_transform(result_encoded)
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
                    # CORRIGIDO: Sem inplace
                    result.loc[:, col] = result[col].fillna(median_val)
                else:
                    result.loc[:, col] = result[col].fillna(0)
        
        # Binárias: moda
        for col in self.mixed_handler.binary_cols:
            if result[col].isna().any():
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    # CORRIGIDO: Sem inplace
                    result.loc[:, col] = result[col].fillna(mode_val[0])
                else:
                    result.loc[:, col] = result[col].fillna(0)
        
        # Nominais: moda
        for col in self.mixed_handler.nominal_cols:
            if result[col].isna().any():
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    # CORRIGIDO: Sem inplace
                    result.loc[:, col] = result[col].fillna(mode_val[0])
                else:
                    result.loc[:, col] = result[col].fillna(0)
        
        # Ordinais: mediana (já em escala [0,1])
        for col in self.mixed_handler.ordinal_cols:
            if result[col].isna().any():
                median_val = result[col].median()
                if not pd.isna(median_val):
                    # CORRIGIDO: Sem inplace
                    result.loc[:, col] = result[col].fillna(median_val)
                else:
                    result.loc[:, col] = result[col].fillna(0.5)
        
        return result

    def _get_scaled_data(self, data: pd.DataFrame, force_refit: bool = False):
        return get_scaled_data(data, self.mixed_handler, cache=self._scaled_cache, force_refit=force_refit)

    def _compute_range_factors(self, data: pd.DataFrame, scaled_data: pd.DataFrame):
        return compute_range_factors(data, scaled_data, self.mixed_handler, verbose=self.verbose)

    def _strategy_iscak_first(self, data_encoded, missing_mask, initial_missing, start_time):
        result = data_encoded.copy()
        if self.verbose:
            print(f"\\n{'='*70}")
            print("FASE 1: ISCA-k PURO")
            print(f"{'='*70}")
            print(f"Missings iniciais: {initial_missing}")
        scaled_data = self._get_scaled_data(result)
        if self.verbose:
            print("[1/3] Calculando Informacao Mutua...")
        self.mi_matrix = calculate_mi_mixed(data_encoded, scaled_data,
                                            self.mixed_handler.numeric_cols,
                                            self.mixed_handler.binary_cols,
                                            self.mixed_handler.nominal_cols,
                                            self.mixed_handler.ordinal_cols,
                                            mi_neighbors=self.mi_neighbors)
        if self.verbose:
            print("[2/3] Ordenando colunas por facilidade...")
        columns_ordered = self._rank_columns(result)
        if self.verbose:
            print(f"      Ordem: {', '.join(columns_ordered[:5])}{'...' if len(columns_ordered) > 5 else ''}")
        if self.verbose:
            print("[3/3] Imputando colunas...")
        n_imputed_per_col = {}
        for col in columns_ordered:
            if not result[col].isna().any():
                continue
            n_before = result[col].isna().sum()
            result[col] = self._impute_column_mixed(result, col, scaled_data)
            n_after = result[col].isna().sum()
            n_imputed = n_before - n_after
            n_imputed_per_col[col] = n_imputed
            if self.verbose and n_imputed > 0:
                print(f"      {col}: {n_imputed}/{n_before} imputados")
        remaining_missing = result.isna().sum().sum()
        progress = initial_missing - remaining_missing
        if self.verbose:
            print(f"\\nProgresso: -{progress} missings ({remaining_missing} restantes)")
        if remaining_missing == 0:
            end_time = time.time()
            self.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': 0,
                'execution_time': end_time - start_time,
                'strategy': 'ISCA-k puro',
                'cycles': 0
            }
            if self.verbose:
                self._print_summary()
            return result
        return self._handle_residuals_with_imr(result, remaining_missing, initial_missing, 
                                               columns_ordered, data_encoded, start_time, n_imputed_per_col)

    def _strategy_imr_first(self, data_encoded, missing_mask, initial_missing, start_time):
        from imputers.imr_imputer import IMRInitializer
        result = data_encoded.copy()
        if self.verbose:
            print(f"\\n{'='*70}")
            print("FASE 1: IMR INICIAL")
            print(f"{'='*70}")
            print(f"Missings iniciais: {initial_missing}")
        non_numeric_cols = (self.mixed_handler.binary_cols + 
                           self.mixed_handler.nominal_cols + 
                           self.mixed_handler.ordinal_cols)
        imr = IMRInitializer(n_iterations=3)
        result = imr.fit_transform(
            result, 
            self.mixed_handler.numeric_cols,
            non_numeric_cols
        )
        after_imr = result.isna().sum().sum()
        if self.verbose:
            print(f"Missings apos IMR: {after_imr}")
        scaled_result = self._get_scaled_data(result, force_refit=True)
        if self.verbose:
            print(f"\\n{'='*70}")
            print("FASE 2: ISCA-k REFINAMENTO")
            print(f"{'='*70}")
        self.mi_matrix = calculate_mi_mixed(result, scaled_result,
                                            self.mixed_handler.numeric_cols,
                                            self.mixed_handler.binary_cols,
                                            self.mixed_handler.nominal_cols,
                                            self.mixed_handler.ordinal_cols,
                                            mi_neighbors=self.mi_neighbors)
        columns_ordered = self._rank_columns(data_encoded)
        n_refined_per_col = {}
        for col in columns_ordered:
            col_missing_mask = missing_mask[col]
            if not col_missing_mask.any():
                continue
            n_before = col_missing_mask.sum()
            refined_series = self._refine_column_mixed(data_encoded, col, scaled_result, col_missing_mask)
            result.loc[col_missing_mask, col] = refined_series[col_missing_mask]
            n_after = result[col].isna().sum()
            n_refined = n_before - n_after
            n_refined_per_col[col] = n_refined
        remaining_missing = result.isna().sum().sum()
        if remaining_missing == 0:
            end_time = time.time()
            self.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': 0,
                'execution_time': end_time - start_time,
                'strategy': 'IMR + ISCA-k',
                'cycles': 0
            }
            if self.verbose:
                self._print_summary()
            return result
        return self._handle_residuals_with_imr(result, remaining_missing, initial_missing,
                                               columns_ordered, data_encoded, start_time, n_refined_per_col)

    def _strategy_simple_bootstrap_first(self, data_encoded, missing_mask, 
                                    initial_missing, start_time):
        """
        Estratégia: Bootstrap simples -> ISCA-k refinamento.
        
        USADA PARA: Dados mistos com < 5% linhas completas.
        SUBSTITUI: IMR (que não funciona para categóricas).
        """
        result = data_encoded.copy()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("FASE 1: BOOTSTRAP SIMPLES (MEDIANA/MODA)")
            print(f"{'='*70}")
            print(f"Missings iniciais: {initial_missing}")
        
        # Bootstrap simples
        result = self._simple_bootstrap(result)
        
        after_bootstrap = result.isna().sum().sum()
        if self.verbose:
            print(f"Missings após bootstrap: {after_bootstrap}")
        
        if after_bootstrap == 0:
            # Bootstrap completou tudo, agora refina com ISCA-k
            scaled_result = self._get_scaled_data(result, force_refit=True)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print("FASE 2: ISCA-k REFINAMENTO")
                print(f"{'='*70}")
            
            self.mi_matrix = calculate_mi_mixed(result, scaled_result,
                                                self.mixed_handler.numeric_cols,
                                                self.mixed_handler.binary_cols,
                                                self.mixed_handler.nominal_cols,
                                                self.mixed_handler.ordinal_cols,
                                                mi_neighbors=self.mi_neighbors)
            
            columns_ordered = self._rank_columns(data_encoded)
            
            for col in columns_ordered:
                col_missing_mask = missing_mask[col]
                if not col_missing_mask.any():
                    continue
                
                refined_series = self._refine_column_mixed(data_encoded, col, 
                                                          scaled_result, col_missing_mask)
                result.loc[col_missing_mask, col] = refined_series[col_missing_mask]
            
            remaining_missing = result.isna().sum().sum()
            
            end_time = time.time()
            self.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': remaining_missing,
                'execution_time': end_time - start_time,
                'strategy': 'Simple Bootstrap + ISCA-k',
                'cycles': 0
            }
            
            if self.verbose:
                self._print_summary()
            
            return result
        else:
            # Fallback se bootstrap falhou (muito raro)
            if self.verbose:
                print(f"AVISO: Bootstrap não completou ({after_bootstrap} missings restantes)")
            
            end_time = time.time()
            self.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': after_bootstrap,
                'execution_time': end_time - start_time,
                'strategy': 'Simple Bootstrap (incompleto)',
                'cycles': 0
            }
            
            return result

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
        result = data[target_col].copy()
        missing_indices = data[target_col].isna()
        complete_mask = ~missing_indices
        if complete_mask.sum() == 0:
            return result
        feature_cols = [c for c in data.columns if c != target_col]
        mi_scores = self.mi_matrix.loc[feature_cols, target_col]
        weights = mi_scores.values
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        range_factors_full = self._compute_range_factors(data, scaled_data)
        numeric_mask = np.array([col in self.mixed_handler.numeric_cols for col in feature_cols], dtype=np.bool_)
        binary_mask = np.array([col in self.mixed_handler.binary_cols for col in feature_cols], dtype=np.bool_)
        ordinal_mask = np.array([col in self.mixed_handler.ordinal_cols for col in feature_cols], dtype=np.bool_)
        nominal_mask = np.array([col in self.mixed_handler.nominal_cols for col in feature_cols], dtype=np.bool_)
        X_ref_scaled = scaled_data.loc[complete_mask, feature_cols].values
        X_ref_original = data.loc[complete_mask, feature_cols].values
        y_ref = data.loc[complete_mask, target_col].values
        is_target_binary = target_col in self.mixed_handler.binary_cols
        is_target_nominal = target_col in self.mixed_handler.nominal_cols
        is_target_ordinal = target_col in self.mixed_handler.ordinal_cols
        is_target_categorical = is_target_binary or is_target_nominal or is_target_ordinal
        for idx in data[missing_indices].index:
            available = [c for c in feature_cols if not pd.isna(data.loc[idx, c])]
            if len(available) == 0:
                continue
            avail_indices = [feature_cols.index(c) for c in available]
            sample_scaled = scaled_data.loc[idx, available].values
            sample_original = data.loc[idx, available].values
            X_ref_scaled_sub = X_ref_scaled[:, avail_indices]
            X_ref_original_sub = X_ref_original[:, avail_indices]
            data_col_indices = [data.columns.get_loc(c) for c in available]
            range_factors_sub = range_factors_full[data_col_indices]
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
                distances = weighted_euclidean_batch(sample_scaled, X_ref_scaled_sub, weights_sub)
            else:
                distances = range_normalized_mixed_distance(
                    sample_scaled, X_ref_scaled_sub,
                    sample_original, X_ref_original_sub,
                    numeric_mask_sub, binary_mask_sub,
                    ordinal_mask_sub, nominal_mask_sub,
                    weights_sub, range_factors_sub
                )
            k = adaptive_k_from_distances(distances, self.min_friends, self.max_friends)
            k = min(k, len(distances))
            if k == 0:
                continue
            friend_idx = np.argpartition(distances, k-1)[:k] if k < len(distances) else np.arange(len(distances))
            friend_values = y_ref[friend_idx]
            friend_distances = distances[friend_idx]
            if is_target_categorical:
                if len(friend_values) == 1:
                    result.loc[idx] = friend_values[0]
                else:
                    weighted_votes = {}
                    for val, dist in zip(friend_values, friend_distances):
                        weight = 1 / (dist + 1e-6)
                        weighted_votes[val] = weighted_votes.get(val, 0) + weight
                    result.loc[idx] = max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                if np.any(friend_distances < 1e-10):
                    exact_mask = friend_distances < 1e-10
                    result.loc[idx] = np.mean(friend_values[exact_mask])
                else:
                    w = 1 / (friend_distances + 1e-6)
                    w = w / w.sum()
                    result.loc[idx] = np.average(friend_values, weights=w)
        return result

    def _refine_column_mixed(self, original_data: pd.DataFrame, target_col: str, 
                             scaled_complete_df: pd.DataFrame, refine_mask_col: pd.Series) -> pd.Series:
        original_complete_mask = ~original_data[target_col].isna()
        if original_complete_mask.sum() == 0:
            return pd.Series(np.nan, index=original_data.index)
        feature_cols = [c for c in original_data.columns if c != target_col]
        mi_scores = self.mi_matrix.loc[feature_cols, target_col]
        weights = mi_scores.values
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        range_factors_full = self._compute_range_factors(original_data, scaled_complete_df)
        numeric_mask = np.array([col in self.mixed_handler.numeric_cols for col in feature_cols], dtype=np.bool_)
        binary_mask = np.array([col in self.mixed_handler.binary_cols for col in feature_cols], dtype=np.bool_)
        ordinal_mask = np.array([col in self.mixed_handler.ordinal_cols for col in feature_cols], dtype=np.bool_)
        nominal_mask = np.array([col in self.mixed_handler.nominal_cols for col in feature_cols], dtype=np.bool_)
        refined = pd.Series(np.nan, index=original_data.index)
        X_ref_scaled = scaled_complete_df.loc[original_complete_mask, feature_cols].values
        X_ref_original = original_data.loc[original_complete_mask, feature_cols].values
        y_ref = original_data.loc[original_complete_mask, target_col].values
        is_target_categorical = (target_col in self.mixed_handler.binary_cols or 
                                target_col in self.mixed_handler.nominal_cols or
                                target_col in self.mixed_handler.ordinal_cols)
        for idx in refine_mask_col[refine_mask_col].index:
            available = [c for c in feature_cols if not pd.isna(original_data.loc[idx, c])]
            if len(available) == 0:
                continue
            avail_indices = [feature_cols.index(c) for c in available]
            sample_scaled = scaled_complete_df.loc[idx, available].values
            sample_original = original_data.loc[idx, available].values
            X_ref_scaled_sub = X_ref_scaled[:, avail_indices]
            X_ref_original_sub = X_ref_original[:, avail_indices]
            data_col_indices = [original_data.columns.get_loc(c) for c in available]
            range_factors_sub = range_factors_full[data_col_indices]
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
                distances = weighted_euclidean_batch(sample_scaled, X_ref_scaled_sub, weights_sub)
            else:
                distances = range_normalized_mixed_distance(
                    sample_scaled, X_ref_scaled_sub,
                    sample_original, X_ref_original_sub,
                    numeric_mask_sub, binary_mask_sub,
                    ordinal_mask_sub, nominal_mask_sub,
                    weights_sub, range_factors_sub
                )
            k = adaptive_k_from_distances(distances, self.min_friends, self.max_friends)
            k = min(k, len(distances))
            if k == 0:
                continue
            friend_idx = np.argpartition(distances, k-1)[:k] if k < len(distances) else np.arange(len(distances))
            friend_values = y_ref[friend_idx]
            friend_distances = distances[friend_idx]
            if is_target_categorical:
                if len(friend_values) == 1:
                    refined.loc[idx] = friend_values[0]
                else:
                    weighted_votes = {}
                    for val, dist in zip(friend_values, friend_distances):
                        weight = 1 / (dist + 1e-6)
                        weighted_votes[val] = weighted_votes.get(val, 0) + weight
                    refined.loc[idx] = max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                if np.any(friend_distances < 1e-10):
                    exact_mask = friend_distances < 1e-10
                    refined.loc[idx] = np.mean(friend_values[exact_mask])
                else:
                    w = 1 / (friend_distances + 1e-6)
                    w = w / w.sum()
                    refined.loc[idx] = np.average(friend_values, weights=w)
        return refined

    def _print_header(self, data: pd.DataFrame):
        print("\\n" + "="*70)
        print("ISCA-k: Information-theoretic Smart Collaborative Approach".center(70))
        print("="*70)
        print(f"\\nDataset: {data.shape[0]} x {data.shape[1]}")
        print(f"Missings: {data.isna().sum().sum()} ({data.isna().sum().sum()/data.size*100:.1f}%)")
        print(f"Parametros: min_friends={self.min_friends}, max_friends={self.max_friends}")
        print(f"MI neighbors: {self.mi_neighbors}")
        print(f"Max cycles: {self.max_cycles}")
        if self.mixed_handler.is_mixed:
            print(f"\\nTipo dados: Misto")
            print(f"  Numericas: {len(self.mixed_handler.numeric_cols)}")
            print(f"  Binarias: {len(self.mixed_handler.binary_cols)}")
            print(f"  Nominais: {len(self.mixed_handler.nominal_cols)}")
            print(f"  Ordinais: {len(self.mixed_handler.ordinal_cols)}")

    def _print_summary(self):
        stats = self.execution_stats
        print("\\n" + "="*70)
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
        print("="*70 + "\\n")
