"""
Estratégia ISCA-k pura.
Usada quando há >= 5% de linhas completas.
"""
from .base_strategy import BaseStrategy


class ISCAkStrategy(BaseStrategy):
    """
    Estratégia que usa apenas ISCA-k para imputação.
    Adequada para datasets com linhas completas suficientes.
    """

    def run(self, imputer, data_encoded, missing_mask, initial_missing, start_time):
        """
        Executa imputação ISCA-k pura.

        Args:
            imputer: Instância de ISCAkCore com configurações
            data_encoded: DataFrame com dados encoded
            missing_mask: Máscara booleana de valores missing
            initial_missing: Contagem inicial de missings
            start_time: Timestamp de início

        Returns:
            DataFrame com valores imputados
        """
        import time
        from core.mi_calculator import calculate_mi_mixed
        from core.fuzzy_clustering import FuzzyClusterIndex

        result = data_encoded.copy()
        n_steps = 4 if imputer.use_fcm else 3

        phase_name = "ISCA-k + PDS" if imputer._effective_pds else "ISCA-k"

        if imputer.verbose:
            print(f"\n{'='*70}")
            print(f"FASE 1: {phase_name}")
            print(f"{'='*70}")

        scaled_data = imputer._get_scaled_data(result)

        if imputer.verbose:
            print(f"  [1/{n_steps}] Calculando Informacao Mutua...")

        imputer.mi_matrix = calculate_mi_mixed(
            data_encoded, scaled_data,
            imputer.mixed_handler.numeric_cols,
            imputer.mixed_handler.binary_cols,
            imputer.mixed_handler.nominal_cols,
            imputer.mixed_handler.ordinal_cols,
            mi_neighbors=imputer.mi_neighbors,
            fast_mode=imputer.fast_mode
        )

        # Fit FCM-PDS para acelerar busca de vizinhos
        if imputer.use_fcm:
            if imputer.verbose:
                print(f"  [2/{n_steps}] Fitting Fuzzy C-Means...")

            # Ajustar n_clusters ao tamanho do dataset
            n_samples = len(scaled_data)
            effective_clusters = min(imputer.n_clusters, n_samples // 5)
            effective_clusters = max(effective_clusters, 3)

            imputer.fcm_index = FuzzyClusterIndex(
                n_clusters=effective_clusters,
                membership_threshold=imputer.fcm_membership_threshold,
                verbose=False,
                random_state=42
            )
            imputer.fcm_index.fit(scaled_data.values)

        if imputer.verbose:
            step = 3 if imputer.use_fcm else 2
            print(f"  [{step}/{n_steps}] Ordenando colunas por facilidade...")

        columns_ordered = imputer._rank_columns(result)

        if imputer.verbose:
            step = 4 if imputer.use_fcm else 3
            print(f"  [{step}/{n_steps}] Imputando colunas...")

        n_imputed_per_col = {}
        for col in columns_ordered:
            if not result[col].isna().any():
                continue
            n_before = result[col].isna().sum()
            result[col] = imputer._impute_column_mixed(result, col, scaled_data)
            n_after = result[col].isna().sum()
            n_imputed = n_before - n_after
            n_imputed_per_col[col] = n_imputed

        remaining_missing = result.isna().sum().sum()
        progress = initial_missing - remaining_missing
        pct_progress = (progress / initial_missing * 100) if initial_missing > 0 else 0

        if imputer.verbose:
            print(f"\n  Resultado: {initial_missing} → {remaining_missing} missings")
            print(f"             {progress} imputados ({pct_progress:.1f}%)")

        if remaining_missing == 0:
            end_time = time.time()
            imputer.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': 0,
                'execution_time': end_time - start_time,
                'strategy': phase_name,
                'phases': [
                    {'name': phase_name, 'before': initial_missing, 'after': 0}
                ],
                'phase2_activated': False,
                'phase2_cycles': 0,
                'phase2_imputed': 0
            }
            if imputer.verbose:
                imputer._print_summary()
            return result

        # Se ainda há missings, trata residuais com modo iterativo
        phase1_stats = {'name': phase_name, 'before': initial_missing, 'after': remaining_missing}
        return imputer._handle_residuals_iterative(
            result, remaining_missing, initial_missing,
            columns_ordered, data_encoded, start_time, n_imputed_per_col, phase1_stats
        )
