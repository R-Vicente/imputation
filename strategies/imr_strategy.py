"""
Estratégia IMR primeiro, depois ISCA-k refinamento.
Usada para dados numéricos com < 5% de linhas completas.
"""
from .base_strategy import BaseStrategy


class IMRStrategy(BaseStrategy):
    """
    Estratégia que usa IMR para bootstrap inicial e ISCA-k para refinamento.
    Adequada para dados numéricos com poucas linhas completas.
    """

    def run(self, imputer, data_encoded, missing_mask, initial_missing, start_time):
        """
        Executa imputação IMR + ISCA-k refinamento.

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
        from imputers.imr_imputer import IMRInitializer
        from core.mi_calculator import calculate_mi_mixed

        result = data_encoded.copy()

        if imputer.verbose:
            print(f"\n{'='*70}")
            print("FASE 1: IMR INICIAL")
            print(f"{'='*70}")
            print(f"Missings iniciais: {initial_missing}")

        non_numeric_cols = (
            imputer.mixed_handler.binary_cols +
            imputer.mixed_handler.nominal_cols +
            imputer.mixed_handler.ordinal_cols
        )

        imr = IMRInitializer(n_iterations=3)
        result = imr.fit_transform(
            result,
            imputer.mixed_handler.numeric_cols,
            non_numeric_cols
        )

        after_imr = result.isna().sum().sum()
        if imputer.verbose:
            print(f"Missings apos IMR: {after_imr}")

        scaled_result = imputer._get_scaled_data(result, force_refit=True)

        if imputer.verbose:
            print(f"\n{'='*70}")
            print("FASE 2: ISCA-k REFINAMENTO")
            print(f"{'='*70}")

        imputer.mi_matrix = calculate_mi_mixed(
            result, scaled_result,
            imputer.mixed_handler.numeric_cols,
            imputer.mixed_handler.binary_cols,
            imputer.mixed_handler.nominal_cols,
            imputer.mixed_handler.ordinal_cols,
            mi_neighbors=imputer.mi_neighbors,
            fast_mode=imputer.fast_mode
        )

        columns_ordered = imputer._rank_columns(data_encoded)
        n_refined_per_col = {}

        for col in columns_ordered:
            col_missing_mask = missing_mask[col]
            if not col_missing_mask.any():
                continue
            n_before = col_missing_mask.sum()
            refined_series = imputer._refine_column_mixed(
                data_encoded, col, scaled_result, col_missing_mask
            )
            result.loc[col_missing_mask, col] = refined_series[col_missing_mask]
            n_after = result[col].isna().sum()
            n_refined = n_before - n_after
            n_refined_per_col[col] = n_refined

        remaining_missing = result.isna().sum().sum()

        if remaining_missing == 0:
            end_time = time.time()
            imputer.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': 0,
                'execution_time': end_time - start_time,
                'strategy': 'IMR + ISCA-k',
                'cycles': 0
            }
            if imputer.verbose:
                imputer._print_summary()
            return result

        return imputer._handle_residuals_with_imr(
            result, remaining_missing, initial_missing,
            columns_ordered, data_encoded, start_time, n_refined_per_col
        )
