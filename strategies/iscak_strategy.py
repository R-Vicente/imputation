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

        result = data_encoded.copy()

        if imputer.verbose:
            print(f"\n{'='*70}")
            print("FASE 1: ISCA-k PURO")
            print(f"{'='*70}")
            print(f"Missings iniciais: {initial_missing}")

        scaled_data = imputer._get_scaled_data(result)

        if imputer.verbose:
            print("[1/3] Calculando Informacao Mutua...")

        imputer.mi_matrix = calculate_mi_mixed(
            data_encoded, scaled_data,
            imputer.mixed_handler.numeric_cols,
            imputer.mixed_handler.binary_cols,
            imputer.mixed_handler.nominal_cols,
            imputer.mixed_handler.ordinal_cols,
            mi_neighbors=imputer.mi_neighbors
        )

        if imputer.verbose:
            print("[2/3] Ordenando colunas por facilidade...")

        columns_ordered = imputer._rank_columns(result)

        if imputer.verbose:
            print(f"      Ordem: {', '.join(columns_ordered[:5])}{'...' if len(columns_ordered) > 5 else ''}")
            print("[3/3] Imputando colunas...")

        n_imputed_per_col = {}
        for col in columns_ordered:
            if not result[col].isna().any():
                continue
            n_before = result[col].isna().sum()
            result[col] = imputer._impute_column_mixed(result, col, scaled_data)
            n_after = result[col].isna().sum()
            n_imputed = n_before - n_after
            n_imputed_per_col[col] = n_imputed
            if imputer.verbose and n_imputed > 0:
                print(f"      {col}: {n_imputed}/{n_before} imputados")

        remaining_missing = result.isna().sum().sum()
        progress = initial_missing - remaining_missing

        if imputer.verbose:
            print(f"\nProgresso: -{progress} missings ({remaining_missing} restantes)")

        if remaining_missing == 0:
            end_time = time.time()
            imputer.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': 0,
                'execution_time': end_time - start_time,
                'strategy': 'ISCA-k puro',
                'cycles': 0
            }
            if imputer.verbose:
                imputer._print_summary()
            return result

        # Se ainda há missings, trata residuais
        return imputer._handle_residuals_with_imr(
            result, remaining_missing, initial_missing,
            columns_ordered, data_encoded, start_time, n_imputed_per_col
        )
