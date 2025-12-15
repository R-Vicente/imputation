"""
Estratégia Bootstrap simples (mediana/moda) seguido de ISCA-k refinamento.
Usada para dados mistos com < 5% de linhas completas.
"""
from .base_strategy import BaseStrategy


class BootstrapStrategy(BaseStrategy):
    """
    Estratégia que usa mediana/moda para bootstrap inicial e ISCA-k para refinamento.
    Adequada para dados mistos com poucas linhas completas.
    """

    def run(self, imputer, data_encoded, missing_mask, initial_missing, start_time):
        """
        Executa imputação Bootstrap + ISCA-k refinamento.

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
            print("FASE 1: BOOTSTRAP SIMPLES (MEDIANA/MODA)")
            print(f"{'='*70}")
            print(f"Missings iniciais: {initial_missing}")

        # Bootstrap simples
        result = imputer._simple_bootstrap(result)

        after_bootstrap = result.isna().sum().sum()
        if imputer.verbose:
            print(f"Missings após bootstrap: {after_bootstrap}")

        if after_bootstrap == 0:
            # Bootstrap completou tudo, agora refina com ISCA-k
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
                mi_neighbors=imputer.mi_neighbors
            )

            columns_ordered = imputer._rank_columns(data_encoded)

            for col in columns_ordered:
                col_missing_mask = missing_mask[col]
                if not col_missing_mask.any():
                    continue

                refined_series = imputer._refine_column_mixed(
                    data_encoded, col, scaled_result, col_missing_mask
                )
                result.loc[col_missing_mask, col] = refined_series[col_missing_mask]

            remaining_missing = result.isna().sum().sum()

            end_time = time.time()
            imputer.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': remaining_missing,
                'execution_time': end_time - start_time,
                'strategy': 'Simple Bootstrap + ISCA-k',
                'cycles': 0
            }

            if imputer.verbose:
                imputer._print_summary()

            return result
        else:
            # Fallback se bootstrap falhou (muito raro)
            if imputer.verbose:
                print(f"AVISO: Bootstrap não completou ({after_bootstrap} missings restantes)")

            end_time = time.time()
            imputer.execution_stats = {
                'initial_missing': initial_missing,
                'final_missing': after_bootstrap,
                'execution_time': end_time - start_time,
                'strategy': 'Simple Bootstrap (incompleto)',
                'cycles': 0
            }

            return result
