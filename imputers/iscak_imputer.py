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
import warnings
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
               column_types_config: str = None,
               delete_allblank: bool = True) -> pd.DataFrame:
        """
        Imputa valores em falta no dataset.

        Args:
            data: DataFrame com valores em falta
            force_categorical: Lista de colunas a forçar como categóricas
            force_ordinal: Dict de colunas ordinais com ordem dos valores
            interactive: Se True, pergunta ao user sobre colunas ambíguas
            column_types_config: Caminho para ficheiro JSON com configuração de tipos
            delete_allblank: Se True, remove linhas 100% vazias antes da imputação.
                            Se False, preenche-as com mediana/moda (fallback).

        Returns:
            DataFrame com valores imputados
        """
        start_time = time.time()
        original_data = data.copy()

        # === DETECTAR E TRATAR LINHAS 100% VAZIAS ===
        allblank_mask = original_data.isna().all(axis=1)
        n_allblank = allblank_mask.sum()
        self._removed_allblank_indices = []  # Guardar para referência

        if n_allblank > 0:
            allblank_indices = original_data.index[allblank_mask].tolist()

            if delete_allblank:
                # Remover linhas e avisar
                self._removed_allblank_indices = allblank_indices
                warnings.warn(
                    f"\n⚠️  {n_allblank} linha(s) removida(s) por estarem 100% vazias.\n"
                    f"    Índices: {allblank_indices}\n"
                    f"    Estas linhas não podem ser imputadas por métodos baseados em distância.",
                    UserWarning
                )
                original_data = original_data[~allblank_mask].copy()

                if self.verbose:
                    print(f"\n⚠️  {n_allblank} linha(s) 100% vazia(s) removidas: {allblank_indices}")
            else:
                # Apenas avisar que vai usar fallback
                warnings.warn(
                    f"\n⚠️  {n_allblank} linha(s) estão 100% vazias.\n"
                    f"    Índices: {allblank_indices}\n"
                    f"    Serão preenchidas com mediana/moda (fallback).",
                    UserWarning
                )
                if self.verbose:
                    print(f"\n⚠️  {n_allblank} linha(s) 100% vazia(s) detectadas: {allblank_indices}")
                    print(f"    Serão preenchidas com mediana/moda (fallback)")

        if column_types_config and Path(column_types_config).exists():
            force_categorical, force_ordinal = MixedDataHandler.load_config(column_types_config)
        data_encoded, self.encoding_info = self.mixed_handler.fit_transform(
            original_data,
            force_categorical=force_categorical,
            force_ordinal=force_ordinal,
            interactive=interactive,
            verbose=self.verbose
        )

        # PDS é controlado pelo parâmetro use_pds
        # O overlap é adaptativo por valor (maximiza overlap, reduz se necessário)
        self._effective_pds = self.use_pds

        missing_mask = data_encoded.isna()
        initial_missing = missing_mask.sum().sum()
        if self.verbose:
            self._print_header(data_encoded)
        complete_rows = (~missing_mask).all(axis=1).sum()
        n_rows = len(data_encoded)
        pct_complete_rows = complete_rows / n_rows * 100
        if self.verbose:
            print(f"\nLinhas 100% completas: {complete_rows}/{n_rows} ({pct_complete_rows:.1f}%)")

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
            if self._effective_pds:
                print(f"Estratégia: ISCA-k+PDS primeiro, fallback se necessário")
            else:
                print(f"Estratégia: ISCA-k clássico")

        # Executar estratégia ISCA-k (com modo iterativo para residuais)
        strategy = ISCAkStrategy()
        result = strategy.run(self, data_encoded, missing_mask, initial_missing, start_time)

        return result

    def _apply_imr_fallback(self, result, original_data, start_time, initial_missing, phase1_stats=None):
        """Aplica IMR como fallback e refina com ISCA-k."""
        import time
        from imputers.imr_imputer import IMRInitializer

        phases = [phase1_stats] if phase1_stats else []
        before_imr = result.isna().sum().sum()

        if self.verbose:
            print(f"\n{'='*70}")
            print("FASE 2: Fallback IMR")
            print(f"{'='*70}")
            print(f"  Método: IMR + Refinamento ISCA-k")

        imr = IMRInitializer(n_iterations=5)
        non_numeric = (self.mixed_handler.binary_cols +
                      self.mixed_handler.nominal_cols +
                      self.mixed_handler.ordinal_cols)
        result = imr.fit_transform(result, self.mixed_handler.numeric_cols, non_numeric)

        after_imr = result.isna().sum().sum()
        n_refined = 0

        if after_imr == 0:
            # Refinar com ISCA-k
            scaled_result = self._get_scaled_data(result, force_refit=True)
            columns_ordered = self._rank_columns(original_data)
            residual_mask = original_data.isna() & ~result.isna()

            for col in columns_ordered:
                if residual_mask[col].any():
                    refined = self._refine_column_mixed(original_data, col, scaled_result, residual_mask[col])
                    result.loc[residual_mask[col], col] = refined[residual_mask[col]]
                    n_refined += residual_mask[col].sum()

        final_missing = result.isna().sum().sum()

        if self.verbose:
            print(f"  IMR: {before_imr} → {after_imr} ({before_imr - after_imr} preenchidos)")
            if n_refined > 0:
                print(f"  Refinamento: {n_refined} valores refinados")

        phases.append({'name': 'IMR+Refinamento', 'before': before_imr, 'after': final_missing})

        end_time = time.time()
        self.execution_stats = {
            'initial_missing': initial_missing,
            'final_missing': final_missing,
            'execution_time': end_time - start_time,
            'strategy': 'ISCA-k+PDS → IMR+Refinamento',
            'phases': phases
        }
        if self.verbose:
            self._print_summary()

        return result

    def _apply_bootstrap_fallback(self, result, original_data, start_time, initial_missing, phase1_stats=None):
        """Aplica mediana/moda como fallback e refina com ISCA-k."""
        import time

        phases = [phase1_stats] if phase1_stats else []
        before_bootstrap = result.isna().sum().sum()

        if self.verbose:
            print(f"\n{'='*70}")
            print("FASE 2: Fallback Bootstrap")
            print(f"{'='*70}")
            print(f"  Método: Mediana/Moda + Refinamento ISCA-k")

        result = self._simple_bootstrap(result)

        after_bootstrap = result.isna().sum().sum()
        n_refined = 0

        if after_bootstrap == 0:
            # Refinar com ISCA-k
            scaled_result = self._get_scaled_data(result, force_refit=True)
            columns_ordered = self._rank_columns(original_data)
            residual_mask = original_data.isna() & ~result.isna()

            for col in columns_ordered:
                if residual_mask[col].any():
                    refined = self._refine_column_mixed(original_data, col, scaled_result, residual_mask[col])
                    result.loc[residual_mask[col], col] = refined[residual_mask[col]]
                    n_refined += residual_mask[col].sum()

        final_missing = result.isna().sum().sum()

        if self.verbose:
            print(f"  Bootstrap: {before_bootstrap} → {after_bootstrap} ({before_bootstrap - after_bootstrap} preenchidos)")
            if n_refined > 0:
                print(f"  Refinamento: {n_refined} valores refinados")

        phases.append({'name': 'Bootstrap+Refinamento', 'before': before_bootstrap, 'after': final_missing})

        end_time = time.time()
        self.execution_stats = {
            'initial_missing': initial_missing,
            'final_missing': final_missing,
            'execution_time': end_time - start_time,
            'strategy': 'ISCA-k+PDS → Bootstrap+Refinamento',
            'phases': phases
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

    def _handle_residuals_iterative(self, result, remaining_missing, initial_missing,
                                      columns_ordered, original_data, start_time, prev_stats,
                                      phase1_stats=None):
        """
        Modo iterativo para tratar missings residuais.

        Cria um subdataset que cresce progressivamente:
        1. Começa com linhas completas
        2. Adiciona linhas com poucos missings, imputa-as
        3. Essas linhas ficam completas e são doadores para as próximas
        4. Repete até processar todas as linhas
        """
        phases = [phase1_stats] if phase1_stats else []
        # Fase 2 usa sempre ISCA-k clássico (sem PDS) para lidar com linhas com muitos missings
        phase_name = "ISCA-k clássico"

        if self.verbose:
            print(f"\n{'='*70}")
            print("FASE 2: Modo Iterativo")
            print(f"{'='*70}")
            print(f"  Método: {phase_name} (sem PDS, min 1 feature)")

        before_phase2 = remaining_missing
        cycle = 0
        stagnant_cycles = 0
        max_stagnant = 3

        # Contar missings por linha ATUAL
        missings_per_row = result.isna().sum(axis=1)

        # Índices ordenados por número de missings (menos primeiro)
        sorted_indices = missings_per_row.sort_values().index.tolist()

        # Subdataset começa com linhas completas
        subdataset_idx = [idx for idx in sorted_indices if missings_per_row[idx] == 0]
        pending_idx = [idx for idx in sorted_indices if missings_per_row[idx] > 0]

        if self.verbose:
            print(f"  Linhas completas: {len(subdataset_idx)}")
            print(f"  Linhas pendentes: {len(pending_idx)}")

        # Se não há linhas completas, usar as com menos missings como base
        if len(subdataset_idx) < self.min_friends and pending_idx:
            # Mover linhas com menos missings para o subdataset
            needed = self.min_friends - len(subdataset_idx)
            to_move = pending_idx[:needed]
            subdataset_idx.extend(to_move)
            pending_idx = pending_idx[needed:]
            if self.verbose:
                print(f"  Expandido subdataset para {len(subdataset_idx)} linhas")

        # Processar linhas pendentes em lotes
        batch_size = max(1, len(pending_idx) // 10)  # ~10 iterações
        batch_size = min(batch_size, 50)  # Não mais que 50 por vez

        while pending_idx and cycle < self.max_cycles * 3 and stagnant_cycles < max_stagnant:
            cycle += 1
            before_cycle = result.isna().sum().sum()

            # Pegar próximo lote de linhas a processar
            current_batch = pending_idx[:batch_size]
            pending_idx = pending_idx[batch_size:]

            if self.verbose:
                print(f"\n  Ciclo {cycle}: processando {len(current_batch)} linhas")
                print(f"    Subdataset: {len(subdataset_idx)} doadores")

            # Criar subdataset com doadores actuais
            subdataset = result.loc[subdataset_idx].copy()

            if len(subdataset) < self.min_friends:
                if self.verbose:
                    print(f"    Poucos doadores, adicionando batch ao subdataset...")
                subdataset_idx.extend(current_batch)
                continue

            # Escalar subdataset
            scaled_sub = self._get_scaled_data(subdataset, force_refit=True)

            # Imputar cada linha do batch
            n_imputed = 0
            for idx in current_batch:
                # Imputar valores missing desta linha usando o subdataset
                for col in columns_ordered:
                    if pd.isna(result.loc[idx, col]):
                        imputed = self._impute_single_value_from_subdataset(
                            result.loc[idx], col, subdataset, scaled_sub
                        )
                        if imputed is not None:
                            result.loc[idx, col] = imputed
                            n_imputed += 1

                # Adicionar ao subdataset (mesmo incompleta, pode ajudar)
                subdataset_idx.append(idx)

            new_remaining = result.isna().sum().sum()
            progress = before_cycle - new_remaining

            if self.verbose:
                print(f"    Imputados: {n_imputed}, Restantes: {new_remaining}")

            if progress == 0:
                stagnant_cycles += 1
            else:
                stagnant_cycles = 0

            remaining_missing = new_remaining

        # Se ainda há missings, fallback com mediana/moda
        if remaining_missing > 0:
            if self.verbose:
                print(f"\n  Fallback: preenchendo {remaining_missing} missings com mediana/moda")
            result = self._simple_bootstrap(result)
            remaining_missing = result.isna().sum().sum()

        phase2_imputed = before_phase2 - remaining_missing
        phases.append({
            'name': f'Iterativo ({cycle} ciclos)',
            'before': before_phase2,
            'after': remaining_missing,
            'cycles': cycle,
            'imputed': phase2_imputed
        })

        end_time = time.time()
        self.execution_stats = {
            'initial_missing': initial_missing,
            'final_missing': remaining_missing,
            'execution_time': end_time - start_time,
            'strategy': f"{phase_name} → Iterativo",
            'phases': phases,
            'phase2_activated': True,
            'phase2_cycles': cycle,
            'phase2_imputed': phase2_imputed
        }

        if self.verbose:
            self._print_summary()

        if remaining_missing > 0:
            import warnings
            warnings.warn(f"Imputação incompleta: {remaining_missing} valores não foram imputados")

        return result

    def _impute_single_value_from_subdataset(self, row_data, target_col, subdataset, scaled_sub):
        """
        Imputa um único valor usando o subdataset como doadores.

        NOTA: Usa sempre modo clássico (sem PDS) porque na Fase 2 as linhas
        residuais têm muitos missings e min_overlap do PDS seria muito restritivo.

        Args:
            row_data: Series com os dados da linha a imputar
            target_col: Coluna a imputar
            subdataset: DataFrame com linhas doadores
            scaled_sub: DataFrame com dados escalados dos doadores

        Returns:
            Valor imputado ou None se não conseguir
        """
        # Doadores válidos: têm o target preenchido
        valid_donors = subdataset[~subdataset[target_col].isna()]
        if len(valid_donors) < self.min_friends:
            return None

        feature_cols = [c for c in subdataset.columns if c != target_col]

        # Features disponíveis no sample (não-missing)
        sample_features = row_data[feature_cols].values.astype(np.float64)
        avail_mask = ~np.isnan(sample_features)
        n_avail = avail_mask.sum()

        # Precisa de pelo menos 1 feature para calcular distância
        # (consistente com _impute_column_mixed que permite n_avail >= 1)
        if n_avail < 1:
            return None

        # Pesos MI
        mi_scores = self.mi_matrix.loc[feature_cols, target_col]
        weights = mi_scores.values.astype(np.float64)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

        # Dados dos doadores (escalonados para distâncias)
        donor_scaled = scaled_sub.loc[valid_donors.index, feature_cols].values.astype(np.float64)
        donor_targets = valid_donors[target_col].values

        # Modo clássico: usar apenas features onde o sample tem valores
        # Os doadores do subdataset são linhas completas, então têm todas as features
        avail_indices = np.where(avail_mask)[0]

        # Escalonar o sample também
        sample_scaled = scaled_sub.loc[row_data.name, feature_cols].values.astype(np.float64) if row_data.name in scaled_sub.index else sample_features
        sample_sub = sample_scaled[avail_indices]
        donors_sub = donor_scaled[:, avail_indices]

        # Filtrar doadores com todas essas features (devem ser todos se subdataset é completo)
        valid_donors_mask = ~np.isnan(donors_sub).any(axis=1)
        if valid_donors_mask.sum() < self.min_friends:
            return None

        donors_valid = donors_sub[valid_donors_mask]
        targets_valid = donor_targets[valid_donors_mask]

        weights_sub = weights[avail_indices]
        weights_sub = weights_sub / weights_sub.sum() if weights_sub.sum() > 0 else np.ones_like(weights_sub) / len(weights_sub)

        distances_valid = weighted_euclidean_batch(sample_sub, donors_valid, weights_sub)

        # Selecionar k vizinhos
        k = adaptive_k_hybrid(
            distances_valid, targets_valid,
            self.min_friends, self.max_friends, self.adaptive_k_alpha
        )
        k = min(k, len(distances_valid))

        nearest_idx = np.argsort(distances_valid)[:k]
        nearest_dist = distances_valid[nearest_idx]
        nearest_vals = targets_valid[nearest_idx]

        # Calcular valor
        is_categorical = (target_col in self.mixed_handler.nominal_cols or
                         target_col in self.mixed_handler.binary_cols)

        if is_categorical:
            # Votos ponderados com tie-breaker por distância média
            weighted_votes = {}
            total_dist_per_class = {}
            count_per_class = {}
            for val, dist in zip(nearest_vals, nearest_dist):
                weight = 1 / (dist + 1e-6)
                weighted_votes[val] = weighted_votes.get(val, 0) + weight
                total_dist_per_class[val] = total_dist_per_class.get(val, 0) + dist
                count_per_class[val] = count_per_class.get(val, 0) + 1
            sorted_classes = sorted(
                weighted_votes.keys(),
                key=lambda v: (-weighted_votes[v], total_dist_per_class[v] / count_per_class[v])
            )
            return sorted_classes[0]
        else:
            if nearest_dist.sum() > 0:
                inv_dist = 1.0 / (nearest_dist + 1e-10)
                w = inv_dist / inv_dist.sum()
                return np.average(nearest_vals, weights=w)
            return np.mean(nearest_vals)

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

        Lógica:
        - Se use_pds=True: tenta com overlap máximo, reduz até encontrar doadores
        - Quando overlap=1 ou use_pds=False: usa modo clássico (sem escala PDS)
        - Modo clássico funciona com qualquer nº de features disponíveis (>=1)
        """
        result = data[target_col].copy()
        missing_mask = data[target_col].isna()
        complete_mask = ~missing_mask
        if complete_mask.sum() == 0:
            return result

        feature_cols = [c for c in data.columns if c != target_col]
        n_features = len(feature_cols)
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

        X_all_scaled = scaled_data[feature_cols].values.astype(np.float64)
        X_all_original = data[feature_cols].values.astype(np.float64)

        is_target_categorical = (target_col in self.mixed_handler.binary_cols or
                                 target_col in self.mixed_handler.nominal_cols or
                                 target_col in self.mixed_handler.ordinal_cols)

        missing_row_indices = np.where(missing_mask.values)[0]
        if len(missing_row_indices) == 0:
            return result

        result_values = result.values.copy()
        range_factors = range_factors_full[[data.columns.get_loc(c) for c in feature_cols]].astype(np.float64)

        for row_idx in missing_row_indices:
            sample_scaled = X_all_scaled[row_idx]
            sample_original = X_all_original[row_idx]

            # Features disponíveis nesta linha
            avail_mask = ~np.isnan(sample_scaled)
            n_avail = avail_mask.sum()

            if n_avail == 0:
                # Sem features disponíveis, não consegue imputar
                continue

            imputed_value = None

            # === MODO PDS ===
            # min_overlap = 50% das features para evitar escala PDS agressiva
            # Se não encontrar doadores, cai para modo clássico
            min_pds_overlap = max(2, n_features // 2)

            if self._effective_pds and n_avail >= min_pds_overlap:
                if not has_categorical:
                    distances, n_shared = weighted_euclidean_pds(
                        sample_scaled, X_ref_scaled, weights, min_pds_overlap
                    )
                else:
                    distances, n_shared = mixed_distance_pds(
                        sample_scaled, X_ref_scaled,
                        sample_original, X_ref_original,
                        numeric_mask, binary_mask,
                        ordinal_mask, nominal_mask,
                        weights, range_factors, min_pds_overlap
                    )

                valid_mask = np.isfinite(distances)
                if valid_mask.sum() >= self.min_friends:
                    distances_valid = distances[valid_mask]
                    y_ref_valid = y_ref[valid_mask]
                    imputed_value = self._compute_imputed_value(
                        distances_valid, y_ref_valid, is_target_categorical
                    )
                # Se não encontrar doadores suficientes, cai para modo clássico abaixo

            # === MODO CLÁSSICO (sem PDS ou overlap=1) ===
            if imputed_value is None and n_avail >= 1:
                avail_indices = np.where(avail_mask)[0]
                sample_scaled_sub = sample_scaled[avail_indices]
                sample_original_sub = sample_original[avail_indices]
                X_ref_scaled_sub = X_ref_scaled[:, avail_indices]
                X_ref_original_sub = X_ref_original[:, avail_indices]

                # Doadores que têm todas as features disponíveis
                valid_donors_mask = ~np.isnan(X_ref_scaled_sub).any(axis=1)
                if valid_donors_mask.sum() >= self.min_friends:
                    X_ref_valid = X_ref_scaled_sub[valid_donors_mask]
                    X_ref_orig_valid = X_ref_original_sub[valid_donors_mask]
                    y_ref_valid = y_ref[valid_donors_mask]

                    weights_sub = weights[avail_indices].copy()
                    if weights_sub.sum() > 0:
                        weights_sub = weights_sub / weights_sub.sum()
                    else:
                        weights_sub = np.ones_like(weights_sub) / len(weights_sub)

                    # Calcular distâncias sem escala PDS
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

                    imputed_value = self._compute_imputed_value(
                        distances_valid, y_ref_valid, is_target_categorical
                    )

            if imputed_value is not None:
                result_values[row_idx] = imputed_value

        return pd.Series(result_values, index=result.index, name=result.name)

    def _compute_imputed_value(self, distances, y_values, is_categorical):
        """Calcula o valor imputado baseado nos vizinhos mais próximos."""
        k = adaptive_k_hybrid(
            distances, y_values,
            min_k=self.min_friends, max_k=self.max_friends,
            alpha=self.adaptive_k_alpha, is_categorical=is_categorical
        )
        k = min(k, len(distances))
        if k == 0:
            return None

        friend_idx = np.argpartition(distances, k-1)[:k] if k < len(distances) else np.arange(len(distances))
        friend_values = y_values[friend_idx]
        friend_distances = distances[friend_idx]

        if is_categorical:
            if len(friend_values) == 1:
                return friend_values[0]

            # Contagem de votos ponderados e distâncias médias por classe
            weighted_votes = {}
            total_dist_per_class = {}
            count_per_class = {}

            for val, dist in zip(friend_values, friend_distances):
                weight = 1 / (dist + 1e-6)
                weighted_votes[val] = weighted_votes.get(val, 0) + weight
                total_dist_per_class[val] = total_dist_per_class.get(val, 0) + dist
                count_per_class[val] = count_per_class.get(val, 0) + 1

            # Ordenar por voto (desc) e depois por distância média (asc) para desempate
            sorted_classes = sorted(
                weighted_votes.keys(),
                key=lambda v: (-weighted_votes[v], total_dist_per_class[v] / count_per_class[v])
            )
            return sorted_classes[0]
        else:
            if np.any(friend_distances < 1e-10):
                exact_mask = friend_distances < 1e-10
                return np.mean(friend_values[exact_mask])
            else:
                w = 1 / (friend_distances + 1e-6)
                w = w / w.sum()
                return np.average(friend_values, weights=w)

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
                    # Votos ponderados com tie-breaker por distância média
                    weighted_votes = {}
                    total_dist_per_class = {}
                    count_per_class = {}
                    for val, dist in zip(friend_values, friend_distances):
                        weight = 1 / (dist + 1e-6)
                        weighted_votes[val] = weighted_votes.get(val, 0) + weight
                        total_dist_per_class[val] = total_dist_per_class.get(val, 0) + dist
                        count_per_class[val] = count_per_class.get(val, 0) + 1
                    sorted_classes = sorted(
                        weighted_votes.keys(),
                        key=lambda v: (-weighted_votes[v], total_dist_per_class[v] / count_per_class[v])
                    )
                    refined_values[row_idx] = sorted_classes[0]
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
        print(f"PDS (partial donors): {self._effective_pds}")
        if self._effective_pds:
            print(f"  Overlap: adaptativo (maximiza por valor)")
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
        print("RESULTADO FINAL")
        print("="*70)

        # Mostrar linhas removidas (se houver)
        if hasattr(self, '_removed_allblank_indices') and self._removed_allblank_indices:
            n_removed = len(self._removed_allblank_indices)
            print(f"\n⚠️  Linhas 100% vazias removidas: {n_removed}")
            print(f"    Índices: {self._removed_allblank_indices}")

        # Mostrar resumo de cada fase
        phases = stats.get('phases', [])
        if phases:
            print("\nFases:")
            for phase in phases:
                before = phase['before']
                after = phase['after']
                imputados = before - after
                pct = (imputados / before * 100) if before > 0 else 0
                cycles_info = f" ({phase['cycles']} ciclos)" if 'cycles' in phase else ""
                print(f"  {phase['name']}: {before} → {after} ({imputados} imputados, {pct:.1f}%){cycles_info}")

        # Info sobre Fase 2
        if stats.get('phase2_activated', False):
            print(f"\n📊 Fase 2 activada: {stats['phase2_cycles']} ciclos, {stats['phase2_imputed']} valores imputados")
        else:
            print(f"\n✅ Fase 1 resolveu tudo (Fase 2 não necessária)")

        # Resumo geral
        print(f"\nTotal: {stats['initial_missing']} → {stats['final_missing']} missings")

        if stats['final_missing'] == 0:
            print("Status: SUCESSO - Dataset 100% completo")
        else:
            print(f"Status: INCOMPLETO - {stats['final_missing']} missings restantes")

        if stats['final_missing'] < stats['initial_missing']:
            taxa = (1 - stats['final_missing']/stats['initial_missing'])*100
            print(f"Taxa de imputação: {taxa:.1f}%")

        print(f"Tempo total: {stats['execution_time']:.2f}s")
        print("="*70 + "\n")
