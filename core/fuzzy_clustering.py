"""
Fuzzy C-Means Clustering com Partial Distance Strategy (PDS)

Baseado em: "Kernel Fuzzy C-means clustering with local adaptive distances"
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259266

PDS: Calcula distâncias parciais usando apenas features disponíveis,
re-escalando pelo recíproco da proporção de valores observados.
"""
import numpy as np
from numba import jit, prange
import warnings


@jit(nopython=True, cache=True)
def _partial_distance_squared(x, centroid):
    """
    Calcula distância quadrada parcial entre x e centroid,
    usando apenas features onde x não é NaN.
    Re-escala pelo factor d_total / n_observed.

    Returns:
        dist_sq: Distância quadrada re-escalada
        n_observed: Número de features observadas
    """
    d_total = len(x)
    dist_sq = 0.0
    n_observed = 0

    for j in range(d_total):
        if not np.isnan(x[j]):
            diff = x[j] - centroid[j]
            dist_sq += diff * diff
            n_observed += 1

    if n_observed == 0:
        return np.inf, 0

    # Re-escala PDS: multiplica por d_total / n_observed
    dist_sq_scaled = dist_sq * (d_total / n_observed)

    return dist_sq_scaled, n_observed


@jit(nopython=True, cache=True)
def _partial_distance_weighted_squared(x, centroid, weights):
    """
    Distância quadrada parcial ponderada por MI weights.
    """
    d_total = len(x)
    dist_sq = 0.0
    weight_sum = 0.0

    for j in range(d_total):
        if not np.isnan(x[j]):
            diff = x[j] - centroid[j]
            dist_sq += weights[j] * diff * diff
            weight_sum += weights[j]

    if weight_sum == 0:
        return np.inf

    # Normaliza pelos pesos observados
    return dist_sq / weight_sum


@jit(nopython=True, parallel=True, cache=True)
def _compute_memberships_pds(X, centroids, m=2.0):
    """
    Calcula fuzzy memberships u_ij para todos os pontos.

    Args:
        X: Matriz (n x d) de dados (pode ter NaN)
        centroids: Matriz (c x d) de centroids
        m: Fuzzifier (tipicamente 2.0)

    Returns:
        U: Matriz (n x c) de memberships
    """
    n = X.shape[0]
    c = centroids.shape[0]
    U = np.zeros((n, c))
    exponent = 2.0 / (m - 1.0)

    for i in prange(n):
        # Calcular distâncias a todos os centroids
        distances = np.zeros(c)
        for j in range(c):
            dist_sq, n_obs = _partial_distance_squared(X[i], centroids[j])
            distances[j] = np.sqrt(dist_sq) if dist_sq < np.inf else np.inf

        # Verificar se alguma distância é zero (ponto coincide com centroid)
        min_dist = np.min(distances)
        if min_dist < 1e-10:
            # Membership = 1 para o cluster mais próximo
            for j in range(c):
                U[i, j] = 1.0 if distances[j] < 1e-10 else 0.0
            # Normalizar se múltiplos zeros
            total = np.sum(U[i])
            if total > 0:
                for j in range(c):
                    U[i, j] /= total
        else:
            # Fórmula FCM padrão
            for j in range(c):
                sum_ratio = 0.0
                for k in range(c):
                    if distances[k] > 0:
                        ratio = distances[j] / distances[k]
                        sum_ratio += ratio ** exponent
                U[i, j] = 1.0 / sum_ratio if sum_ratio > 0 else 0.0

    return U


@jit(nopython=True, cache=True)
def _update_centroids_pds(X, U, m=2.0):
    """
    Actualiza centroids usando PDS (ignora NaN).

    Para cada feature j, o centroid é calculado apenas com
    os pontos que têm essa feature observada.
    """
    n, d = X.shape
    c = U.shape[1]
    centroids = np.zeros((c, d))

    for j in range(c):
        for k in range(d):
            numerator = 0.0
            denominator = 0.0
            for i in range(n):
                if not np.isnan(X[i, k]):
                    u_power = U[i, j] ** m
                    numerator += u_power * X[i, k]
                    denominator += u_power

            if denominator > 0:
                centroids[j, k] = numerator / denominator
            else:
                # Fallback: média global da feature
                mean_val = 0.0
                count = 0
                for i in range(n):
                    if not np.isnan(X[i, k]):
                        mean_val += X[i, k]
                        count += 1
                centroids[j, k] = mean_val / count if count > 0 else 0.0

    return centroids


@jit(nopython=True, cache=True)
def _compute_objective_pds(X, centroids, U, m=2.0):
    """
    Calcula função objectivo J do FCM com PDS.
    """
    n = X.shape[0]
    c = centroids.shape[0]
    J = 0.0

    for i in range(n):
        for j in range(c):
            dist_sq, _ = _partial_distance_squared(X[i], centroids[j])
            if dist_sq < np.inf:
                J += (U[i, j] ** m) * dist_sq

    return J


def fcm_pds(X, n_clusters, m=2.0, max_iter=100, tol=1e-4,
            random_state=None, verbose=False):
    """
    Fuzzy C-Means com Partial Distance Strategy.

    Args:
        X: numpy array (n x d), pode conter NaN
        n_clusters: número de clusters c
        m: fuzzifier (default 2.0)
        max_iter: máximo de iterações
        tol: tolerância para convergência
        random_state: seed para reproducibilidade
        verbose: mostrar progresso

    Returns:
        centroids: (c x d) centroids dos clusters
        U: (n x c) matriz de memberships
        n_iter: número de iterações até convergência
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    if random_state is not None:
        np.random.seed(random_state)

    # Inicializar centroids usando k-means++ style (com PDS)
    centroids = _initialize_centroids_pds(X, n_clusters)

    # Iterações FCM
    prev_J = np.inf
    for iteration in range(max_iter):
        # E-step: calcular memberships
        U = _compute_memberships_pds(X, centroids, m)

        # M-step: actualizar centroids
        centroids = _update_centroids_pds(X, U, m)

        # Calcular função objectivo
        J = _compute_objective_pds(X, centroids, U, m)

        if verbose:
            print(f"  FCM iter {iteration+1}: J = {J:.4f}")

        # Verificar convergência
        if abs(prev_J - J) < tol:
            if verbose:
                print(f"  FCM converged at iteration {iteration+1}")
            break

        prev_J = J

    return centroids, U, iteration + 1


def _initialize_centroids_pds(X, n_clusters):
    """
    Inicializa centroids usando método similar a k-means++,
    adaptado para dados com NaN.
    """
    n, d = X.shape
    centroids = np.zeros((n_clusters, d))

    # Primeiro centroid: ponto aleatório (preferir completo se possível)
    complete_mask = ~np.isnan(X).any(axis=1)
    if complete_mask.sum() > 0:
        complete_indices = np.where(complete_mask)[0]
        first_idx = np.random.choice(complete_indices)
    else:
        first_idx = np.random.randint(n)

    centroids[0] = np.nan_to_num(X[first_idx], nan=0.0)

    # Restantes centroids: probabilístico baseado em distância
    for k in range(1, n_clusters):
        distances = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(k):
                dist_sq, n_obs = _partial_distance_squared(X[i], centroids[j])
                if n_obs > 0 and dist_sq < min_dist:
                    min_dist = dist_sq
            distances[i] = min_dist if min_dist < np.inf else 0.0

        # Normalizar para probabilidades
        prob = distances / distances.sum() if distances.sum() > 0 else np.ones(n) / n

        # Escolher próximo centroid
        next_idx = np.random.choice(n, p=prob)
        centroids[k] = np.nan_to_num(X[next_idx], nan=0.0)

    return centroids


def compute_memberships_for_point(point, centroids, m=2.0):
    """
    Calcula memberships para um único ponto (útil para pontos novos).

    Args:
        point: array (d,) pode ter NaN
        centroids: array (c, d)
        m: fuzzifier

    Returns:
        memberships: array (c,)
    """
    point = np.asarray(point, dtype=np.float64).reshape(1, -1)
    U = _compute_memberships_pds(point, centroids, m)
    return U[0]


def get_top_clusters(memberships, n_top=3, threshold=0.1):
    """
    Retorna índices dos clusters mais relevantes para um ponto.
    Usa apenas os top-n clusters com maior membership.

    Args:
        memberships: array (c,) de memberships
        n_top: número de clusters a retornar (sempre retorna n_top)
        threshold: não usado (mantido para compatibilidade)

    Returns:
        cluster_indices: array de índices dos n_top clusters mais relevantes
    """
    # Sempre retorna os top-n clusters com maior membership
    return np.argsort(memberships)[-n_top:][::-1]


def estimate_local_density(memberships, cluster_sizes):
    """
    Estima densidade local baseada nos memberships.

    Densidade = soma(membership * tamanho_cluster) / tamanho_total

    Args:
        memberships: array (c,) de memberships do ponto
        cluster_sizes: array (c,) com número de pontos em cada cluster

    Returns:
        density: estimativa de densidade local
    """
    total_size = cluster_sizes.sum()
    if total_size == 0:
        return 1.0

    weighted_size = np.sum(memberships * cluster_sizes)
    density = weighted_size / total_size

    return density


class FuzzyClusterIndex:
    """
    Índice de clusters fuzzy para acelerar busca de vizinhos.

    Usa hard assignments (cada ponto pertence ao cluster com maior membership)
    mas permite busca em múltiplos clusters.
    """

    def __init__(self, n_clusters=10, m=2.0, membership_threshold=0.1,
                 max_iter=100, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.m = m
        self.membership_threshold = membership_threshold  # Não usado, mantido para compatibilidade
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self.centroids = None
        self.memberships = None  # (n x c)
        self.hard_assignments = None  # (n,) - cluster dominante de cada ponto
        self.cluster_members = None  # Lista de arrays com índices por cluster
        self.cluster_sizes = None
        self.is_fitted = False

    def fit(self, X):
        """
        Ajusta o índice FCM aos dados.

        Args:
            X: array (n x d), pode conter NaN
        """
        X = np.asarray(X, dtype=np.float64)

        if self.verbose:
            print(f"  Fitting FCM-PDS with {self.n_clusters} clusters...")

        # Executar FCM-PDS
        self.centroids, self.memberships, n_iter = fcm_pds(
            X, self.n_clusters, self.m, self.max_iter,
            random_state=self.random_state, verbose=self.verbose
        )

        # Hard assignments: cada ponto pertence ao cluster com maior membership
        self.hard_assignments = np.argmax(self.memberships, axis=1)

        # Construir índice de membros por cluster (hard)
        self._build_cluster_index()

        self.is_fitted = True

        if self.verbose:
            print(f"  FCM-PDS fitted in {n_iter} iterations")
            print(f"  Cluster sizes: {self.cluster_sizes}")

        return self

    def _build_cluster_index(self):
        """Constrói índice de pontos por cluster usando hard assignments."""
        n = len(self.hard_assignments)
        c = self.n_clusters

        self.cluster_members = []
        self.cluster_sizes = np.zeros(c, dtype=np.int64)

        for j in range(c):
            # Pontos cujo cluster dominante é j
            members = np.where(self.hard_assignments == j)[0]
            self.cluster_members.append(members)
            self.cluster_sizes[j] = len(members)

    def get_candidate_donors(self, query_memberships, n_top_clusters=3):
        """
        Retorna índices dos donors candidatos baseado nos clusters relevantes.

        Args:
            query_memberships: array (c,) de memberships do ponto query
            n_top_clusters: número de clusters a considerar

        Returns:
            candidate_indices: array de índices dos donors candidatos
        """
        if not self.is_fitted:
            raise ValueError("FuzzyClusterIndex not fitted. Call fit() first.")

        # Obter top clusters para a query
        top_clusters = get_top_clusters(query_memberships, n_top_clusters)

        # Juntar membros de todos os clusters relevantes
        candidates = []
        for cluster_idx in top_clusters:
            candidates.extend(self.cluster_members[cluster_idx])

        return np.array(candidates, dtype=np.int64)

    def get_memberships_for_point(self, point):
        """Calcula memberships para um ponto novo."""
        if not self.is_fitted:
            raise ValueError("FuzzyClusterIndex not fitted. Call fit() first.")

        return compute_memberships_for_point(point, self.centroids, self.m)

    def estimate_density(self, memberships):
        """Estima densidade local para um ponto."""
        return estimate_local_density(memberships, self.cluster_sizes)


def select_n_clusters(X, max_clusters=15, min_clusters=3, method='silhouette'):
    """
    Selecciona número óptimo de clusters usando silhouette ou elbow.

    Args:
        X: dados (pode ter NaN)
        max_clusters: máximo de clusters a testar
        min_clusters: mínimo de clusters
        method: 'silhouette' ou 'elbow'

    Returns:
        optimal_c: número óptimo de clusters
    """
    n = X.shape[0]
    max_c = min(max_clusters, n // 5)  # Pelo menos 5 pontos por cluster

    if max_c < min_clusters:
        return min_clusters

    scores = []
    for c in range(min_clusters, max_c + 1):
        centroids, U, _ = fcm_pds(X, c, max_iter=50, random_state=42)
        J = _compute_objective_pds(X, centroids, U)
        scores.append((c, J))

    if method == 'elbow':
        # Encontrar "cotovelo" - maior redução relativa de J
        best_c = min_clusters
        max_improvement = 0
        for i in range(1, len(scores)):
            improvement = (scores[i-1][1] - scores[i][1]) / scores[i-1][1]
            if improvement > max_improvement:
                max_improvement = improvement
                best_c = scores[i][0]
        return best_c
    else:
        # Silhouette simplificado (baseado em J normalizado)
        # Escolher c que minimiza J/c
        normalized = [(c, J/c) for c, J in scores]
        best_c = min(normalized, key=lambda x: x[1])[0]
        return best_c
