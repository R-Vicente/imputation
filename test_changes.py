"""
Script de teste para validar as alterações:
1. Parâmetro delete_allblank no ISCAkCore
2. Nova função introduce_mnar (estatisticamente correcta)
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.datasets import load_iris

# Importar o imputer
from iscak_core import ISCAkCore


def introduce_mnar(data, missing_rate, seed, na_attributes=None):
    """
    Missing Not At Random - implementação estatisticamente correcta.
    Os missings dependem do PRÓPRIO VALOR que vai faltar.
    Valores mais ALTOS têm maior probabilidade de ser missing.
    """
    np.random.seed(seed)
    data_missing = data.copy()

    N = len(data)
    M = len(data.columns)
    V = N * M

    if na_attributes is None:
        NA = max(1, M // 2)
    else:
        NA = min(na_attributes, M)

    target_missing = int(missing_rate * V)

    all_cols = list(range(M))
    np.random.shuffle(all_cols)
    MDAttributes = all_cols[:NA]

    missing_per_attr = target_missing // NA
    extra = target_missing % NA

    for i, col_idx in enumerate(MDAttributes):
        n_missing = missing_per_attr + (1 if i < extra else 0)
        n_missing = min(n_missing, N)

        if n_missing <= 0:
            continue

        values = data_missing.iloc[:, col_idx].values.astype(float)
        ranks = pd.Series(values).rank(pct=True, na_option='bottom').values
        probs = ranks + 0.01
        probs = probs / probs.sum()

        valid_idx = np.where(~pd.isna(data_missing.iloc[:, col_idx]))[0]

        if len(valid_idx) == 0:
            continue

        valid_probs = probs[valid_idx]
        valid_probs = valid_probs / valid_probs.sum()

        n_to_select = min(n_missing, len(valid_idx))
        selected = np.random.choice(valid_idx, size=n_to_select, replace=False, p=valid_probs)

        for idx in selected:
            data_missing.iloc[idx, col_idx] = np.nan

    return data_missing


def introduce_mcar(data, missing_rate, seed):
    """MCAR para comparação."""
    np.random.seed(seed)
    data_missing = data.copy()
    N = len(data)
    M = len(data.columns)
    MDR = missing_rate

    while True:
        X = np.random.randint(0, N)
        Y = np.random.randint(0, M)
        if not pd.isna(data_missing.iloc[X, Y]):
            data_missing.iloc[X, Y] = np.nan
            R = data_missing.isna().sum().sum() / (N * M)
            if R >= MDR:
                break
    return data_missing


def test_delete_allblank():
    """Testa o parâmetro delete_allblank."""
    print("=" * 70)
    print("TESTE 1: Parâmetro delete_allblank")
    print("=" * 70)

    # Criar dataset com linhas 100% vazias forçadas
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=['f1', 'f2', 'f3', 'f4'])

    # Usar MCAR com taxa alta para potencialmente criar linhas vazias
    data_missing = introduce_mcar(data, 0.30, seed=42)

    # Forçar 2 linhas completamente vazias
    data_missing.iloc[10, :] = np.nan
    data_missing.iloc[50, :] = np.nan

    n_allblank = data_missing.isna().all(axis=1).sum()
    print(f"\nDataset: {data_missing.shape}")
    print(f"Missings: {data_missing.isna().sum().sum()}")
    print(f"Linhas 100% vazias: {n_allblank}")

    # Teste com delete_allblank=True
    print("\n--- delete_allblank=True ---")
    imputer1 = ISCAkCore(verbose=True, fast_mode=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result1 = imputer1.impute(data_missing.copy(), interactive=False, delete_allblank=True)

    print(f"Shape resultado: {result1.shape}")
    print(f"Índices removidos: {imputer1._removed_allblank_indices}")

    # Teste com delete_allblank=False
    print("\n--- delete_allblank=False ---")
    imputer2 = ISCAkCore(verbose=True, fast_mode=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result2 = imputer2.impute(data_missing.copy(), interactive=False, delete_allblank=False)

    print(f"Shape resultado: {result2.shape}")
    print(f"Missings restantes: {result2.isna().sum().sum()}")

    return True


def test_mnar():
    """Testa a nova função MNAR."""
    print("\n" + "=" * 70)
    print("TESTE 2: Nova função MNAR")
    print("=" * 70)

    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=['f1', 'f2', 'f3', 'f4'])

    # Aplicar MNAR
    data_mnar = introduce_mnar(data, 0.20, seed=42)

    print(f"\nDataset original: {data.shape}")
    print(f"Missings introduzidos: {data_mnar.isna().sum().sum()}")
    print(f"Taxa efectiva: {data_mnar.isna().sum().sum() / data_mnar.size:.2%}")

    # Verificar que valores altos têm mais missings
    print("\n--- Verificação MNAR (valores altos → mais missings) ---")
    for col in data.columns:
        if data_mnar[col].isna().any():
            missing_idx = data_mnar[col].isna()
            mean_original_missing = data.loc[missing_idx, col].mean()
            mean_original_present = data.loc[~missing_idx, col].mean()
            print(f"{col}: média valores missing={mean_original_missing:.3f}, "
                  f"média valores presentes={mean_original_present:.3f}")

    # Imputar
    print("\n--- Imputação ---")
    imputer = ISCAkCore(verbose=True, fast_mode=True)
    result = imputer.impute(data_mnar, interactive=False)

    print(f"\nMissings após imputação: {result.isna().sum().sum()}")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTES DAS ALTERAÇÕES RECENTES")
    print("=" * 70 + "\n")

    try:
        test_delete_allblank()
        print("\n✅ TESTE 1 PASSOU\n")
    except Exception as e:
        print(f"\n❌ TESTE 1 FALHOU: {e}\n")
        import traceback
        traceback.print_exc()

    try:
        test_mnar()
        print("\n✅ TESTE 2 PASSOU\n")
    except Exception as e:
        print(f"\n❌ TESTE 2 FALHOU: {e}\n")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("TESTES CONCLUÍDOS")
    print("=" * 70)
