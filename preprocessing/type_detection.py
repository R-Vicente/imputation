import numpy as np
import pandas as pd
import json

class MixedDataHandler:
    def __init__(self, categorical_threshold: int = 10):
        self.categorical_threshold = categorical_threshold
        self.numeric_cols = []
        self.binary_cols = []
        self.nominal_cols = []
        self.ordinal_cols = []
        self.original_dtypes = {}
        self.nominal_mappings = {}
        self.ordinal_mappings = {}
        self.is_mixed = False

    def detect_types(self, data: pd.DataFrame, 
                    force_categorical: list = None,
                    force_ordinal: dict = None,
                    interactive: bool = True) -> dict:
        force_categorical = force_categorical or []
        force_ordinal = force_ordinal or {}
        type_info = {}
        ambiguous_cols = []
        for col in data.columns:
            dtype = data[col].dtype
            n_unique = data[col].nunique()
            if col in force_ordinal:
                type_info[col] = {
                    'type': 'ordinal',
                    'reason': 'user specified',
                    'n_categories': n_unique,
                    'order': force_ordinal[col]
                }
                continue
            if col in force_categorical:
                type_info[col] = {
                    'type': 'nominal',
                    'reason': 'user specified',
                    'n_categories': n_unique
                }
                continue
            auto_type = self._auto_detect_type(data[col], dtype, n_unique)
            if (auto_type['type'] == 'ambiguous' and 
                self._is_genuinely_ambiguous(data[col], dtype, n_unique)):
                ambiguous_cols.append(col)
                type_info[col] = auto_type
            else:
                if auto_type['type'] == 'ambiguous':
                    type_info[col] = {
                        'type': 'numeric',
                        'reason': 'ambiguous but defaulting to numeric',
                        'n_categories': None
                    }
                else:
                    type_info[col] = auto_type
        if ambiguous_cols and interactive:
            resolved = self._ask_user_about_ambiguous(data, ambiguous_cols)
            for col in resolved['numeric']:
                type_info[col] = {
                    'type': 'numeric',
                    'reason': 'user classified as numeric',
                    'n_categories': None
                }
            for col in resolved['categorical']:
                type_info[col] = {
                    'type': 'nominal',
                    'reason': 'user classified as nominal',
                    'n_categories': data[col].nunique()
                }
            for col, order in resolved['ordinal'].items():
                type_info[col] = {
                    'type': 'ordinal',
                    'reason': 'user classified as ordinal',
                    'n_categories': len(order),
                    'order': order
                }
        elif ambiguous_cols and not interactive:
            for col in ambiguous_cols:
                type_info[col] = {
                    'type': 'numeric',
                    'reason': 'ambiguous, defaulting to numeric (non-interactive)',
                    'n_categories': None
                }
        return type_info

    def _auto_detect_type(self, series: pd.Series, dtype, n_unique: int) -> dict:
        if dtype == 'object' or dtype.name == 'category' or pd.api.types.is_string_dtype(dtype):
            return {'type':'nominal','reason':'non-numeric dtype','n_categories':n_unique}
        if dtype == 'bool':
            return {'type':'binary','reason':'boolean dtype','n_categories':2}
        if pd.api.types.is_numeric_dtype(dtype) and n_unique == 2:
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({0,1}):
                return {'type':'binary','reason':'two values {0, 1}','n_categories':2}
        if pd.api.types.is_numeric_dtype(dtype) and n_unique <= self.categorical_threshold:
            unique_vals = sorted(series.dropna().unique())
            if len(unique_vals) > 0:
                is_consecutive = all(
                    abs(unique_vals[i+1] - unique_vals[i] - 1) < 0.01
                    for i in range(len(unique_vals)-1)
                ) if len(unique_vals) > 1 else True
                if is_consecutive and 3 <= n_unique <= 10:
                    return {'type':'ambiguous','reason':f'{n_unique} consecutive values','n_categories':n_unique}
                elif is_consecutive and n_unique == 2:
                    return {'type':'ambiguous','reason':'two consecutive values (not 0,1)','n_categories':2}
                else:
                    return {'type':'numeric','reason':f'{n_unique} non-consecutive values','n_categories':None}
        return {'type':'numeric','reason':'numeric with many unique values','n_categories':None}

    def _is_genuinely_ambiguous(self, series: pd.Series, dtype, n_unique: int) -> bool:
        if pd.api.types.is_numeric_dtype(dtype):
            if n_unique < 2 or n_unique > 10:
                return False
            unique_vals = sorted(series.dropna().unique())
            is_consecutive = all(
                abs(unique_vals[i+1] - unique_vals[i] - 1) < 0.01
                for i in range(len(unique_vals)-1)
            )
            if not is_consecutive:
                return False
            if n_unique == 2 and set(unique_vals).issubset({0,1}):
                return False
            return True
        if dtype == 'object' or pd.api.types.is_string_dtype(dtype):
            if n_unique > 20:
                return False
            return n_unique >= 2
        return False

    def _ask_user_about_ambiguous(self, data: pd.DataFrame, ambiguous_cols: list) -> dict:
        print("\\n" + "="*70)
        print("VARIAVEIS AMBIGUAS DETECTADAS".center(70))
        print("="*70)
        categorical = []
        ordinal = {}
        numeric = []
        for col in ambiguous_cols:
            unique_vals = sorted(data[col].dropna().unique())
            n_unique = len(unique_vals)
            print(f"\\nColuna: '{col}'")
            print(f"Valores unicos ({n_unique}): {unique_vals[:15]}")
            print("\\nComo tratar esta variavel?")
            print("[1] Numerica (ex: idade, rating, score)")
            print("[2] Categorica Nominal (ex: codigo de categoria, ID de grupo)")
            print("[3] Categorica Ordinal (ex: educacao, nivel, tamanho)")
            while True:
                choice = input("Escolha (1/2/3): ").strip()
                if choice == '1':
                    numeric.append(col)
                    break
                elif choice == '2':
                    categorical.append(col)
                    break
                elif choice == '3':
                    order_input = input("Ordem: ").strip()
                    order_list = [x.strip() for x in order_input.split(',')]
                    try:
                        if pd.api.types.is_numeric_dtype(data[col].dtype):
                            order_list = [float(x) if '.' in x else int(x) for x in order_list]
                    except ValueError:
                        print("ERRO: Valores devem ser numericos para esta coluna")
                        continue
                    unique_set = set(map(str, unique_vals))
                    order_set = set(map(str, order_list))
                    missing = unique_set - order_set
                    extra = order_set - unique_set
                    if missing or extra:
                        retry = input("\\nTentar novamente? (s/n): ").strip().lower()
                        if retry == 's':
                            continue
                        else:
                            numeric.append(col)
                            break
                    ordinal[col] = order_list
                    break
                else:
                    print("Opcao invalida. Escolha 1, 2 ou 3.\\n")
        print("="*70)
        return {'categorical': categorical, 'ordinal': ordinal, 'numeric': numeric}

    def fit_transform(self, data: pd.DataFrame, 
                     force_categorical: list = None,
                     force_ordinal: dict = None,
                     interactive: bool = True,
                     verbose: bool = False) -> tuple:
        type_info = self.detect_types(data, force_categorical, force_ordinal, interactive)
        self.numeric_cols = [col for col, info in type_info.items() if info['type'] == 'numeric']
        self.binary_cols = [col for col, info in type_info.items() if info['type'] == 'binary']
        self.nominal_cols = [col for col, info in type_info.items() if info['type'] == 'nominal']
        self.ordinal_cols = [col for col, info in type_info.items() if info['type'] == 'ordinal']
        self.is_mixed = len(self.binary_cols) + len(self.nominal_cols) + len(self.ordinal_cols) > 0
        if verbose:
            self._print_type_summary()
        data_encoded = data.copy()
        encoding_info = {}
        encoding_info['numeric'] = {col: 'no encoding' for col in self.numeric_cols}
        encoding_info['binary'] = {col: 'no encoding' for col in self.binary_cols}
        for col in self.nominal_cols:
            self.original_dtypes[col] = data[col].dtype
            categories = data[col].dropna().unique()
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            reverse_mapping = {idx: cat for cat, idx in mapping.items()}
            self.nominal_mappings[col] = {'forward': mapping, 'reverse': reverse_mapping, 'n_categories': len(categories)}
            data_encoded[col] = data[col].map(mapping)
            encoding_info.setdefault('nominal', {})[col] = {'categories': list(categories), 'n_categories': len(categories), 'encoding': 'integer codes'}
        for col in self.ordinal_cols:
            self.original_dtypes[col] = data[col].dtype
            order = type_info[col]['order']
            n_cats = len(order)
            if n_cats == 1:
                scaled_values = [0.5]
            else:
                scaled_values = [i / (n_cats - 1) for i in range(n_cats)]
            mapping = {cat: val for cat, val in zip(order, scaled_values)}
            reverse_mapping = {val: cat for cat, val in mapping.items()}
            self.ordinal_mappings[col] = {'forward': mapping, 'reverse': reverse_mapping, 'order': order, 'n_categories': n_cats}
            data_encoded[col] = data[col].map(mapping)
            encoding_info.setdefault('ordinal', {})[col] = {'order': order, 'n_categories': n_cats, 'encoding': 'uniform [0,1]', 'scaled_values': scaled_values}
        return data_encoded, encoding_info

    def inverse_transform(self, data_encoded: pd.DataFrame) -> pd.DataFrame:
        data_decoded = data_encoded.copy()
        for col in self.nominal_cols:
            if col not in data_decoded.columns:
                continue
            reverse_mapping = self.nominal_mappings[col]['reverse']
            values_rounded = data_decoded[col].round().astype('Int64')
            max_idx = len(reverse_mapping) - 1
            values_clipped = values_rounded.clip(lower=0, upper=max_idx)
            data_decoded[col] = values_clipped.map(reverse_mapping)
            try:
                data_decoded[col] = data_decoded[col].astype(self.original_dtypes[col])
            except:
                pass
        for col in self.ordinal_cols:
            if col not in data_decoded.columns:
                continue
            reverse_mapping = self.ordinal_mappings[col]['reverse']
            order = self.ordinal_mappings[col]['order']
            n_cats = len(order)
            def find_nearest_category(val):
                if pd.isna(val):
                    return np.nan
                val = np.clip(val, 0, 1)
                if n_cats == 1:
                    return order[0]
                else:
                    idx = round(val * (n_cats - 1))
                    idx = np.clip(idx, 0, n_cats - 1)
                    return order[idx]
            data_decoded[col] = data_decoded[col].apply(find_nearest_category)
            try:
                data_decoded[col] = data_decoded[col].astype(self.original_dtypes[col])
            except:
                pass
        return data_decoded

    def _print_type_summary(self):
        print("\\nCLASSIFICACAO DE VARIAVEIS:")
        print(f"  Numericas: {len(self.numeric_cols)}")
        if self.numeric_cols[:5]:
            print(f"    {self.numeric_cols[:5]}")
        print(f"  Binarias: {len(self.binary_cols)}")
        if self.binary_cols[:5]:
            print(f"    {self.binary_cols[:5]}")
        print(f"  Nominais: {len(self.nominal_cols)}")
        if self.nominal_cols[:5]:
            print(f"    {self.nominal_cols[:5]}")
        print(f"  Ordinais: {len(self.ordinal_cols)}")
        if self.ordinal_cols[:5]:
            print(f"    {self.ordinal_cols[:5]}")
    def save_config(self, filepath: str):
        config = {
            'numeric_cols': self.numeric_cols,
            'binary_cols': self.binary_cols,
            'nominal_cols': self.nominal_cols,
            'ordinal_cols': self.ordinal_cols,
            'ordinal_mappings': {
                col: {'order': mapping['order'], 'n_categories': mapping['n_categories']}
                for col, mapping in self.ordinal_mappings.items()
            },
            'categorical_threshold': self.categorical_threshold
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\\nConfiguracao salva em: {filepath}")
    @classmethod
    def load_config(cls, filepath: str) -> tuple:
        with open(filepath, 'r') as f:
            config = json.load(f)
        force_categorical = config['nominal_cols']
        force_ordinal = {col: mapping['order'] for col, mapping in config['ordinal_mappings'].items()}
        print(f"\\nConfiguracao carregada de: {filepath}")
        print(f"  Nominais: {len(force_categorical)}")
        print(f"  Ordinais: {len(force_ordinal)}")
        return force_categorical, force_ordinal
