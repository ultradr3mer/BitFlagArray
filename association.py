from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import fpgrowth, association_rules

from commonEncoding import get_bits
from reverseEncoding import IndexedBit, Association, add_rule


def _one_hot(columns: np.ndarray) -> pd.DataFrame:
    """Bit-Matrix (n_rows, n_bits) → One-Hot-DataFrame.

    Jedes bit_i=v wird eine eigene bool-Spalte.
    """
    n_rows, n_bits = columns.shape
    col_names = [f"{i}={v}" for i in range(n_bits) for v in (0, 1)]
    one_hot = pd.DataFrame(False, index=range(n_rows), columns=col_names)
    for row_idx in range(n_rows):
        for col_idx in range(n_bits):
            one_hot.loc[row_idx, f"{col_idx}={int(columns[row_idx, col_idx])}"] = True
    return one_hot


def _parse_item(s: str) -> IndexedBit:
    idx, val = s.split("=")
    return IndexedBit(int(idx), int(val))


def _find_fixed_bits(columns: np.ndarray) -> List[IndexedBit]:
    """Spalten, die in allen Zeilen denselben Wert haben (Level 0)."""
    result: List[IndexedBit] = []
    for idx in range(columns.shape[1]):
        uniq = np.unique(columns[:, idx])
        if len(uniq) == 1:
            result.append(IndexedBit(idx, int(uniq[0])))
    return result


def build(columns: np.ndarray) -> Dict[int, List[Association]]:
    """
    Findet deterministische Assoziationsregeln (confidence = 1.0) aus
    einer Bit-Matrix `columns` (n_rows × n_bits).

    - Level-0 (konstante Spalten) als `Association(frozenset(), ...)`.
    - Bedingte Regeln via mlxtend `fpgrowth` + `association_rules`.
    - Redundante Superset-Regeln werden durch `add_rule` entfernt.

    Returns:
        `rules_by_consequent` (Index nach Konsequent-Index).
    """
    n_rows, n_bits = columns.shape
    rules_by_consequent: Dict[int, List[Association]] = defaultdict(list)

    # Level 0: konstante Spalten (mlxtend liefert diese nicht als Regeln)
    for cons in _find_fixed_bits(columns):
        add_rule(Association(frozenset(), cons), rules_by_consequent)

    # One-Hot-Encoding für mlxtend
    one_hot = _one_hot(columns)

    # Frequent Itemsets — fpgrowth skaliert besser als apriori
    min_support = 1.0 / n_rows
    freq = fpgrowth(one_hot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return rules_by_consequent

    # Deterministische Regeln, jeweils ein-Bit-Konsequent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rules_df = association_rules(freq, metric="confidence", min_threshold=1.0)
    if not rules_df.empty:
        rules_df = rules_df[rules_df["consequents"].apply(lambda s: len(s) == 1)]
        # Nach Antezedenz-Groesse sortieren, damit kleinere Regeln zuerst
        # eingetragen werden und Superset-Redundanz via add_rule entfaellt.
        rules_df = rules_df.sort_values(
            by="antecedents", key=lambda s: s.apply(len)
        )

        for _, rule in rules_df.iterrows():
            antecedent = frozenset(_parse_item(s) for s in rule["antecedents"])
            consequent = _parse_item(next(iter(rule["consequents"])))
            add_rule(Association(antecedent, consequent), rules_by_consequent)

    return rules_by_consequent


if __name__ == "__main__":
    bits_to_take = 32
    num_possible = np.pow(2, bits_to_take)
    base = Path("bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint32)

        values, counts = np.unique(x, return_counts=True)

        bits = get_bits(values)

        rules = build(bits)
        for target, rule_list in sorted(rules.items()):
            print(f"target {target}:")
            for rule in rule_list:
                print(f"  {rule}")