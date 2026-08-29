from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import fpgrowth, association_rules

from clarautils.BitFlagArray import Bitty, SliceView
from clarautils.commonEncoding import get_bits
from reverseEncoding import IndexedBit, Association, add_rule, determined_indices


def _one_hot(bit_view: SliceView, orig_indices: List[int]) -> pd.DataFrame:
    """Bit-View → One-Hot-DataFrame mit Original-Indizes als Spaltennamen."""
    columns_bits = bit_view.get_bitwise()
    n_rows = columns_bits.shape[0]
    col_names = [f"{idx}={v}" for idx in orig_indices for v in (0, 1)]
    one_hot = pd.DataFrame(False, index=range(n_rows), columns=col_names)
    for row_idx in range(n_rows):
        for col_pos, idx in enumerate(orig_indices):
            one_hot.loc[row_idx, f"{idx}={int(columns_bits[row_idx, col_pos])}"] = True
    return one_hot


def _parse_item(s: str) -> IndexedBit:
    idx, val = s.split("=")
    return IndexedBit(int(idx), int(val))


def _find_fixed_bits(bit_view: SliceView, orig_indices: List[int]) -> List[IndexedBit]:
    """Konstante Spalten (Level 0) mit ihren Original-Indizes."""
    columns_bits = bit_view.get_bitwise()
    result: List[IndexedBit] = []
    for pos, idx in enumerate(orig_indices):
        uniq = np.unique(columns_bits[:, pos])
        if len(uniq) == 1:
            result.append(IndexedBit(idx, int(uniq[0])))
    return result


def _rules_from_itemsets(
    freq: pd.DataFrame,
    rules_by_consequent: Dict[int, List[Association]],
) -> None:
    """Filtert deterministische Regeln aus Frequent Itemsets und traegt
    sie (nicht-redundant) via `add_rule` ein."""
    if freq.empty:
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rules_df = association_rules(freq, metric="confidence", min_threshold=1.0)
    if rules_df.empty:
        return

    rules_df = rules_df[rules_df["consequents"].apply(lambda s: len(s) == 1)]
    rules_df = rules_df.sort_values(by="antecedents", key=lambda s: s.apply(len))

    for _, rule in rules_df.iterrows():
        antecedent = frozenset(_parse_item(s) for s in rule["antecedents"])
        consequent = _parse_item(next(iter(rule["consequents"])))
        add_rule(Association(antecedent, consequent), rules_by_consequent)


def build(
    data: Bitty,
    *,
    max_level: int | None = None,
    min_support: float | None = None,
) -> Dict[int, List[Association]]:
    """
    Findet deterministische Assoziationsregeln (confidence = 1.0) aus
    einer `Bitty`, level-weise wie `find_association_rules`.

    Schritte:
      1. Level 0: konstante Spalten finden und entfernen.
      2. Level 1..k: `fpgrowth` mit `max_len=k+1` auf den verbleibenden
         Bits.  Nur Regeln mit k-Bit-Antezedenz werden in Level k neu
         hinzugefuegt (kleinere werden per `add_rule` als Redundanz
         erkannt).  Indizes mit bedingten Regeln werden NICHT entfernt
         — unterschiedliche Antezedenzen koennen denselben Index mit
         unterschiedlichem Wert bestimmen.

    Noise-Bits, die nie deterministisch werden, bleiben im Datensatz.
    `fpgrowth` mit `max_len` deckelt die Itemset-Groesse pro Level.

    Args:
        data:       Bitty mit den gepackten Werten.
        max_level:  Maximale Antezedenz-Groesse (default: bit_count).
        min_support: Mindest-Support fuer fpgrowth.
                     Default: 1/n_rows (jedes vorkommende Itemset).

    Returns:
        `rules_by_consequent` (Index nach Konsequent-Index).
    """
    n_rows = len(data)
    bit_count = data.get_bit_count()
    if max_level is None:
        max_level = bit_count
    if min_support is None:
        min_support = 1.0 / n_rows

    all_indices = list(range(bit_count))
    rules_by_consequent: Dict[int, List[Association]] = defaultdict(list)

    print(f"build: {n_rows} rows, {bit_count} bits, max_level={max_level}, "
          f"min_support={min_support:.4f}")

    # Level 0: konstante Spalten finden und entfernen
    fixed = _find_fixed_bits(data, all_indices)
    if fixed:
        print(f"  level 0: {len(fixed)} fixed bits: "
              f"{', '.join(str(b) for b in sorted(fixed, key=lambda b: b.index))}")
    for cons in fixed:
        add_rule(Association(frozenset(), cons), rules_by_consequent)

    fixed_indices = {bit.index for bit in fixed}
    remaining = [i for i in all_indices if i not in fixed_indices]
    print(f"  remaining after level 0: {len(remaining)} bits {remaining}")

    # Level 1..max_level
    for level in range(1, max_level + 1):
        if not remaining:
            break

        bit_view = data.b[remaining]
        one_hot = _one_hot(bit_view, remaining)

        freq = fpgrowth(
            one_hot,
            min_support=min_support,
            max_len=level + 1,
            use_colnames=True,
        )
        print(f"  level {level}: {len(freq)} itemsets (max_len={level + 1})")

        before = sum(len(v) for v in rules_by_consequent.values())
        _rules_from_itemsets(freq, rules_by_consequent)
        after = sum(len(v) for v in rules_by_consequent.values())

        if after > before:
            print(f"          +{after - before} rules")
            for target, rule_list in rules_by_consequent.items():
                for r in rule_list[before:]:
                    print(f"            {r}")
        else:
            print(f"          no new rules")

    total = sum(len(v) for v in rules_by_consequent.values())
    print(f"build: done, {total} rules total")
    for target, rule_list in sorted(rules_by_consequent.items()):
        print(f"  target {target}: {len(rule_list)} rule(s)")
        for r in rule_list:
            print(f"    {r}")

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

        bitty = Bitty(values)

        rules = build(bitty)
        for target, rule_list in sorted(rules.items()):
            print(f"target {target}:")
            for rule in rule_list:
                print(f"  {rule}")