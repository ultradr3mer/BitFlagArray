from collections import defaultdict
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, List, Tuple, Iterable, FrozenSet, Set, Dict

import numpy as np
import numpy.typing as npt

from BitFlagArray import Bitty, NBitArray
from commonEncoding import get_bits

base = Path("bins")


class IndexedBit(NamedTuple):
    class Fields(StrEnum):
        index = "index"
        value = "value"

    index: int
    value: int

    def __repr__(self):
        return f"{self.index}={self.value})"

    @classmethod
    def from_multiple(
        cls,
        mult_indices: List[int] | Tuple[int, ...] | npt.NDArray[np.unsignedinteger],
        mult_values: List[int] | Tuple[int, ...] | npt.NDArray[np.unsignedinteger] | int,
    ) -> FrozenSet["IndexedBit"]:
        if isinstance(mult_values, int):
            mult_values = get_bits(mult_values, len(mult_indices))
        return frozenset(cls(i, int(v)) for i, v in zip(mult_indices, mult_values))


class Association(NamedTuple):
    antecedent: FrozenSet[IndexedBit]
    consequent: IndexedBit


def add_rule(
    rule: Association,
    rules_by_consequent: Dict[int, List[Association]],
) -> bool:
    """
    Fuegt die Regel nur hinzu, wenn keine staerkere/bereits bekannte
    Regel mit gleichem Ziel existiert.

    {a=1} => c=1 ist bereits bekannt => {a=1, b=0} => c=1 ist redundant.

    True:  Regel wurde hinzugefuegt
    False: Regel war redundant
    """
    target = rule.consequent.index

    if any(
        old.antecedent <= rule.antecedent and old.consequent == rule.consequent
        for old in rules_by_consequent.get(target, [])
    ):
        return False

    rules_by_consequent.setdefault(target, []).append(rule)
    return True


def applicable_rules(
    rules_by_consequent: Dict[int, List[Association]],
    known: Set[IndexedBit] | FrozenSet[IndexedBit],
) -> List[Association]:
    """Regeln, deren Antezedenz komplett in `known` enthalten ist und
    deren Zielindex noch nicht bestimmt ist."""
    known_indices = {bit.index for bit in known}

    return [
        rule
        for target_index, rules in rules_by_consequent.items()
        if target_index not in known_indices
        for rule in rules
        if rule.antecedent <= known
    ]


def candidate_consequent_indices(
    antecedent: FrozenSet[IndexedBit] | Set[IndexedBit],
    known: Set[IndexedBit] | FrozenSet[IndexedBit],
    bit_count: int,
) -> Set[int]:
    """Indizes, die weder in der Antezedenz noch bereits bestimmt sind."""
    antecedent_indices = {bit.index for bit in antecedent}
    known_indices = {bit.index for bit in known}
    return set(range(bit_count)) - antecedent_indices - known_indices


class CombinationGen:
    """Erzeugt alle in `bits` tatsaechlich vorkommenden Antezedenzen
    fuer eine gegebene Kombinationsgroesse (Indizes sind ungeordnet,
    daher Kombinationen statt Permutationen)."""

    def __init__(self, item_count: int):
        self.items = range(item_count)

    def gen(
        self,
        draw_count: int,
        bits: npt.NDArray[np.integer],
    ) -> Iterable[FrozenSet[IndexedBit]]:
        if draw_count == 0:
            yield frozenset()
            return

        for comb in combinations(self.items, draw_count):
            sub = bits[:, list(comb)]
            for row in np.unique(sub, axis=0):
                yield IndexedBit.from_multiple(comb, tuple(int(v) for v in row))


def _matching_mask(
    bits: npt.NDArray[np.integer],
    antecedent: FrozenSet[IndexedBit],
) -> npt.NDArray[np.bool_]:
    mask = np.ones(bits.shape[0], dtype=bool)
    for bit in antecedent:
        mask &= bits[:, bit.index] == bit.value
    return mask


def find_consequents(
    bits: npt.NDArray[np.integer],
    antecedent: FrozenSet[IndexedBit],
    known_indices: Set[int],
    rules_by_consequent: Dict[int, List[Association]],
) -> List[Association]:
    """Bestimmt alle durch `antecedent` festlegbaren Bits, deren Index
    noch nicht bekannt ist, und traegt sie (nicht-redundant) ein."""
    bit_count = bits.shape[1]
    candidates = (
        set(range(bit_count))
        - {bit.index for bit in antecedent}
        - known_indices
    )
    if not candidates:
        return []

    mask = _matching_mask(bits, antecedent)
    if not mask.any():
        return []

    new_rules: List[Association] = []
    for idx in candidates:
        col = bits[mask, idx]
        uniq = np.unique(col)
        if len(uniq) == 1:
            rule = Association(antecedent, IndexedBit(idx, int(uniq[0])))
            if add_rule(rule, rules_by_consequent):
                new_rules.append(rule)
    return new_rules


def find_association_rules(
    data: Bitty,
) -> Dict[int, List[Association]]:
    """
    Level 0 (leere Antezedenz) sammelt einzeln fixierte Werte.
    Jedes hoehere Level erzeugt Antezedenzen der Groesse `level_idx`
    und leitet daraus weitere Konsequenten ab. Bereits bestimmte
    Indizes stoeren nicht weiter, da sie als Kandidaten ausgeschlossen
    werden.
    """
    bits = data.get_bitwise()
    bit_count = data.get_bit_count()

    rules_by_consequent: Dict[int, List[Association]] = defaultdict(list)
    known_indices: Set[int] = set()
    gen = CombinationGen(bit_count)

    for level_idx in range(bit_count + 1):
        for antecedent in gen.gen(draw_count=level_idx, bits=bits):
            find_consequents(bits, antecedent, known_indices, rules_by_consequent)
        known_indices = set(rules_by_consequent.keys())

    return rules_by_consequent


if __name__ == "__main__":
    bits_to_take = 32
    num_possible = np.pow(2, bits_to_take)

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint32)

        values, counts = np.unique(x, return_counts=True)

        rules = find_association_rules(Bitty(values))
        for target, rule_list in rules.items():
            print(f"target {target}:")
            for rule in rule_list:
                print(f"  {rule}")
