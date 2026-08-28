from collections import defaultdict
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, List, Tuple, Iterable, FrozenSet, Set, Dict

import numpy as np
import numpy.typing as npt

from BitFlagArray import Bitty, NBitArray, SliceView
from common import AccessibleAry
from commonEncoding import get_bits, get_number

base = Path("bins")


class IndexedBit(NamedTuple):
    class Fields(StrEnum):
        index = "index"
        value = "value"

    index: int
    value: int

    def __repr__(self):
        return f"{self.index}={self.value}"

    @classmethod
    def from_multiple(
        cls,
        mult_indices: List[int] | Tuple[int, ...] | npt.NDArray[np.unsignedinteger],
        mult_values: List[int] | Tuple[int, ...] | npt.NDArray[np.unsignedinteger] | int,
    ) -> FrozenSet["IndexedBit"]:
        if isinstance(mult_values, int):
            mult_values = get_bits(mult_values, len(mult_indices))
        return frozenset(cls(i, int(v)) for i, v in zip(mult_indices, mult_values))

class AccIdxBit(AccessibleAry[IndexedBit]):
    def __init__(self, inner: Iterable[IndexedBit] | List[IndexedBit]):
        if not isinstance(inner, List):
            inner = list(inner)
        inner.sort(key=lambda bi: bi.index)
        super().__init__(inner)

    def __repr__(self):
        return f"{self[IndexedBit.Fields.index]}={self[IndexedBit.Fields.value]}"

    def select(self, data: NBitArray) -> SliceView:
        indices = self[IndexedBit.Fields.index]
        values = self[IndexedBit.Fields.value]
        mask = data.b[indices].get_array() == get_number(values).value
        return data.i[mask]


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


# def candidate_consequent_indices(
#     antecedent: FrozenSet[IndexedBit] | Set[IndexedBit],
#     known: Set[IndexedBit] | FrozenSet[IndexedBit],
#     bit_count: int,
# ) -> Set[int]:
#     """Indizes, die weder in der Antezedenz noch bereits bestimmt sind."""
#     antecedent_indices = {bit.index for bit in antecedent}
#     known_indices = {bit.index for bit in known}
#     return set(range(bit_count)) - antecedent_indices - known_indices


class CombinationGen:
    """Erzeugt alle in `data` tatsaechlich vorkommenden Antezedenzen
    fuer eine gegebene Kombinationsgroesse.

    `combinations` (nicht `permutations`) ist korrekt: jeder Index
    erscheint pro Kombination hoechstens einmal.  Da die Antezedenz ein
    `frozenset` ist (Reihenfolge irrelevant), liefert `combinations`
    genau die mengenwertigen Teilmengen ohne Duplikate.

    Arbeitet auf den gepackten Zahlenwerten von `SliceView` – die Bits
    werden erst aufgespalten, wenn sie gebraucht werden.
    """

    def __init__(self, item_count: int):
        self.items = range(item_count)

    def gen(
        self,
        draw_count: int,
        data: Bitty,
    ) -> Iterable[FrozenSet[IndexedBit]]:
        if draw_count == 0:
            yield frozenset()
            return

        for comb in combinations(self.items, draw_count):
            comb_list = list(comb)
            for val in np.unique(data.b[comb_list]):
                yield IndexedBit.from_multiple(comb_list, int(val))


def determined_indices(
    rules_by_consequent: Dict[int, List[Association]],
) -> Set[int]:
    """Indizes, die bedingungslos (leere Antezedenz) bestimmt sind."""
    return {
        target
        for target, rules in rules_by_consequent.items()
        if any(not r.antecedent for r in rules)
    }


def find_consequents(
    data: Bitty,
    antecedent: FrozenSet[IndexedBit],
    determined: Set[int],
) -> List[IndexedBit]:
    """Bestimmt alle durch `antecedent` festlegbaren Bits, deren Index
    noch nicht bedingungslos bestimmt ist (`determined`).

    `determined` enthaelt die Indizes der Level-0-Regeln (leere
    Antezedenz).  Diese stoeren nicht weiter, da sie als Kandidaten
    ausgeschlossen werden.  Indizes mit bedingten Regeln werden nicht
    ausgeschlossen – unterschiedliche Antezedenzen koennen denselben
    Index mit unterschiedlichem Wert bestimmen.  Redundanz unter Regeln
    prueft erst `add_rule`.
    """
    bit_count = data.get_bit_count()
    candidates = (
        set(range(bit_count))
        - {bit.index for bit in antecedent}
        - determined
    )
    if not candidates:
        return []

    if antecedent:
        a = AccIdxBit(antecedent)
        matching = a.select(data)
        if len(matching) == 0:
            return []
    else:
        matching = data

    candidate_list = sorted(candidates)
    col_values = matching.b[candidate_list].get_bitwise()
    first_row = col_values[0:1, :]
    const_mask = np.all(col_values == first_row, axis=0)

    return [
        IndexedBit(idx, int(col_values[0, i]))
        for i, idx in enumerate(candidate_list)
        if const_mask[i]
    ]


def find_association_rules(
    data: Bitty,
) -> Dict[int, List[Association]]:
    """
    Level 0 (leere Antezedenz) sammelt bedingungslos fixierte Werte als
    `Association(frozenset(), ...)`.  Jedes hoehere Level erzeugt
    Antezedenzen der Groesse `level_idx` und leitet daraus weitere
    Konsequenten ab.

    Bedingungslos bestimmte Indizes (Level 0) stoeren nicht weiter,
    da sie ueber `determined_indices` als Kandidaten ausgeschlossen
    werden.  `add_rule` verhindert, dass eine bedingte Regel fuer einen
    bereits bedingungslos bestimmten Index hinzugefuegt wird.
    """
    bit_count = data.get_bit_count()

    rules_by_consequent: Dict[int, List[Association]] = defaultdict(list)
    gen = CombinationGen(bit_count)

    for level_idx in range(bit_count + 1):
        determined = determined_indices(rules_by_consequent)
        for antecedent in gen.gen(draw_count=level_idx, data=data):
            for cons in find_consequents(data, antecedent, determined):
                add_rule(Association(antecedent, cons), rules_by_consequent)

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
        for target, rule_list in sorted(rules.items()):
            print(f"target {target}:")
            for rule in rule_list:
                print(f"  {rule}")
