from enum import StrEnum
from idlelib.debugger_r import FrameProxy
from itertools import permutations
from pathlib import Path
from typing import NamedTuple, List, Iterable, Tuple, Set, FrozenSet, Dict

import numpy as np
import numpy.typing as npt

from BitFlagArray import Bitty, NBitArray
from commonEncoding import get_bits

base = Path("bins")

bits_to_take = 32
num_possible = np.pow(2, bits_to_take)

class IndexedBit(NamedTuple):
    class Fields(StrEnum):
        index = "index"
        value = "value"
    index: int
    value: int

    def __repr__(self):
        return f"{self.index}={self.value})"

    @classmethod
    def from_multiple(cls,
                      mult_indices : List[int] | Tuple[int,...] | npt.NDArray[np.unsignedinteger],
                      mult_values : List[int] | Tuple[int,...] | npt.NDArray[np.unsignedinteger] | int) -> Set[IndexedBit]:
        if isinstance(mult_values, int):
            mult_values = get_bits(mult_values, len(mult_indices))
        gen = [IndexedBit(i,v) for i,v in zip(mult_indices, mult_values)]
        return set(gen)

class Association(NamedTuple):
    antecedent: Set[IndexedBit]
    consequent: IndexedBit

class PermutationGen:
    def __init__(self, item_count: int = 32, variant_count: int = 2):
        self.items = range(item_count)
        self.variants = range(variant_count)

    def gen(self, draw_count: int, data: NBitArray) -> Iterable[Set[IndexedBit]]:
        if draw_count == 0:
            return
        for perm in permutations(self.items, draw_count):
            for val in np.unique(data.b[perm]):
              yield IndexedBit.from_multiple(perm, val)

def get_consequent(data: NBitArray,
                   antecedent: FrozenSet[IndexedBit],
                   associations: Dict[FrozenSet[IndexedBit],IndexedBit]) -> List[Association]:
    # From antecendent => getassociatons => determine undefined_indices
    defined = associations[antecedent]
    undefined_indices = [i for i in range(data.get_bit_count()) if i not in defined or antecedent]
    undefined_data = data.b[undefined_indices].get_bitwise()
    mean = np.mean(undefined_data, axis=0)
    association_rules = []
    still_undefined: set[int] = set()
    for index, bit in zip(item.undefined_indices, mean):
        if bit == 1 or bit == 0:
            association_rules.append(Association(item, IndexedBit(index, bit)))
        else:
            still_undefined.add(index)
    return association_rules

def find_association_rules(data: Bitty):
    associations: List[Association] = []
    perm_gen = PermutationGen()

    for level_idx in range(data.get_bit_count()):
        level_associations: List[Association] = []
        antecedent_list = perm_gen.gen(draw_count=level_idx, data=data)
        for possible_antecedent in antecedent_list:
            c, undef = get_consequent(data, possible_antecedent, associations)
            level_associations.extend(c)

        print("loop:", i, "associations:", f"\n".join([str(value) for value in level_associations]))
        associations.extend(level_associations)

    return associations

for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
    with open(path, "rb") as f:
        buffer = f.read()
    name = path.name

    x = np.frombuffer(buffer, dtype=np.uint32)

    values, counts = np.unique(x, return_counts=True)

    find_association_rules(Bitty(values))
