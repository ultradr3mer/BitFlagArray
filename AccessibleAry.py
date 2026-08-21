from enum import StrEnum, Enum
from typing import NamedTuple, List, Iterable, Generic, TypeVar, Any, Type, Literal

import numpy.typing as npt
import numpy as np

from BitFlagArray import NBitArray, NBitAryOnly

T_item = TypeVar('T_item', bound=object)
T_fld = TypeVar('T_fld', bound=Enum)


class AccessibleAry(Generic[T_item,T_fld]):
    def __init__(self, inner: Iterable[Any] | npt.NDArray[np.unsignedinteger]):
        self.inner = inner

    def __getitem__(self, key):
        if isinstance(key, Enum) or isinstance(key, str):
            return self.select(key)
        return self.inner.__getitem__(key)

    def __setitem__(self, key, value):
        self.inner.__setitem__(key, value)

    def select(self, attr):
        return [getattr(i, attr, ) for i in self.inner]


class IndexedBit(NamedTuple):
    index: int
    value: int

class LoopData(NamedTuple):
    class Fields(StrEnum):
        data = "dat ①/Ⓐ⒜ a"
        condition = "condition"
        undefined_indices = "undefined_indices"

    data: NBitArray
    condition: List[IndexedBit]
    undefined_indices: List[int]


item1 = LoopData(NBitAryOnly(np.array([1, 2, 3, 3]),4),
                 [IndexedBit(0, 0)],
                 undefined_indices=[1, 2, 3, 4, 5])
item2 = LoopData(NBitAryOnly(np.array([1, 4, 1, 3]),4),
                 [IndexedBit(1, 1)],
                 undefined_indices=[1, 2, 3, 4, 5])
item3 = LoopData(NBitAryOnly(np.array([2, 7, 3, 3]),4),
                 [IndexedBit(3, 0)],
                 undefined_indices=[1, 2, 3, 4, 5])

ary = AccessibleAry([item1, item2, item3])

print(ary[1])
print(ary[2])

con = ary[LoopData.Fields.condition]
print(con)