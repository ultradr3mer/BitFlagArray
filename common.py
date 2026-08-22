from argparse import ArgumentError
from enum import StrEnum
from typing import TypeVar, Generic, Type, List, Iterable, Any, NamedTuple

import numpy as np

T_ex = TypeVar('T_ex', bound=Exception)
T_ret = TypeVar('T_ret')


#====================================================
#               INLINE EXCEPTION UTIL
#====================================================

class ExcRaiser(Generic[T_ret]):
    def __init__(self, exception_cls: Type[T_ex], message: str):
        self.exception_cls = exception_cls
        self.message = message

    def do_raise(self) -> T_ret:
        raise self.exception_cls(self.message)

#====================================================
#                   TYPE DETECTION
#====================================================

exc_to_many_bit: ExcRaiser[type] = ExcRaiser(ArgumentError, "To many bits to fit.")
_UINT_TYPES = (np.ubyte, np.uint16, np.uint32, np.uint64)

def _get_type(value: int, attr: str):
    for dtype in _UINT_TYPES:
        if value <= getattr(np.iinfo(dtype), attr):
            return dtype

    return exc_to_many_bit.do_raise()


def get_type_for_scalar(value: int):
    return _get_type(value, "max")

def get_type_for_bit_count(bit_count: int):
    return _get_type(bit_count, "bits")

def get_type_for_array(ary: np.ndarray | List[Any] | Iterable[Any]):
    return _get_type(np.max(ary), "max")


def iter_bits(data):
    for byte in data:
        for i in range(8):
            yield (byte >> (7 - i)) & 1


#====================================================
#           ACCESSING FELDS OVER CONTAINER
#====================================================

T_item = TypeVar('T_item')

class AccessibleAry(Generic[T_item]):
    def __init__(self, inner: List[Any]):
        self.inner = inner

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_item_attr(key)
        else:
            return self.inner.__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.set_item_attr(key, value)
        self.__setitem__(key, value)

    def __array__(self, dtype=None, copy=False):
        return np.array(self.inner, dtype=dtype, copy=copy)

    def __len__(self):
        if isinstance(self.inner, np.ndarray):
            return self.inner.size
        elif isinstance(self.inner, Iterable):
            return len(self.inner)
        return len(self.inner)

    def __repr__(self):
        return f"{type(self)}, length={self.__len__()},\n{self.inner}"

    def __eq__(self, other):
        return np.array(self) == np.array(other)

    def set_item_attr(self, key, value):
        for item in self.inner:
            setattr(item, key, value)
        pass

    def get_item_attr(self, attr):
        return [getattr(i, attr,) for i in self.inner]

    def extend(self, param):
        self.inner.extend(param)




class LoopData(NamedTuple):
    class Fields(StrEnum):
        data = "data"
        condition = "condition"
        undefined_indices = "undefined_indices"
    data: NBitArray
    condition: List[IndexedBit]
    undefined_indices: List[int]

