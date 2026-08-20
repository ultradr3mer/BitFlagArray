from argparse import ArgumentError
from typing import TypeVar, Generic, Type, List, Iterable, Any

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
