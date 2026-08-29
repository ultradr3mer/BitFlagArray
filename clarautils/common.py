from dataclasses import dataclass
from typing import TypeVar, Generic, Type, List, Iterable, Any, NamedTuple, overload, Literal, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

def get_first_or_default(value: Iterable):
    for item in value:
        return item
    return None


def get_first_or(value: Iterable, other: Any):
    for item in value:
        return item
    if isinstance(other, Exception):
        raise other
    return other

T_ex = TypeVar('T_ex', bound=Exception)
T_ret = TypeVar('T_ret')


#====================================================
#               INLINE EXCEPTION UTIL
#====================================================

class ExceptionRaiser(Generic[T_ret]):
    def __init__(self, exception_cls: Type[T_ex], message: str):
        self.exception_cls = exception_cls
        self.message = message

    def do_raise(self) -> T_ret:
        raise self.exception_cls(self.message)

ExcRaiser = ExceptionRaiser

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


def iter_bits(data):
    for byte in data:
        for i in range(8):
            yield (byte >> (7 - i)) & 1


