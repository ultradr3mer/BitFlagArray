from dataclasses import dataclass
from typing import TypeVar, Generic, Type, List, Iterable, Any, NamedTuple, overload, Literal, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

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



#====================================================
#                   TYPE DETECTION
#====================================================


#try2
#
# T_cols = TypeVar('T_cols', bound=Tuple[Any, ...])
# T_col_container = TypeVar('T_col_container', bound=npt.NDArray|np.ScalarType)
#
# # Skeleton for generic field type later
# class Skeleton(NamedTuple, Generic[T_col_container]):
#     pass
#
# T_skeleton = TypeVar('T_skeleton', bound=Skeleton[npt.NDArray|np.ScalarType])
#
#
# # Field definitions for generic field type later
# class TblAndColumn(Skeleton[T_col_container]):
#     kind: T_col_container[np.bool]
#     abs_min: T_col_container[np.uint64]
#     max: T_col_container[np.uint64]
#     bits: T_col_container[np.uint8]
#
#
# # Table type for forwarding fieldypes to table and item
# class TableType(Generic[T_skeleton], T_skeleton[np.ndarray]):
#     pass
#
#     def __iter__(self) -> Iterable[T_skeleton[np.ScalarType]]:
#         pass
#
#
# # proper type
# class MyTable(TableType[TblAndColumn]):
#     pass


ttest = np.dtype([
('kind', np.bool),
('abs_min', np.uint64),
('max', np.uint64),
('bits', np.uint8),
],)

# class ConstructionFrame(Generic[[
# ('kind', np.bool),
# ('abs_min', np.uint64),
# ('max', np.uint64),
# ('bits', np.uint8),
# ]]):
#
class TypeLookup2(LookupTable[[
    ('signed',   np.bool),
    ('abs_min',  np.uint64),
    ('max',      np.uint64),
    ('bits',     np.uint8),
    ('prev_max', np.int64),
]]):
    pass



def build_tlook():
    kind = {'u': False, 'i': True}
    sizes = [1, 2, 4, 8]
    types = np.array([
        np.dtype(f"{k}{s}")
        for k in kind
        for s in sizes
    ])

    rows = []
    last_max = {-1: -1, 1: -1}
    for t in types:
        s = 1 if t.kind == 'i' else -1
        rows.append((
            kind[t.kind],
            -np.iinfo(t).min,
            np.iinfo(t).max,
            np.iinfo(t).bits,
            last_max[s],
        ))
        last_max[s] = int(np.iinfo(t).max)

    return TypeLookup2.build(rows)


_LOOKUP_TBL = build_tlook()

val = 123

# `and`-trick form: no parens needed, `and` binds looser than comparisons.
smallest = (
    _LOOKUP_TBL.signed == False
    and _LOOKUP_TBL.max >= val
    and _LOOKUP_TBL.prev_max < val
)
print("and-trick ->", smallest)

rows = [t for t in _LOOKUP_TBL if t.signed == True]
print("signed rows ->", rows)
print("row 0 ->", _LOOKUP_TBL[0])
print("len ->", len(_LOOKUP_TBL))

# `&`-pipeline form (explicit, no bool magic): same result.
smallest_alt = (
    (((_LOOKUP_TBL.signed == False) & _LOOKUP_TBL.max) >= val)
    & _LOOKUP_TBL.prev_max
) < val
print("pipeline  ->", smallest_alt)

def _get_type(value: int, attr: str, signed: bool = False):
    # for dtype in _S_INT_TYPES if signed else _U_INT_TYPES:
    #     if value <= getattr(np.iinfo(dtype), attr):
    #         return dtype
    #
    # return exc_to_many_bit.do_raise()
    pass

def get_type_for_scalar(value: int, signed: bool = False):
    # w = np.where(_LOOKUP_TBL.lookup.max >= value)
    # return _LOOKUP_TBL(, signed)
    pass

def get_type_for_bit_count(bit_count: int, signed: bool = False):
    return _get_type(bit_count, "bits", signed)

def get_type_for_array(ary: np.ndarray | List[Any] | Iterable[Any], signed: bool = False):
    if isinstance(ary, np.ndarray):
        return _get_type(np.max(np.abs(ary) if signed else ary), "max", signed)
    else:
        if signed:
            return _get_type(max(-min(ary), max(ary)), "max", signed)
        else:
            return _get_type(max(ary), "max", signed)

@overload
def get_as_signed(a: np.dtype[Any]) -> np.dtype[Any]: ...

@overload
def get_as_signed(a: npt.ArrayLike) -> np.ndarray | np.dtype[Any]: ...


def get_as_signed(
    a: npt.ArrayLike | np.dtype[Any],
) -> np.ndarray | np.dtype[Any]:
    if isinstance(a, np.dtype):
        if a.kind == "i":
            return a
        if a.kind != "u":
            raise TypeError("Expected an integer dtype")

        return np.dtype(f"i{a.itemsize}")

    a = np.asarray(a)

    if a.dtype.kind == "i":
        return a
    if a.dtype.kind != "u":
        raise TypeError("Expected an integer array")

    return a.astype(np.dtype(f"i{a.dtype.itemsize}"))



@overload
def get_as_unsigned(a: np.dtype[Any]) -> np.dtype[Any]: ...

@overload
def get_as_unsigned(a: npt.ArrayLike) -> np.ndarray | np.dtype[Any]: ...


def get_as_unsigned(
    a: npt.ArrayLike | np.dtype[Any],
    fit: bool = False,
) -> np.ndarray | np.dtype[Any]:
    if isinstance(a, np.dtype):
        if a.kind == "u":
            return a
        if a.kind != "i":
            raise TypeError("Expected an integer dtype")

        return np.dtype(f"u{a.itemsize}")

    a = np.asarray(a)

    if a.dtype.kind == "u":
        return a
    if a.dtype.kind != "i":
        raise TypeError("Expected an integer array")

    return a.astype(np.dtype(f"u{a.dtype.itemsize}"))


def iter_bits(data):
    for byte in data:
        for i in range(8):
            yield (byte >> (7 - i)) & 1


