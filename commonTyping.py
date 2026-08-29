from typing import TypeVar, Generic, Type, List, Iterable, Any, NamedTuple, overload, Literal, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

from QueryableTable import QTblSpecialCol


# ====================================================
#                   TYPE DETECTION
# ====================================================

class DRow(NamedTuple):
    signed: np.bool
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8


# class NpTblWithResult:
#     ...
#
#     def create_columns(self) -> List[Any]: ...


class TypeTable(QTblSpecialCol[DRow, Type[Any]]):
    def __init__(self, data: npt.ArrayLike, result_dict: npt.NDArray[Any]):
        super().__init__("IntegerTypeTable", data, result_dict, DRow)
        self.kind, self.abs_min, self.max, self.bits = self.create_columns()


def build_type_tbl():
    kind = {'u': False, 'i': True}
    sizes = [1, 2, 4, 8]
    types = np.array([
        np.dtype(f"{k}{s}")
        for k in kind
        for s in sizes
    ])

    data = []
    for t in types:
        data.append((
            kind[t.kind],
            -np.iinfo(t).min,
            np.iinfo(t).max,
            np.iinfo(t).bits,
        ))

    from GenricTable import NpTblWithResult
    return TypeTable(data=data, result_dict=types)


INTEGER_TYPES = build_type_tbl()

val = 123

# `and`-trick form: no parens needed, `and` binds looser than comparisons.
smallest = INTEGER_TYPES.get_first(
    INTEGER_TYPES.kind == False
    & INTEGER_TYPES.max >= val
)
print("and-trick ->", smallest)

rows = [t for t in INTEGER_TYPES if t.signed == True]
print("signed rows ->", rows)
print("row 0 ->", INTEGER_TYPES[0])
print("len ->", len(INTEGER_TYPES))

# `&`-pipeline form (explicit, no bool magic): same result.
smallest_alt = (
                       (((INTEGER_TYPES.signed == False) & INTEGER_TYPES.max) >= val)
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