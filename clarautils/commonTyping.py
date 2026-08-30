from typing import TypeVar, Generic, Type, List, Iterable, Any, overload, Literal, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt
from .GenericTable import TableFields
from .QueryableTable import QTblSpecialCol, CCol, Query, Undefined
from .common import ExcRaiser


# ====================================================
#                   TYPE DETECTION
# ====================================================

class DTableFields(TableFields):
    signed: CCol | np.bool_
    abs_min: CCol | np.uint64
    max: CCol | np.uint64
    bits: CCol | np.uint8


class TypeTable(QTblSpecialCol[DTableFields, Type[Any]], DTableFields):

    def __init__(self, data: npt.ArrayLike, result_dict: npt.NDArray[Any]):
        super().__init__("IntegerTypeTable", data, result_dict)



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

    return TypeTable(data=data, result_dict=types)


INTEGER_TYPES = build_type_tbl()


# ====================================================
#               TYPE LOOKUP
# ====================================================

exc_too_big: ExcRaiser[np.dtype] = ExcRaiser(Exception, "value to big")
exc_too_many_bits: ExcRaiser[np.dtype] = ExcRaiser(Exception, "to many bits requested")


def _first_or_raise(query: Query, raiser: ExcRaiser[np.dtype]) -> np.dtype:
    """Evaluate the query against INTEGER_TYPES; first result or raise."""
    found = INTEGER_TYPES.get_all(query)
    if len(found) == 0:
        return raiser.do_raise()
    return found[0]


def _get_type(value: int, attr: str, signed: "bool | Undefined" = False) -> np.dtype:
    """Smallest dtype of the family where row[attr] >= value."""
    type_tbl = INTEGER_TYPES
    return _first_or_raise(
        (type_tbl.signed == signed) & (getattr(type_tbl, attr) >= value),
        exc_too_many_bits if attr == "bits" else exc_too_big,
    )


def _get_type_for_bounds(low: int, high: int, signed: "bool | Undefined" = False) -> np.dtype:
    """Smallest dtype of the family that can hold every value in [low, high].

    ``abs_min`` bounds the negative side (so -128 fits int8), ``max`` the
    positive one. ``low=0`` disables the negative bound (legacy unsigned).
    """
    type_tbl = INTEGER_TYPES
    return _first_or_raise(
        (type_tbl.signed == signed)
        & (type_tbl.abs_min >= -low)
        & (type_tbl.max >= high),
        exc_too_big,
    )


def get_type_for_scalar(value: int, signed: "bool | Undefined" = False) -> np.dtype:
    """Smallest dtype that can hold the scalar.

    Legacy unsigned lookup checks ``max`` only; signed (or ``Undefined`` =
    either family) also honors ``abs_min`` for negative values.
    """
    return _get_type_for_bounds(value if signed else 0, value, signed)


def get_type_for_bit_count(bit_count: int, signed: "bool | Undefined" = False) -> np.dtype:
    return _get_type(bit_count, "bits", signed)


def get_type_for_array(ary: np.ndarray | List[Any] | Iterable[Any], signed: "bool | Undefined" = False) -> np.dtype:
    """Smallest dtype that can hold the whole array/list/iterable."""
    if isinstance(ary, np.ndarray):
        low = int(np.min(ary)) if signed else 0
        high = int(np.max(ary))
    else:
        ary = list(ary)
        low = min(ary) if signed else 0
        high = max(ary)
    return _get_type_for_bounds(low, high, signed)


@overload
def get_as_signed(a: np.dtype[Any]) -> np.dtype[Any]: ...


@overload
def get_as_signed(a: npt.ArrayLike, fit: bool = False) -> np.ndarray | np.dtype[Any]: ...


def get_as_signed(
        a: npt.ArrayLike | np.dtype[Any],
        fit: bool = False,
) -> np.ndarray | np.dtype[Any]:
    if isinstance(a, np.dtype):
        if a.kind == "i":
            return a
        if a.kind != "u":
            raise TypeError("Expected an integer dtype")

        return np.dtype(f"i{a.itemsize}")

    a = np.asarray(a)

    if a.dtype.kind not in ("i", "u"):
        raise TypeError("Expected an integer array")

    if fit:
        target = _get_type_for_bounds(int(np.min(a)), int(np.max(a)), True)
        return a if a.dtype == target else a.astype(target)

    if a.dtype.kind == "i":
        return a
    return a.astype(np.dtype(f"i{a.dtype.itemsize}"))


@overload
def get_as_unsigned(a: np.dtype[Any]) -> np.dtype[Any]: ...


@overload
def get_as_unsigned(a: npt.ArrayLike, fit: bool = False) -> np.ndarray | np.dtype[Any]: ...


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

    if a.dtype.kind not in ("i", "u"):
        raise TypeError("Expected an integer array")

    if fit:
        target = _get_type_for_bounds(int(np.min(a)), int(np.max(a)), False)
        return a if a.dtype == target else a.astype(target)

    if a.dtype.kind == "u":
        return a
    return a.astype(np.dtype(f"u{a.dtype.itemsize}"))


# ====================================================
#               DEMO
# ====================================================

if __name__ == "__main__":
    print("item  ->", INTEGER_TYPES[0])
    print("range ->", INTEGER_TYPES[4:8])
    print()

    print("scalar 255 unsigned ->", get_type_for_scalar(255))
    print("scalar 256 unsigned ->", get_type_for_scalar(256))
    print("scalar -128 signed  ->", get_type_for_scalar(-128, signed=True))
    print("scalar -129 signed  ->", get_type_for_scalar(-129, signed=True))
    print("scalar -128 either  ->", get_type_for_scalar(-128, Undefined))
    print("bits 9 unsigned     ->", get_type_for_bit_count(9))
    print("bits 9 signed       ->", get_type_for_bit_count(9, signed=True))
    print("array               ->", get_type_for_array([1, 2, 300]))
    print("array signed        ->", get_type_for_array([-128, 5], signed=True))
    print("array signed nd     ->", get_type_for_array(np.array([-1000, 1000]), signed=True))

    try:
        get_type_for_bit_count(65)
    except Exception as e:
        print("bits 65 ->", e)

    try:
        get_type_for_scalar(2 ** 64)
    except Exception as e:
        print("scalar to big ->", e)