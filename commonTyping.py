from typing import TypeVar, Generic, Type, List, Iterable, Any, overload, Literal, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt

from GenricTable import row_type_from_fields
from QueryableTable import QTblSpecialCol, CCol


# ====================================================
#                   TYPE DETECTION
# ====================================================

class DRowFields:
    """Single shared row/table declaration.

    Each field is annotated ``CCol | scalar``: on the table it is a column
    (``CCol``), on a row it is a scalar. ``TypeTable`` inherits this class,
    so the fields exist statically on the table; ``DRow`` (the NamedTuple
    rows) is generated from it - fields are declared exactly once.
    """
    signed: CCol | np.bool_
    abs_min: CCol | np.uint64
    max: CCol | np.uint64
    bits: CCol | np.uint8


DRow = row_type_from_fields('DRow', DRowFields)


class TypeTable(QTblSpecialCol[DRow, Type[Any]], DRowFields):
    """Inherits the shared DRowFields declaration - nothing redeclared here."""

    def __init__(self, data: npt.ArrayLike, result_dict: npt.NDArray[Any]):
        super().__init__("IntegerTypeTable", data, result_dict, DRow)



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

if __name__ == "__main__":
    val = 123

    # `and`-trick form: no parens needed, `and` binds looser than comparisons.
    smallest = INTEGER_TYPES.get_first(
        INTEGER_TYPES.signed == False
        and INTEGER_TYPES.max >= val
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