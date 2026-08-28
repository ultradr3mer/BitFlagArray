from argparse import ArgumentError
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar, Generic, Type, List, Iterable, Any, NamedTuple, overload, Literal, get_type_hints, \
    get_origin, get_args

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

# ====================================================
#                   QUERYABLE TABLE
# ====================================================

T_constr_item = TypeVar('T_constr_item', bound=Any)

Undefined = Literal

@dataclass(frozen=True)
class ConstraintSelection:
    indices: npt.NDArray[np.intp]

    def __and__(self, other):
        if isinstance(other, ConstraintSelection):
            return ConstraintSelection(
                np.intersect1d(self.indices, other.indices)
            )
        if isinstance(other, ConstraintColumn):
            return other[self]
        return NotImplemented

    def __bool__(self) -> bool:
        raise TypeError(
            "ConstraintSelection cannot be used with 'and'; "
            "use '&' or an explicit filtering method."
        )

import operator

class ConstraintColumn(Generic[T_constr_item]):
    def __init__(
        self,
        data: npt.NDArray[T_constr_item],
        selector: str | None = None,
        *,
        _column: npt.NDArray[T_constr_item] | None = None,
        _parent_indices: npt.NDArray[np.intp] | None = None,
    ):
        if _column is not None:
            self.column = _column
            self.table = data
            self.parent_indices = _parent_indices
            return

        if selector is None:
            self.column = data
        else:
            self.column = data[selector]
        self.table = data
        self.parent_indices = None

    def _resolve(self, local_indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        if self.parent_indices is None:
            return local_indices
        return self.parent_indices[local_indices]

    def __getitem__(self, key):
        if isinstance(key, ConstraintSelection):
            return ConstraintColumn(
                self.table,
                _column=self.column[key.indices],
                _parent_indices=key.indices,
            )
        return self.column[key]

    def _compare(self, value: object, op) -> ConstraintSelection:
        if value is Undefined:
            mask = np.ones(len(self.column), dtype=bool)
        else:
            mask = op(self.column, value)

        local = np.flatnonzero(mask)
        return ConstraintSelection(self._resolve(local))

    def __eq__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.eq)

    def __ne__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.ne)

    def __lt__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.lt)

    def __le__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.le)

    def __gt__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.gt)

    def __ge__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.ge)

    def __hash__(self) -> int:
        return id(self)


T_tbl_impl = TypeVar('T_tbl_impl', bound="LookupTable")

CCol = ConstraintColumn

class LookupTable(Generic[T_tbl_impl]):
    @classmethod
    def _field_names(cls) -> list[str]:
        hints = get_type_hints(cls)
        return [
            name for name, ann in hints.items()
            if get_origin(ann) is CCol
        ]

    @classmethod
    def dtype(cls) -> np.dtype:
        hints = get_type_hints(cls)

        fields = []

        for name in cls._field_names():
            annotation = hints[name]

            # Bei CCol[np.uint64] ist:
            # origin == CCol
            # args == (np.uint64,)
            if get_origin(annotation) is not CCol:
                raise TypeError(
                    f"{cls.__name__}.{name} must be annotated as CCol[...]"
                )

            (item_type,) = get_args(annotation)
            fields.append((name, np.dtype(item_type)))

        return np.dtype(fields)

    @classmethod
    def build(cls, array: npt.ArrayLike) -> "LookupTable":
        dtype = cls.dtype()
        ary = np.asarray(array, dtype=dtype)

        obj = cls()
        for name in cls._field_names():
            setattr(obj, name, CCol(ary, name))
        obj._array = ary
        return obj

    @classmethod
    def unpack(cls, array: npt.ArrayLike):
        """Gibt die einzelnen rohen NumPy-Spalten zurück."""
        ary = np.asarray(array, dtype=cls.dtype())
        return tuple(ary[name] for name in cls._field_names())



#====================================================
#                   TYPE DETECTION
#====================================================



class TypeLookup2(LookupTable):
    signed: CCol[np.bool]
    abs_min: CCol[np.uint64]
    max: CCol[np.uint64]
    bits: CCol[np.uint64]



def build_tlook():
    kind = {'u': False, 'i': True}
    sizes = [1, 2, 4, 8]
    types = np.array([
        np.dtype(f"{k}{s}")
        for k in kind
        for s in sizes
    ])
    return TypeLookup2.build(
            [
                (
                    kind[t.kind],
                    -np.iinfo(t).min,
                    np.iinfo(t).max,
                    np.iinfo(t).bits,
                )
                for t in types
            ]
        )


_LOOKUP_TBL = build_tlook()
_sel = _LOOKUP_TBL.signed == False
print(_sel)
print((_LOOKUP_TBL.signed == None & _LOOKUP_TBL.max) > 123)
print((_sel & _LOOKUP_TBL.max) <= 123)
print((_sel & _LOOKUP_TBL.max) < 123)

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


