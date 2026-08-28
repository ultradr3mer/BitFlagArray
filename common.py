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

# ====================================================
#                   QUERYABLE TABLE
# ====================================================

T_constr_item = TypeVar('T_constr_item', bound=Any)

Undefined = Literal

# Stack for the `and`-trick: `bool(selection)` (called by `and`) pushes the
# selection here; the next comparison on a column pops it and applies it as a
# restriction. Enables: `selection and column >= value`.
_pending_selections: list["ConstraintSelection"] = []


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
        # Called by `and`. Stash self so the following column comparison can
        # pick it up as a restriction. Returning True makes `and` evaluate and
        # return its right-hand side.
        _pending_selections.append(self)
        return True

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
        # `and`-trick: if a pending selection was stashed by `__bool__`,
        # restrict this column to it before comparing.
        col = self
        if _pending_selections:
            col = col[_pending_selections.pop()]

        if value is Undefined:
            mask = np.ones(len(col.column), dtype=bool)
        else:
            mask = op(col.column, value)

        local = np.flatnonzero(mask)
        return ConstraintSelection(col._resolve(local))

    def __eq__(self, value: object) -> ConstraintSelection:  # type: ignore[override]
        return self._compare(value, operator.eq)

    def __ne__(self, value: object) -> ConstraintSelection:  # type: ignore[override]
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


CCol = ConstraintColumn

# dtype spec: list of (name, dtype) tuples.
TFieldSpec = list[tuple[str, Any]]

# Module-level cache for generated row NamedTuples per LookupTable subclass.
_row_types: dict[type, type] = {}


class LookupTable:
    """Base for dtype-first lookup tables.

    Subclass via::

        class MyTable(LookupTable[[
            ('signed', np.bool),
            ('max',    np.uint64),
        ]]):
            pass
    """

    _fields_spec: TFieldSpec = []
    _array: npt.NDArray[Any] | None

    def __init__(self):
        self._array = None

    if TYPE_CHECKING:
        def __getattr__(self, name: str) -> CCol: ...

    # ---- dtype-first machinery ----

    def __class_getitem__(cls, spec: TFieldSpec):
        # Validate up front so typos surface at class-creation time.
        if not isinstance(spec, (list, tuple)):
            raise TypeError("LookupTable[...] expects a list of (name, dtype) tuples")
        norm = []
        for entry in spec:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                raise TypeError(f"field spec entry must be (name, dtype), got {entry!r}")
            name, dt = entry
            if not isinstance(name, str):
                raise TypeError(f"field name must be str, got {name!r}")
            norm.append((name, np.dtype(dt)))
        return type(
            f"LookupTableSpec",
            (LookupTable,),
            {"_fields_spec": norm},
        )

    @classmethod
    def _field_names(cls) -> list[str]:
        return [name for name, _ in cls._fields_spec]

    @classmethod
    def dtype(cls) -> np.dtype:
        return np.dtype(list(cls._fields_spec))

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

    @classmethod
    def _row_type(cls) -> type:
        cached = _row_types.get(cls)
        if cached is not None:
            return cached
        Row = NamedTuple(
            f"{cls.__name__}Row",
            [(name, Any) for name in cls._field_names()],
        )
        _row_types[cls] = Row
        return Row

    def __iter__(self):
        Row = type(self)._row_type()
        names = type(self)._field_names()
        for rec in self._array:
            yield Row(*[rec[n] for n in names])

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            Row = type(self)._row_type()
            names = type(self)._field_names()
            rec = self._array[key]
            return Row(*[rec[n] for n in names])
        return self._array[key]

    def __len__(self):
        return len(self._array)




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


