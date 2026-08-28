from dataclasses import dataclass
from typing import TypeVar, Generic, Type, List, Iterable, Any, NamedTuple, overload, Literal, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt


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


