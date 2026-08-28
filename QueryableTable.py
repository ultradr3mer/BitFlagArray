import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, Literal, get_type_hints, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from GenricTable import NpColDef, to_col_defs, _col_defs_from_rowtype, FieldSpec


#====================================================
#               DTYPE HELPERS
#====================================================

def _dtype_from_rowtype(rowtype: type) -> np.dtype[Any]:
    return np.dtype([(c.name, c.type) for c in _col_defs_from_rowtype(rowtype)])


#====================================================
#               QUERY INFRASTRUCTURE
#====================================================

Undefined = Literal

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
        _pending_selections.append(self)
        return True


class ConstraintColumn:
    __slots__ = ("table", "column", "parent_indices")

    def __init__(
        self,
        data: np.ndarray,
        selector: str | None = None,
        *,
        _column: np.ndarray | None = None,
        _parent_indices: npt.NDArray[np.intp] | None = None,
    ):
        if _column is not None:
            self.column = _column
            self.table = data
            self.parent_indices = _parent_indices
            return
        self.column = data if selector is None else data[selector]
        self.table = data
        self.parent_indices = None

    def _resolve(self, local_indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        return local_indices if self.parent_indices is None else self.parent_indices[local_indices]

    def __getitem__(self, key):
        if isinstance(key, ConstraintSelection):
            return ConstraintColumn(
                self.table,
                _column=self.column[key.indices],
                _parent_indices=key.indices,
            )
        return self.column[key]

    def _compare(self, value: object, op) -> ConstraintSelection:
        col = self
        if _pending_selections:
            col = col[_pending_selections.pop()]
        if value is Undefined:
            mask = np.ones(len(col.column), dtype=bool)
        else:
            mask = op(col.column, value)
        return ConstraintSelection(col._resolve(np.flatnonzero(mask)))

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


#====================================================
#               TABLE HIERARCHY
#====================================================

class Table[TRow](ABC):
    """Base for typed lookup tables backed by a numpy structured array.

    Subclass and override ``get_row_type()`` to bind a row NamedTuple::

        class UserTable(Table[UserRow]):
            @classmethod
            def get_row_type(cls) -> type[UserRow]:
                return UserRow

    Then build from data::

        tbl = UserTable.build(data)
    """

    def __init__(self, ary: np.ndarray) -> None:
        self._ary = ary
        self._init_columns()

    @classmethod
    @abstractmethod
    def get_row_type(cls) -> type[TRow]:
        """Return the row NamedTuple type for this table."""

    @abstractmethod
    def _init_columns(self) -> None:
        """Create column attributes from ``self._ary``."""

    @classmethod
    def build(cls, data: npt.ArrayLike) -> "Table[TRow]":
        """Creator: derive dtype from row type, convert data, instantiate."""
        row_type = cls.get_row_type()
        dtype = _dtype_from_rowtype(row_type)
        ary = np.asarray(data, dtype=dtype)
        return cls(ary)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._ary.dtype

    @property
    def row_type(self) -> type:
        return type(self).get_row_type()

    @property
    def _field_names(self) -> list[str]:
        return list(type(self).get_row_type()._fields)

    def __iter__(self):
        row_type = type(self).get_row_type()
        names = row_type._fields
        for rec in self._ary:
            yield row_type(**{n: rec[n] for n in names})

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            row_type = type(self).get_row_type()
            names = row_type._fields
            rec = self._ary[key]
            return row_type(**{n: rec[n] for n in names})
        return self._ary[key]

    def __len__(self):
        return len(self._ary)

    def rows(self) -> list[Any]:
        return list(self)


class PlainTable[TRow](Table[TRow]):
    """Table with plain numpy array columns."""

    def _init_columns(self) -> None:
        for name in self._field_names:
            setattr(self, name, self._ary[name])


class QueryableTable[TRow](Table[TRow]):
    """Table with ConstraintColumn columns for querying."""

    def _init_columns(self) -> None:
        for name in self._field_names:
            setattr(self, name, ConstraintColumn(self._ary, name))


#====================================================
#               DYNAMIC FACTORY
#====================================================

def create_table(
    name: str,
    row_type: type,
    *,
    queryable: bool = True,
) -> type[Table[Any]]:
    """Dynamically create a Table subclass for the given row type.

    Usage::

        UserTable = create_table("UserTable", UserRow)
        tbl = UserTable.build(data)
    """
    base = QueryableTable if queryable else PlainTable
    return type(name, (base,), {
        "get_row_type": classmethod(lambda cls: row_type),
    })


#====================================================
#               DEMO
#====================================================

if __name__ == "__main__":
    class TypeLookupRow(NamedTuple):
        signed: np.bool
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8
        prev_max: np.int64

    # --- via subclass ---
    class TypeLookup(QueryableTable[TypeLookupRow]):
        @classmethod
        def get_row_type(cls) -> type[TypeLookupRow]:
            return TypeLookupRow

    # --- via factory ---
    TypeLookupDyn = create_table("TypeLookupDyn", TypeLookupRow, queryable=True)

    kind = {'u': False, 'i': True}
    sizes = [1, 2, 4, 8]
    types = [np.dtype(f"{k}{s}") for k in kind for s in sizes]
    rows = []
    last_max = {-1: -1, 1: -1}
    for t in types:
        s = 1 if t.kind == 'i' else -1
        rows.append((kind[t.kind], -np.iinfo(t).min, np.iinfo(t).max, np.iinfo(t).bits, last_max[s]))
        last_max[s] = int(np.iinfo(t).max)

    qtbl = TypeLookup.build(rows)
    print("subclass ->", qtbl.dtype, "len:", len(qtbl))
    print("row 0:", qtbl[0])

    qtbl2 = TypeLookupDyn.build(rows)
    print("factory  ->", qtbl2.dtype, "len:", len(qtbl2))

    sel = qtbl.signed == False
    print("signed == False ->", sel)

    val = 123
    smallest = (
        qtbl.signed == False
        and qtbl.max >= val
        and qtbl.prev_max < val
    )
    print("smallest unsigned for 123 ->", smallest)

    smallest_alt = (
        (((qtbl.signed == False) & qtbl.max) >= val) & qtbl.prev_max
    ) < val
    print("pipeline ->", smallest_alt)

    signed_rows = [t for t in qtbl if t.signed == True]
    print("signed rows:", len(signed_rows))

    # --- plain table ---
    class PlainLookup(PlainTable[TypeLookupRow]):
        @classmethod
        def get_row_type(cls) -> type[TypeLookupRow]:
            return TypeLookupRow

    ptbl = PlainLookup.build(rows)
    print("plain ->", type(ptbl.max), ptbl.max)
