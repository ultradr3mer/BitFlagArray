import operator
from dataclasses import dataclass
from typing import TypeVar, Generic, Any, NamedTuple, Literal, get_type_hints

import numpy as np
import numpy.typing as npt

from GenricTable import TableType, NpColDef, _col_defs_from_rowtype


#====================================================
#               QUERY INFRASTRUCTURE
#====================================================

Undefined = Literal

T_constr_item = TypeVar('T_constr_item', bound=Any)

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
        _pending_selections.append(self)
        return True


class ConstraintColumn(Generic[T_constr_item]):
    __slots__ = ("table", "column", "parent_indices")

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
#               QUERYABLE TABLE
#====================================================

class QueryableTableType(TableType):
    """TableType that builds QueryableTable instances with ConstraintColumn columns."""

    def build(self, data: npt.ArrayLike) -> "QueryableTable":
        ary = np.asarray(data, dtype=self._dtype)
        return QueryableTable(ary, self._fields, self._row_type)


class QueryableTable:
    """A built queryable table with ConstraintColumn columns.

    Created via ``QueryableTableType.build(data)`` — not instantiated directly.
    """

    def __init__(self, ary: np.ndarray, fields: list[NpColDef], row_type: type):
        self._ary = ary
        self._fields = fields
        self._row_type = row_type
        for f in fields:
            setattr(self, f.name, ConstraintColumn(ary, f.name))

    def __iter__(self):
        for rec in self._ary:
            yield self._row_type(**{f.name: rec[f.name] for f in self._fields})

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            rec = self._ary[key]
            return self._row_type(**{f.name: rec[f.name] for f in self._fields})
        return self._ary[key]

    def __len__(self):
        return len(self._ary)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._ary.dtype

    @property
    def row_type(self) -> type:
        return self._row_type

    def rows(self) -> list[Any]:
        return list(self)





if __name__ == "__main__":
    class TypeLookupRow(NamedTuple):
        signed: np.bool
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8
        prev_max: np.int64

    kind = {'u': False, 'i': True}
    sizes = [1, 2, 4, 8]
    types = [np.dtype(f"{k}{s}") for k in kind for s in sizes]
    rows = []
    last_max = {-1: -1, 1: -1}
    for t in types:
        s = 1 if t.kind == 'i' else -1
        rows.append((kind[t.kind], -np.iinfo(t).min, np.iinfo(t).max, np.iinfo(t).bits, last_max[s]))
        last_max[s] = int(np.iinfo(t).max)

    TypeLookup = create_queryable_table("TypeLookup", rowtype=TypeLookupRow)
    qtbl = TypeLookup.build(rows)

    print("dtype:", qtbl.dtype)
    print("len:", len(qtbl))
    print("row 0:", qtbl[0])

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
    print("signed rows:", signed_rows)
