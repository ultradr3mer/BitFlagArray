from typing import NamedTuple, Any, get_type_hints, TYPE_CHECKING, Literal
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from common import ConstraintColumn


class NpColDef(NamedTuple):
    name: str
    type: np.dtype[Any]


type FieldSpec = list[tuple[str, npt.DTypeLike]]


def to_col_defs(spec: FieldSpec) -> list[NpColDef]:
    """Convert a raw (name, dtype) field-spec into a list of NpColDef."""
    return [NpColDef(name, np.dtype(dt)) for name, dt in spec]


def _col_defs_from_rowtype(rowtype: type) -> list[NpColDef]:
    hints = get_type_hints(rowtype)
    return [NpColDef(name, np.dtype(hints[name])) for name in rowtype._fields]


class TableType:
    """Factory + metadata for a dynamically-built lookup table.

    Holds the numpy structured dtype, the row NamedTuple, and the table
    NamedTuple, plus ``build``/``row`` helpers.
    """

    __slots__ = ("_dtype", "_row_type", "_tbl_type", "_fields")

    def __init__(self, name: str, fields: list[NpColDef], rowtype: type | None = None):
        self._fields = fields
        self._dtype = np.dtype([(f.name, f.type) for f in fields])

        self._row_type = rowtype if rowtype is not None else NamedTuple(
            f"{name}Row",
            [(f.name, f.type.type) for f in fields],
        )

        table_fields: list[tuple[str, type[Any]]] = [
            (f.name, npt.NDArray[Any]) for f in fields
        ]
        table_fields.append(("row_type", type(self._row_type)))
        self._tbl_type = NamedTuple(f"{name}Table", table_fields)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def row_type(self) -> type:
        return self._row_type

    @property
    def tbl_type(self) -> type:
        return self._tbl_type

    def build(self, data: npt.ArrayLike) -> Any:
        ary = np.asarray(data, dtype=self._dtype)
        return self._tbl_type(
            **{f.name: ary[f.name] for f in self._fields},
            row_type=self._row_type,
        )

    def row(self, ary: np.ndarray[Any, Any], idx: int) -> Any:
        rec = ary[idx]
        return self._row_type(**{f.name: rec[f.name] for f in self._fields})

    def rows(self, ary: np.ndarray[Any, Any]) -> list[Any]:
        return [self.row(ary, i) for i in range(len(ary))]


def create_table_type(
    name: str,
    fields: list[NpColDef] | None = None,
    *,
    rowtype: type | None = None,
) -> TableType:
    if rowtype is not None:
        fields = _col_defs_from_rowtype(rowtype)
    elif fields is None:
        raise TypeError("create_table_type needs either `fields` or `rowtype`")
    return TableType(name, fields, rowtype)


#====================================================
#               QUERYABLE TABLE
#====================================================

import operator
from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintSelection:
    indices: npt.NDArray[np.intp]

    def __and__(self, other):
        if isinstance(other, ConstraintSelection):
            return ConstraintSelection(np.intersect1d(self.indices, other.indices))
        if isinstance(other, ConstraintColumn):
            return other[self]
        return NotImplemented

    def __bool__(self) -> bool:
        _pending_selections.append(self)
        return True


_pending_selections: list[ConstraintSelection] = []


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


Undefined = Literal


class QueryableTableType(TableType):
    """TableType that builds tables with ConstraintColumn columns."""

    def build(self, data: npt.ArrayLike) -> "QueryableTable":
        ary = np.asarray(data, dtype=self._dtype)
        return QueryableTable(ary, self._fields, self._row_type)

    def row(self, ary: np.ndarray, idx: int) -> Any:
        rec = ary[idx]
        return self._row_type(**{f.name: rec[f.name] for f in self._fields})

    def rows(self, ary: np.ndarray) -> list[Any]:
        return [self.row(ary, i) for i in range(len(ary))]


class QueryableTable:
    """A built queryable table with ConstraintColumn columns."""

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


def create_queryable_table(
    name: str,
    fields: list[NpColDef] | None = None,
    *,
    rowtype: type | None = None,
) -> QueryableTableType:
    if rowtype is not None:
        fields = _col_defs_from_rowtype(rowtype)
    elif fields is None:
        raise TypeError("create_queryable_table needs either `fields` or `rowtype`")
    return QueryableTableType(name, fields, rowtype)


class RowType(NamedTuple):
    kind: np.uint8
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8


class TypeLookupRow(NamedTuple):
    signed: np.bool
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8
    prev_max: np.int64


if __name__ == "__main__":
    Other = create_table_type(name="Other", rowtype=RowType)
    print("Other dtype:", Other.dtype)
    _tbl = Other.build([(1, 0, 255, 8), (2, 1, 65535, 16)])
    print("Other table ->", _tbl)

    _ary = np.asarray([(1, 0, 255, 8)], dtype=Other.dtype)
    print("Other row 0 ->", Other.row(_ary, 0))

    _test_spec: FieldSpec = [
        ('kind', np.bool), ('abs_min', np.uint64),
        ('max', np.uint64), ('bits', np.uint8),
    ]
    MyTypes = create_table_type("MyTypes", to_col_defs(_test_spec))
    tbl = MyTypes.build([
        (True, 0, 255, 8), (False, 1, 65535, 16), (True, 2, 4294967295, 32),
    ])
    print("MyTypes table ->", tbl)

    # ---- QueryableTable demo ----
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

    print("\n--- QueryableTable ---")
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
