from typing import NamedTuple, Any, get_type_hints, TYPE_CHECKING, TypeVar, Generic
import numpy as np
import numpy.typing as npt


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

    for r in MyTypes.rows(np.asarray([
        (True, 0, 255, 8), (False, 1, 65535, 16), (True, 2, 4294967295, 32),
    ], dtype=MyTypes.dtype)):
        print("  row ->", r)
