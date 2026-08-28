from typing import NamedTuple, Any, cast
import numpy as np
import numpy.typing as npt


class NpColDef(NamedTuple):
    name: str
    type: np.dtype[Any]


type FieldSpec = list[tuple[str, npt.DTypeLike]]


def to_col_defs(spec: FieldSpec) -> list[NpColDef]:
    """Convert a raw (name, dtype) field-spec into a list of NpColDef."""
    return [NpColDef(name, np.dtype(dt)) for name, dt in spec]


class TableType:
    """Factory + metadata for a dynamically-built lookup table.

    Holds the numpy structured dtype, the row NamedTuple, and the table
    NamedTuple, plus ``build``/``row`` helpers.
    """

    __slots__ = ("_dtype", "_row_type", "_tbl_type", "_fields")

    def __init__(self, name: str, fields: list[NpColDef]):
        self._fields = fields
        self._dtype = np.dtype([(f.name, f.type) for f in fields])

        row_fields: list[tuple[str, type[Any]]] = [
            (f.name, f.type.type) for f in fields
        ]
        self._row_type = NamedTuple(f"{name}Row", row_fields)

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


def create_table_type(name: str, fields: list[NpColDef]) -> TableType:
    return TableType(name, fields)


_test_spec: FieldSpec = [
    ('kind', np.bool), ('abs_min', np.uint64),
    ('max', np.uint64), ('bits', np.uint8),
]
_col_defs = to_col_defs(_test_spec)
print(_col_defs)

MyTypes = create_table_type("MyTypes", _col_defs)
print(MyTypes)
print("dtype:", MyTypes.dtype)

rows_data = [
    (True, 0,   255,       8),
    (False, 1,  65535,     16),
    (True, 2,  4294967295, 32),
]
tbl = MyTypes.build(rows_data)
print("table ->", tbl)
print("kind  ->", tbl.kind)
print("max   ->", tbl.max)
print("bits  ->", tbl.bits)

ary = np.asarray(rows_data, dtype=MyTypes.dtype)
print("row 0 ->", MyTypes.row(ary, 0))
print("row 1 ->", MyTypes.row(ary, 1))

for r in MyTypes.rows(ary):
    print("  row ->", r)

for i in range(len(tbl.kind)):
    print(f"  manual row {i} ->", MyTypes.row_type(
        kind=tbl.kind[i],
        abs_min=tbl.abs_min[i],
        max=tbl.max[i],
        bits=tbl.bits[i],
    ))
