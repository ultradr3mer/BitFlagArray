from typing import TypeVar, NamedTuple, Generic, Any
import numpy as np
import numpy.typing as npt


class NpColDef(NamedTuple):
    name: str
    type: npt.DTypeLike


# dtype-spec input form: [('kind', np.bool), ('abs_min', np.uint64), ...]
type FieldSpec = list[tuple[str, npt.DTypeLike]]


def to_col_defs(spec: FieldSpec) -> list[NpColDef]:
    """Convert a raw (name, dtype) field-spec into a list of NpColDef."""
    return [NpColDef(name, np.dtype(dt)) for name, dt in spec]


T_base = TypeVar('T_base', bound=Any)
T_row = TypeVar('T_row', bound=Any)


class AnotherTbl(Generic[T_base, T_row]):
    pass


def create_table_type(
    table_name: str,
    fields: list[NpColDef],
):
    dtype = np.dtype([(f.name, f.type) for f in fields])

    row_fields: list[tuple[str, type[Any]]] = [(n, np.dtype(t).type) for n, t in fields]
    row_type = NamedTuple(f"{table_name}Row", row_fields)

    table_fields: list[tuple[str, type[Any]]] = [
        (f.name, npt.NDArray[np.dtype(f.type)]) for f in fields
    ]
    table_fields.append(("row_type", row_type))
    tbl_type = NamedTuple(f"{table_name}Table", table_fields)

    def build(data: npt.ArrayLike) -> tbl_type:  # type: ignore[valid-type]
        ary = np.asarray(data, dtype=dtype)
        return tbl_type(
            **{f.name: ary[f.name] for f in fields},
            row_type=row_type,
        )

    def row(ary_: np.ndarray, idx: int) -> row_type:  # type: ignore[valid-type]
        rec = ary_[idx]
        return row_type(**{f.name: rec[f.name] for f in fields})

    tbl_type.build = staticmethod(build)
    tbl_type.row = staticmethod(row)
    tbl_type.dtype = staticmethod(lambda: dtype)
    tbl_type.row_type = row_type
    return tbl_type


_test_spec = [('kind', np.bool), ('abs_min', np.uint64), ('max', np.uint64), ('bits', np.uint8)]
_col_defs = to_col_defs(_test_spec)
print(_col_defs)

MyTypes = create_table_type("MyTypes", _col_defs)
print(MyTypes)
print("dtype:", MyTypes.dtype())

rows_data = [
    (True, 0,   255,     8),
    (False, 1,  65535,   16),
    (True, 2,  4294967295, 32),
]
tbl = MyTypes.build(rows_data)
print("table ->", tbl)
print("kind  ->", tbl.kind)
print("max   ->", tbl.max)
print("bits  ->", tbl.bits)

print("row 0 ->", MyTypes.row(tbl._asdict() and np.asarray(rows_data, dtype=MyTypes.dtype()), 0))
print("row 1 ->", MyTypes.row(np.asarray(rows_data, dtype=MyTypes.dtype()), 1))

for i in range(len(tbl.kind)):
    print(f"  row {i} ->", tbl.row_type(
        kind=tbl.kind[i],
        abs_min=tbl.abs_min[i],
        max=tbl.max[i],
        bits=tbl.bits[i],
    ))

print("pause")