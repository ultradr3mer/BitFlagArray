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
    ary: np.ndarray | None = None,
):
    row_fields: list[tuple[str, type[Any]]] = [(n, np.dtype(t).type) for n, t in fields]
    row_type = NamedTuple(f"{table_name}Row", row_fields)
    table_fields: list[tuple[str, type[Any]]] = [
        (n, npt.NDArray[np.dtype(t)]) for n, t in fields
    ]
    table_fields.append(("row_type", row_type))
    tbl_type = NamedTuple(f"{table_name}Table", table_fields)
    return tbl_type


_test_spec = [('kind', np.bool), ('abs_min', np.uint64), ('max', np.uint64), ('bits', np.uint8)]
_col_defs = to_col_defs(_test_spec)
print(_col_defs)
_test = create_table_type("MyTypes", _col_defs)
print(_test)

_test(kind=np.array((True)))
