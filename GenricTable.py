from typing import NamedTuple, Any, get_type_hints
import numpy as np
import numpy.typing as npt


#====================================================
#               COLUMN DEFINITIONS
#====================================================

class NpColDef(NamedTuple):
    name: str
    type: np.dtype[Any]


type FieldSpec = list[tuple[str, npt.DTypeLike]]


def to_col_defs(spec: FieldSpec) -> list[NpColDef]:
    """Convert a raw (name, dtype) field-spec into a list of NpColDef."""
    return [NpColDef(name, np.dtype(dt)) for name, dt in spec]


def _col_defs_from_rowtype(rowtype: type) -> list[NpColDef]:
    """Derive column definitions from a NamedTuple row type's annotations."""
    hints = get_type_hints(rowtype)
    return [NpColDef(name, np.dtype(hints[name])) for name in rowtype._fields]


if __name__ == "__main__":
    spec: FieldSpec = [
        ('kind', np.bool), ('abs_min', np.uint64),
        ('max', np.uint64), ('bits', np.uint8),
    ]
    print(to_col_defs(spec))

    class RowType(NamedTuple):
        kind: np.uint8
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8

    print(_col_defs_from_rowtype(RowType))
