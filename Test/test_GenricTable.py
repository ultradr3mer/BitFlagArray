import numpy as np
import pytest

from GenricTable import (
    NpColDef,
    to_col_defs,
    _col_defs_from_rowtype,
    FieldSpec,
)


# --------------------------------------------------------------------
# NpColDef / to_col_defs
# --------------------------------------------------------------------

def test_to_col_defs_basic():
    spec: FieldSpec = [
        ('kind', np.bool), ('abs_min', np.uint64),
        ('max', np.uint64), ('bits', np.uint8),
    ]
    cols = to_col_defs(spec)
    assert len(cols) == 4
    assert all(isinstance(c, NpColDef) for c in cols)
    assert cols[0].name == 'kind'
    assert cols[0].type == np.dtype(np.bool)
    assert cols[1].type == np.dtype(np.uint64)
    assert cols[3].type == np.dtype(np.uint8)


def test_to_col_defs_normalizes_dtype():
    cols = to_col_defs([('x', 'u8'), ('y', 'bool')])
    assert cols[0].type == np.dtype('u8')
    assert cols[1].type == np.dtype('bool')


# --------------------------------------------------------------------
# _col_defs_from_rowtype
# --------------------------------------------------------------------

class _RowType(np.typing.NDArray.__class__ if False else type):  # placeholder
    pass


class RowType(pytest.NamedTuple if False else __import__('typing').NamedTuple):
    kind: np.uint8
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8


def test_col_defs_from_rowtype():
    from typing import NamedTuple

    class MyRow(NamedTuple):
        kind: np.uint8
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8

    cols = _col_defs_from_rowtype(MyRow)
    assert len(cols) == 4
    assert cols[0] == NpColDef('kind', np.dtype(np.uint8))
    assert cols[1] == NpColDef('abs_min', np.dtype(np.uint64))
    assert cols[2] == NpColDef('max', np.dtype(np.uint64))
    assert cols[3] == NpColDef('bits', np.dtype(np.uint8))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
