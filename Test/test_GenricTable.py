import numpy as np
import pytest
from typing import NamedTuple

from GenricTable import (
    NpColDef,
    to_col_defs,
    get_defs_from_rowtype,
    FieldSpec,
    Table,
    PlainTable,
    TableCreator,
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
    assert cols[3].type == np.dtype(np.uint8)


def test_to_col_defs_normalizes_dtype():
    cols = to_col_defs([('x', 'u8'), ('y', 'bool')])
    assert cols[0].type == np.dtype('u8')
    assert cols[1].type == np.dtype('bool')


# --------------------------------------------------------------------
# _col_defs_from_rowtype
# --------------------------------------------------------------------

def test_col_defs_from_rowtype():
    class MyRow(NamedTuple):
        kind: np.uint8
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8

    cols = get_defs_from_rowtype(MyRow)
    assert len(cols) == 4
    assert cols[0] == NpColDef('kind', np.dtype(np.uint8))
    assert cols[1] == NpColDef('abs_min', np.dtype(np.uint64))


# --------------------------------------------------------------------
# Table base / PlainTable
# --------------------------------------------------------------------

class RowType(NamedTuple):
    kind: np.uint8
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8


class MyPlain(PlainTable[RowType]):
    @classmethod
    def get_row_type(cls) -> type[RowType]:
        return RowType


def test_table_is_abstract():
    with pytest.raises(TypeError):
        Table("x", [], RowType)  # type: ignore[abstract]


def test_plain_table_build():
    tbl = MyPlain.build("test", [(1, 0, 255, 8), (2, 1, 65535, 16)])
    assert tbl.name == "test"
    assert tbl.row_type is RowType
    assert len(tbl.fields) == 4
    assert len(tbl) == 2
    assert tbl.dtype.names == ('kind', 'abs_min', 'max', 'bits')


def test_plain_table_columns_are_arrays():
    tbl = MyPlain.build("test", [(1, 0, 255, 8)])
    assert isinstance(tbl.kind, np.ndarray)
    assert tbl.kind.dtype == np.uint8
    assert isinstance(tbl.max, np.ndarray)


def test_plain_table_getitem():
    tbl = MyPlain.build("test", [(1, 0, 255, 8), (2, 1, 65535, 16)])
    row = tbl[0]
    assert isinstance(row, RowType)
    assert int(row.kind) == 1
    assert int(row.max) == 255


def test_plain_table_iter():
    tbl = MyPlain.build("test", [(1, 0, 255, 8), (2, 1, 65535, 16)])
    rows = list(tbl)
    assert len(rows) == 2
    assert int(rows[1].max) == 65535


def test_plain_table_rows_helper():
    tbl = MyPlain.build("test", [(1, 0, 255, 8)])
    rows = tbl.rows()
    assert len(rows) == 1
    assert isinstance(rows[0], RowType)


# --------------------------------------------------------------------
# TableCreator
# --------------------------------------------------------------------

def test_table_creator():
    creator = TableCreator[RowType, MyPlain](MyPlain)
    tbl = creator.build("via_creator", [(1, 0, 255, 8)])
    assert isinstance(tbl, MyPlain)
    assert tbl.name == "via_creator"
    assert tbl.row_type is RowType
    assert len(tbl) == 1
    assert int(tbl[0].max) == 255


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
