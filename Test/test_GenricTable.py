import numpy as np
import pytest
from typing import NamedTuple

from GenricTable import (
    NpColDef,
    get_defs_from_rowtype,
    dtype_from_fields,
    NPColAdapter,
    NPContainerCreator,
    Table,
)


# --------------------------------------------------------------------
# NpColDef / get_defs_from_rowtype / dtype_from_fields
# --------------------------------------------------------------------

class RowType(NamedTuple):
    kind: np.uint8
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8


def test_col_defs_from_rowtype():
    cols = get_defs_from_rowtype(RowType)
    assert len(cols) == 4
    assert cols[0] == NpColDef('kind', np.dtype(np.uint8))
    assert cols[1] == NpColDef('abs_min', np.dtype(np.uint64))
    assert cols[2] == NpColDef('max', np.dtype(np.uint64))
    assert cols[3] == NpColDef('bits', np.dtype(np.uint8))


def test_np_col_def_repr():
    col = NpColDef('kind', np.dtype(np.uint8))
    assert 'kind' in repr(col)
    assert 'uint8' in repr(col)


def test_dtype_from_fields():
    fields = get_defs_from_rowtype(RowType)
    dt = dtype_from_fields(fields)
    assert dt.names == ('kind', 'abs_min', 'max', 'bits')
    assert dt['kind'] == np.dtype(np.uint8)
    assert dt['max'] == np.dtype(np.uint64)


# --------------------------------------------------------------------
# Table construction
# --------------------------------------------------------------------

@pytest.fixture
def tbl():
    return Table(name="test",
                 data=[(1, 0, 255, 8), (2, 1, 65535, 16)],
                 row_type=RowType,
                 col_a_cre=NPContainerCreator())


def test_table_build(tbl):
    assert tbl.name == "test"
    assert tbl.row_type is RowType
    assert len(tbl.fields) == 4
    assert len(tbl) == 2
    assert tbl.dtype.names == ('kind', 'abs_min', 'max', 'bits')


def test_table_empty_data():
    tbl = Table(name="empty",
                data=[],
                row_type=RowType,
                col_a_cre=NPContainerCreator())
    assert len(tbl) == 0
    assert list(tbl) == []


def test_table_columns_are_arrays(tbl):
    assert isinstance(tbl.kind, np.ndarray)
    assert tbl.kind.dtype == np.uint8
    assert isinstance(tbl.max, np.ndarray)
    assert tbl.max.dtype == np.uint64
    assert tbl.max.tolist() == [255, 65535]


def test_table_adapters(tbl):
    assert set(tbl.adapter) == {'kind', 'abs_min', 'max', 'bits'}
    for name, adapter in tbl.adapter.items():
        assert isinstance(adapter, NPColAdapter)
        assert adapter.name == name
        assert adapter.parent is tbl


def test_table_field_names(tbl):
    assert tbl._field_names == ['kind', 'abs_min', 'max', 'bits']


def test_table_getitem_int(tbl):
    row = tbl[0]
    assert isinstance(row, RowType)
    assert int(row.kind) == 1
    assert int(row.max) == 255


def test_table_getitem_slice_returns_structured_array(tbl):
    sub = tbl[0:1]
    assert isinstance(sub, np.ndarray)
    assert sub.dtype == tbl.dtype
    assert len(sub) == 1


def test_table_iter(tbl):
    rows = list(tbl)
    assert len(rows) == 2
    assert all(isinstance(r, RowType) for r in rows)
    assert int(rows[1].max) == 65535


def test_table_rows_helper(tbl):
    rows = tbl.rows()
    assert len(rows) == 2
    assert isinstance(rows[0], RowType)


def test_table_repr(tbl):
    assert 'test' in repr(tbl)
    assert 'len=2' in repr(tbl)


# --------------------------------------------------------------------
# Adapter protocol
# --------------------------------------------------------------------

def test_adapter_repr(tbl):
    adapter = tbl.adapter['kind']
    assert 'kind' in repr(adapter)
    assert repr(tbl) in repr(adapter)


def test_adapter_init_column():
    adapter = NPColAdapter(parent=None, name='bits')
    ary = np.array([(1, 2, 3)], dtype=[('kind', 'u1'), ('max', 'u8'), ('bits', 'u1')])
    col = adapter.init_column(ary, np.uint8)
    assert col.tolist() == [3]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
