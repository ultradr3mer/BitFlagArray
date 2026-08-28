import numpy as np
import pytest

from GenricTable import (
    NpColDef,
    to_col_defs,
    create_table_type,
    TableType,
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
# create_table_type / TableType
# --------------------------------------------------------------------

@pytest.fixture
def my_types():
    spec: FieldSpec = [
        ('kind', np.bool), ('abs_min', np.uint64),
        ('max', np.uint64), ('bits', np.uint8),
    ]
    return create_table_type("MyTypes", to_col_defs(spec))


def test_create_table_type_returns_table_type(my_types):
    assert isinstance(my_types, TableType)


def test_dtype(my_types):
    dt = my_types.dtype
    assert dt.names == ('kind', 'abs_min', 'max', 'bits')
    assert dt['kind'] == np.dtype(np.bool)
    assert dt['max'] == np.dtype(np.uint64)
    assert dt['bits'] == np.dtype(np.uint8)


def test_row_type(my_types):
    Row = my_types.row_type
    row = Row(kind=True, abs_min=0, max=255, bits=8)
    assert int(row.max) == 255
    assert int(row.bits) == 8
    assert bool(row.kind) is True


def test_build(my_types):
    tbl = my_types.build([
        (True, 0, 255, 8),
        (False, 1, 65535, 16),
    ])
    assert len(tbl.kind) == 2
    assert tbl.kind[0] == True
    assert tbl.kind[1] == False
    assert int(tbl.max[0]) == 255
    assert int(tbl.max[1]) == 65535
    assert int(tbl.bits[0]) == 8
    assert int(tbl.bits[1]) == 16


def test_build_field_dtypes(my_types):
    tbl = my_types.build([(True, 0, 255, 8)])
    assert tbl.kind.dtype == np.bool
    assert tbl.abs_min.dtype == np.uint64
    assert tbl.max.dtype == np.uint64
    assert tbl.bits.dtype == np.uint8


def test_row(my_types):
    ary = np.asarray(
        [(True, 0, 255, 8), (False, 1, 65535, 16)],
        dtype=my_types.dtype,
    )
    row0 = my_types.row(ary, 0)
    row1 = my_types.row(ary, 1)
    assert bool(row0.kind) is True
    assert bool(row1.kind) is False
    assert int(row0.max) == 255
    assert int(row1.bits) == 16


def test_rows(my_types):
    ary = np.asarray(
        [(True, 0, 255, 8), (False, 1, 65535, 16), (True, 2, 4294967295, 32)],
        dtype=my_types.dtype,
    )
    rows = my_types.rows(ary)
    assert len(rows) == 3
    assert bool(rows[0].kind) is True
    assert bool(rows[1].kind) is False
    assert int(rows[2].max) == 4294967295


def test_row_is_hashable(my_types):
    row = my_types.row_type(kind=True, abs_min=0, max=255, bits=8)
    assert hash(row)


def test_row_equality(my_types):
    Row = my_types.row_type
    r1 = Row(True, 0, 255, 8)
    r2 = Row(True, 0, 255, 8)
    r3 = Row(False, 0, 255, 8)
    assert r1 == r2
    assert r1 != r3


def test_table_unpacking(my_types):
    tbl = my_types.build([
        (True, 0, 255, 8),
        (False, 1, 65535, 16),
    ])
    kind, abs_min, max_, bits, _row_type = tbl
    assert np.array_equal(kind, [True, False])
    assert np.array_equal(max_, [255, 65535])


def test_table_has_row_type_attribute(my_types):
    tbl = my_types.build([(True, 0, 255, 8)])
    assert tbl.row_type is my_types.row_type


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
