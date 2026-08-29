import numpy as np
import pytest
from typing import NamedTuple, get_type_hints

from GenericTable import (
    NpColDef,
    get_defs_from_table_type,
    dtype_from_fields,
    Table,
    TableFields,
    table_item_from_fields,
    table_range_from_type,
    NpColTable,
    item_type_of,
    range_type_of,
)


# --------------------------------------------------------------------
# Table type / range / item hints
# --------------------------------------------------------------------

class RowType(NamedTuple):
    kind: np.uint8
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8


class DeclaredRow(NamedTuple):
    kind: np.ndarray | np.uint8
    bits: np.ndarray | np.uint8


def test_col_defs_from_table_type():
    cols = get_defs_from_table_type(RowType)
    assert len(cols) == 4
    assert cols[0] == NpColDef('kind', np.dtype(np.uint8))
    assert cols[1] == NpColDef('abs_min', np.dtype(np.uint64))
    assert cols[2] == NpColDef('max', np.dtype(np.uint64))
    assert cols[3] == NpColDef('bits', np.dtype(np.uint8))


def test_col_defs_from_declared_unions():
    cols = get_defs_from_table_type(DeclaredRow)
    assert cols[0] == NpColDef('kind', np.dtype(np.uint8))
    assert cols[1] == NpColDef('bits', np.dtype(np.uint8))


def test_item_and_range_of_hint():
    hint = np.ndarray | np.uint8
    assert item_type_of(hint) is np.uint8
    assert range_type_of(hint) is np.ndarray
    assert item_type_of(np.uint8) is np.uint8
    assert range_type_of(np.uint8) is None


def test_np_col_def_repr():
    col = NpColDef('kind', np.dtype(np.uint8))
    assert 'kind' in repr(col)
    assert 'uint8' in repr(col)


def test_dtype_from_fields():
    fields = get_defs_from_table_type(RowType)
    dt = dtype_from_fields(fields)
    assert dt.names == ('kind', 'abs_min', 'max', 'bits')
    assert dt['kind'] == np.dtype(np.uint8)
    assert dt['max'] == np.dtype(np.uint64)


def test_table_item_from_fields():
    class DeclFields(TableFields):
        kind: np.ndarray | np.uint8
        max: np.ndarray | np.uint64

    item = table_item_from_fields(DeclFields)
    assert item.__name__ == "Decl"                       # 'Fields' suffix stripped
    assert item._fields == ("kind", "max")
    assert get_type_hints(item)["kind"] is np.uint8      # specific: item members only
    assert get_type_hints(item)["max"] is np.uint64


def test_table_range_from_type():
    rng_type = table_range_from_type(RowType)
    assert rng_type.__name__ == "RowTypeRange"
    assert rng_type._fields == RowType._fields
    rng_type2 = table_range_from_type(DeclaredRow)
    assert rng_type2.__annotations__["kind"] is np.ndarray


def test_table_range_basename():
    rng_type = table_range_from_type(RowType, basename="Other")
    assert rng_type.__name__ == "OtherRange"


# --------------------------------------------------------------------
# Table construction
# --------------------------------------------------------------------

@pytest.fixture
def tbl():
    return Table(name="test",
                 data=[(1, 0, 255, 8), (2, 1, 65535, 16)],
                 table_type=RowType)


def test_table_build(tbl):
    assert tbl.name == "test"
    assert tbl.table_type is RowType
    assert len(tbl.fields) == 4
    assert len(tbl) == 2
    assert tbl.dtype.names == ('kind', 'abs_min', 'max', 'bits')


def test_table_empty_data():
    tbl = Table(name="empty",
                data=[],
                table_type=RowType)
    assert len(tbl) == 0
    assert list(tbl) == []


def test_table_columns_are_arrays(tbl):
    assert isinstance(tbl.kind, np.ndarray)
    assert tbl.kind.dtype == np.uint8
    assert isinstance(tbl.max, np.ndarray)
    assert tbl.max.dtype == np.uint64
    assert tbl.max.tolist() == [255, 65535]


def test_table_field_names(tbl):
    assert tbl._field_names == ['kind', 'abs_min', 'max', 'bits']


def test_table_col_attrs_identity(tbl):
    assert tbl.kind is tbl.cols[0]
    assert tbl.abs_min is tbl.cols[1]
    assert tbl.max is tbl.cols[2]
    assert tbl.bits is tbl.cols[3]


def test_check_col_attrs_fires_on_tampered_attr(tbl):
    tbl.kind = tbl.bits
    with pytest.raises(AssertionError):
        tbl._check_col_attrs()


# --------------------------------------------------------------------
# Column types are specified by the table type
# --------------------------------------------------------------------

def test_declared_range_type_is_enforced():
    from QueryableTable import QueryableTable

    with pytest.raises(TypeError, match="declares range ndarray, got ConstraintColumn"):
        QueryableTable("bad", [(1, 8)], DeclaredRow)


def test_np_col_table_builds_declared_ranges():
    tbl = NpColTable(name="ok",
                     data=[(1, 8)],
                     table_type=DeclaredRow)
    assert isinstance(tbl.kind, np.ndarray)   # declared range member


# --------------------------------------------------------------------
# Declaration-driven tables — variants derived inside the framework
# --------------------------------------------------------------------

def test_table_derives_from_inherited_declaration():
    class DeclFields(TableFields):
        kind: np.ndarray | np.uint8
        bits: np.ndarray | np.uint8

    class DeclTable(Table, DeclFields):
        pass

    tbl = DeclTable(name="decl", data=[(1, 8), (2, 16)])
    assert tbl.table_type is DeclFields                       # the declaration IS the table type
    assert tbl.item_type.__name__ == "Decl"
    assert tbl.item_type._fields == ("kind", "bits")
    assert get_type_hints(tbl.item_type)["kind"] is np.uint8      # item variant: specific scalars
    assert tbl.range_type.__name__ == "DeclRange"
    assert get_type_hints(tbl.range_type)["kind"] is np.ndarray   # range variant: specific ranges
    assert tbl.item_type is DeclTable(name="x", data=[(3, 8)]).item_type  # stable, cached per declaration

    row = tbl[0]
    assert isinstance(row, tbl.item_type)
    assert row.kind == np.uint8(1)
    rng = tbl[0:1]
    assert rng.__class__ is tbl.range_type
    assert [int(v) for v in rng.kind] == [1]


def test_table_without_type_or_declaration_raises():
    class BareTable(Table):
        pass

    with pytest.raises(TypeError, match="TableFields"):
        BareTable("bare", [(1,)])


def test_explicit_table_type_beats_inherited_declaration():
    class DeclFields(TableFields):
        kind: np.ndarray | np.uint8
        bits: np.ndarray | np.uint8

    class DeclTable(Table, DeclFields):
        pass

    tbl = DeclTable(name="legacy", data=[(1, 0, 255, 8)], table_type=RowType)
    assert tbl.table_type is RowType
    assert tbl.item_type is RowType


# --------------------------------------------------------------------
# Item / Range variants on selection
# --------------------------------------------------------------------

def test_getitem_int_returns_item_variant(tbl):
    row = tbl[0]
    assert isinstance(row, RowType)
    assert int(row.kind) == 1
    assert int(row.max) == 255


def test_getitem_numpy_int_returns_item_variant(tbl):
    assert tbl[np.int64(1)].max == np.uint64(65535)


def test_getitem_slice_returns_range_variant(tbl):
    rng = tbl[0:1]
    assert rng.__class__ is tbl.range_type
    assert rng.__class__.__name__ == "RowTypeRange"
    assert [int(v) for v in rng.kind] == [1]
    assert [int(v) for v in rng.max] == [255]

    rng2 = tbl[0:2]
    assert [int(v) for v in rng2.bits] == [8, 16]


def test_table_iter(tbl):
    rows = list(tbl)
    assert len(rows) == 2
    assert all(isinstance(r, RowType) for r in rows)
    assert int(rows[1].max) == 65535


def test_table_rows_helper(tbl):
    rows = tbl.rows()
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert isinstance(rows[0], RowType)


def test_table_repr(tbl):
    assert 'test' in repr(tbl)
    assert 'len=2' in repr(tbl)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
