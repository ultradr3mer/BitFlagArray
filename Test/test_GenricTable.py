import numpy as np
import pytest

from GenricTable import (
    TblAndColumn,
    MyTable,
    MyRow,
)


# --------------------------------------------------------------------
# skeleton / type-level
# --------------------------------------------------------------------

def test_tbl_and_column_is_named_tuple():
    assert issubclass(TblAndColumn, tuple)
    assert TblAndColumn._fields == ("kind", "abs_min", "max", "bits")


def test_my_table_field_types_are_ndarray():
    tbl = MyTable(
        kind=np.array([True, False]),
        abs_min=np.array([0, 1], dtype=np.uint64),
        max=np.array([255, 65535], dtype=np.uint64),
        bits=np.array([8, 16], dtype=np.uint8),
    )
    assert isinstance(tbl.kind, np.ndarray)
    assert tbl.kind.dtype == np.bool
    assert tbl.abs_min.dtype == np.uint64
    assert tbl.max.dtype == np.uint64
    assert tbl.bits.dtype == np.uint8


def test_my_row_field_types_are_scalars():
    row = MyRow(kind=True, abs_min=0, max=255, bits=8)
    # scalars, not arrays
    assert not isinstance(row.kind, np.ndarray)
    assert row.kind is True or bool(row.kind) is True
    assert int(row.max) == 255
    assert int(row.bits) == 8


def test_table_and_row_share_field_names():
    assert MyTable._fields == MyRow._fields == TblAndColumn._fields


# --------------------------------------------------------------------
# construction / indexing
# --------------------------------------------------------------------

def test_table_construction_keyword():
    tbl = MyTable(
        kind=np.array([True]),
        abs_min=np.array([1], dtype=np.uint64),
        max=np.array([1], dtype=np.uint64),
        bits=np.array([1], dtype=np.uint8),
    )
    assert len(tbl.kind) == 1
    assert bool(tbl.kind[0]) is True


def test_table_construction_positional():
    tbl = MyTable(
        np.array([True]),
        np.array([1], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([1], dtype=np.uint8),
    )
    assert len(tbl) == 4  # NamedTuple length = number of fields
    assert bool(tbl.kind[0]) is True


def test_row_is_hashable():
    row = MyRow(kind=True, abs_min=0, max=255, bits=8)
    assert hash(row)  # NamedTuple is hashable


def test_row_equality():
    r1 = MyRow(True, 0, 255, 8)
    r2 = MyRow(True, 0, 255, 8)
    r3 = MyRow(False, 0, 255, 8)
    assert r1 == r2
    assert r1 != r3


def test_table_unpacking():
    tbl = MyTable(
        kind=np.array([True, False]),
        abs_min=np.array([0, 1], dtype=np.uint64),
        max=np.array([255, 65535], dtype=np.uint64),
        bits=np.array([8, 16], dtype=np.uint8),
    )
    kind, abs_min, max_, bits = tbl
    assert np.array_equal(kind, [True, False])
    assert np.array_equal(max_, [255, 65535])


def test_row_unpacking():
    row = MyRow(True, 0, 255, 8)
    kind, abs_min, max_, bits = row
    assert kind is True or bool(kind) is True
    assert bits == 8


# --------------------------------------------------------------------
# dtype derivation (what LookupTable machinery needs to know)
# --------------------------------------------------------------------

def test_field_dtypes_from_numpy_arrays():
    tbl = MyTable(
        kind=np.array([True]),
        abs_min=np.array([1], dtype=np.uint64),
        max=np.array([1], dtype=np.uint64),
        bits=np.array([1], dtype=np.uint8),
    )
    assert tbl.kind.dtype == np.bool
    assert tbl.abs_min.dtype == np.uint64
    assert tbl.max.dtype == np.uint64
    assert tbl.bits.dtype == np.uint8


def test_field_dtypes_match_spec():
    """The field dtypes of MyTable columns must match the TblAndColumn spec."""
    spec_dtypes = {
        'kind':     np.dtype(np.bool),
        'abs_min':  np.dtype(np.uint64),
        'max':      np.dtype(np.uint64),
        'bits':     np.dtype(np.uint8),
    }
    tbl = MyTable(
        kind=np.array([True], dtype=spec_dtypes['kind']),
        abs_min=np.array([1], dtype=spec_dtypes['abs_min']),
        max=np.array([1], dtype=spec_dtypes['max']),
        bits=np.array([1], dtype=spec_dtypes['bits']),
    )
    for name, expected in spec_dtypes.items():
        assert getattr(tbl, name).dtype == expected


# --------------------------------------------------------------------
# numpy interop
# --------------------------------------------------------------------

def test_table_columns_are_views_or_copies_of_source():
    kind = np.array([True, False])
    tbl = MyTable(
        kind=kind,
        abs_min=np.array([0, 1], dtype=np.uint64),
        max=np.array([255, 65535], dtype=np.uint64),
        bits=np.array([8, 16], dtype=np.uint8),
    )
    assert tbl.kind is kind  # NamedTuple stores the reference


def test_iteration_over_rows_from_table_columns():
    tbl = MyTable(
        kind=np.array([True, False, True]),
        abs_min=np.array([0, 1, 2], dtype=np.uint64),
        max=np.array([255, 65535, 255], dtype=np.uint64),
        bits=np.array([8, 16, 8], dtype=np.uint8),
    )
    rows = [
        MyRow(
            kind=bool(tbl.kind[i]),
            abs_min=int(tbl.abs_min[i]),
            max=int(tbl.max[i]),
            bits=int(tbl.bits[i]),
        )
        for i in range(len(tbl.kind))
    ]
    assert len(rows) == 3
    assert rows[0].kind is True or bool(rows[0].kind) is True
    assert rows[1].kind is False or bool(rows[1].kind) is False
    assert rows[2].max == 255
    assert rows[1].bits == 16


def test_structured_array_to_rows():
    """Simulate what LookupTable.__iter__ does: structured array -> MyRow."""
    dt = np.dtype([
        ('kind', np.bool),
        ('abs_min', np.uint64),
        ('max', np.uint64),
        ('bits', np.uint8),
    ])
    arr = np.array(
        [(True, 0, 255, 8), (False, 1, 65535, 16)],
        dtype=dt,
    )
    rows = [MyRow(kind=r['kind'], abs_min=r['abs_min'], max=r['max'], bits=r['bits']) for r in arr]
    assert bool(rows[0].kind) is True
    assert bool(rows[1].kind) is False
    assert int(rows[0].max) == 255
    assert int(rows[1].bits) == 16


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
