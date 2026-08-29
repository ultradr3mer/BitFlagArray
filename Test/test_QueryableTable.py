from typing import NamedTuple

import numpy as np
import pytest

from clarautils.QueryableTable import (
    QueryableTable,
    ConstraintColumn,
    Constraint,
    Query,
    Undefined,
)


# --------------------------------------------------------------------
# Row types
# --------------------------------------------------------------------

class TypeLookupRow(NamedTuple):
    signed: np.bool_
    abs_min: np.uint64
    max: np.uint64
    bits: np.uint8
    prev_max: np.int64


class SimpleRow(NamedTuple):
    id: np.uint32
    value: np.uint64


@pytest.fixture
def lookup_data():
    kind = {'u': False, 'i': True}
    sizes = [1, 2, 4, 8]
    types = [np.dtype(f"{k}{s}") for k in kind for s in sizes]
    rows = []
    last_max = {-1: -1, 1: -1}
    for t in types:
        s = 1 if t.kind == 'i' else -1
        rows.append((kind[t.kind], -np.iinfo(t).min, np.iinfo(t).max, np.iinfo(t).bits, last_max[s]))
        last_max[s] = int(np.iinfo(t).max)
    return rows


@pytest.fixture
def tbl(lookup_data) -> QueryableTable[TypeLookupRow]:
    return QueryableTable("TypeLookup", lookup_data, TypeLookupRow)


# --------------------------------------------------------------------
# QueryableTable — construction
# --------------------------------------------------------------------

def test_build(tbl):
    assert tbl.name == "TypeLookup"
    assert tbl.table_type is TypeLookupRow
    assert tbl.dtype.names == ('signed', 'abs_min', 'max', 'bits', 'prev_max')
    assert tbl.dtype['signed'] == np.dtype(np.bool_)
    assert tbl.dtype['max'] == np.dtype(np.uint64)


def test_build_columns_are_constraint_columns(tbl):
    assert isinstance(tbl.signed, ConstraintColumn)
    assert isinstance(tbl.max, ConstraintColumn)
    assert isinstance(tbl.prev_max, ConstraintColumn)


def test_len(tbl):
    assert len(tbl) == 8


def test_getitem_int_returns_row(tbl):
    row = tbl[0]
    assert isinstance(row, TypeLookupRow)
    assert bool(row.signed) is False
    assert int(row.max) == 255


def test_iter_yields_rows(tbl):
    rows = list(tbl)
    assert len(rows) == 8
    assert all(isinstance(r, TypeLookupRow) for r in rows)
    assert bool(rows[0].signed) is False
    assert bool(rows[4].signed) is True


def test_rows_helper(tbl):
    rows = tbl.rows()
    assert len(rows) == 8
    assert isinstance(rows[0], TypeLookupRow)


def test_adapters_use_base_init(tbl):
    # column attrs are the columns themselves; the adapter machinery is gone
    assert tbl.signed is tbl.cols[0]
    assert tbl.signed.name == 'signed'
    assert tbl.signed.table is tbl.data


def test_direct_construction_simple_row():
    stbl: QueryableTable[SimpleRow] = QueryableTable("simple", [(0, 10), (1, 20), (2, 30)], SimpleRow)
    assert stbl.name == "simple"
    assert stbl.table_type is SimpleRow
    assert len(stbl) == 3
    q = stbl.value >= 20
    assert list(q.indices) == [1, 2]


# --------------------------------------------------------------------
# QueryableTable — lazy queries (constraints compose, evaluate on demand)
# --------------------------------------------------------------------

def test_eq_query(tbl):
    q = tbl.signed == False
    assert isinstance(q, Constraint)
    assert list(q.indices) == [0, 1, 2, 3]


def test_and_query(tbl):
    val = 123
    smallest = (tbl.signed == False) & (tbl.max >= val) & (tbl.prev_max < val)
    assert isinstance(smallest, Query)
    assert list(smallest.indices) == [0]


def test_or_query(tbl):
    q = (tbl.bits <= 8) | (tbl.max >= 2 ** 32)
    assert list(q.indices) == [0, 3, 4, 7]


def test_undefined_constraint(tbl):
    q = tbl.signed == Undefined
    assert list(q.indices) == [0, 1, 2, 3, 4, 5, 6, 7]
    # composing with Undefined keeps the other constraint intact
    combined = (tbl.signed == Undefined) & (tbl.bits >= 32)
    assert list(combined.indices) == [2, 3, 6, 7]


def test_filter_rows_via_iter(tbl):
    signed_rows = [t for t in tbl if t.signed == True]
    assert len(signed_rows) == 4
    assert all(bool(r.signed) is True for r in signed_rows)


def test_ne_query(tbl):
    q = tbl.signed != False
    assert list(q.indices) == [4, 5, 6, 7]


def test_lt_query(tbl):
    q = tbl.bits < 16
    assert list(q.indices) == [0, 4]


def test_le_query(tbl):
    q = tbl.bits <= 8
    assert list(q.indices) == [0, 4]


def test_gt_query(tbl):
    q = tbl.bits > 8
    assert list(q.indices) == [1, 2, 3, 5, 6, 7]


def test_ge_query(tbl):
    q = tbl.bits >= 16
    assert list(q.indices) == [1, 2, 3, 5, 6, 7]


def test_query_intersection(tbl):
    q1 = tbl.signed == False
    q2 = tbl.bits >= 16
    combined = q1 & q2
    assert list(combined.indices) == [1, 2, 3]


# --------------------------------------------------------------------
# Query execution — get_first / get_all, Item / Range variants
# --------------------------------------------------------------------

def test_column_get_all(tbl):
    q = tbl.bits >= 32
    assert [int(v) for v in tbl.max.get_all(q)] == [2 ** 32 - 1, 2 ** 64 - 1, 2 ** 31 - 1, 2 ** 63 - 1]


def test_column_get_first(tbl):
    assert int(tbl.bits.get_first(tbl.signed == True)) == 8


def test_column_getitem_item_and_range(tbl):
    assert int(tbl.bits[0]) == 8                                   # Item Type
    rng = tbl.bits[1:3]                                             # Range Type variant
    assert isinstance(rng, ConstraintColumn)
    assert [int(v) for v in rng.column] == [16, 32]


def test_table_get_all_returns_range_variant(tbl):
    rng = tbl.get_all(tbl.bits <= 8)
    assert rng.__class__.__name__ == "TypeLookupRowRange"
    assert [bool(v) for v in rng.signed.column] == [False, True]
    assert [int(v) for v in rng.bits.column] == [8, 8]


def test_table_get_first_returns_item_variant(tbl):
    row = tbl.get_first(tbl.signed == True)
    assert isinstance(row, TypeLookupRow)
    assert bool(row.signed) is True
    assert int(row.bits) == 8


def test_table_slice_returns_range_variant(tbl):
    rng = tbl[4:8]
    assert rng.__class__.__name__ == "TypeLookupRowRange"
    assert all(bool(v) for v in rng.signed.column)
    assert [int(v) for v in rng.bits.column] == [8, 16, 32, 64]


def test_table_select_max(tbl):
    selection = tbl.max.get_all(tbl.abs_min >= 341)
    assert selection.size == 3

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

