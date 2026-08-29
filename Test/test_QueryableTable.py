from typing import NamedTuple

import numpy as np
import pytest

from QueryableTable import (
    QueryableTable,
    ConstraintColAdapter,
    ConstraintColumn,
    ConstraintSelection,
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
    assert tbl.row_type is TypeLookupRow
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
    adapter = tbl.adapter['signed']
    assert isinstance(adapter, ConstraintColAdapter)
    assert adapter.name == 'signed'
    assert adapter.parent is tbl


def test_direct_construction_simple_row():
    stbl: QueryableTable[SimpleRow] = QueryableTable("simple", [(0, 10), (1, 20), (2, 30)], SimpleRow)
    assert stbl.name == "simple"
    assert stbl.row_type is SimpleRow
    assert len(stbl) == 3
    sel = stbl.value >= 20
    assert list(sel.indices) == [1, 2]


# --------------------------------------------------------------------
# QueryableTable — querying
# --------------------------------------------------------------------

def test_eq_query(tbl):
    sel = tbl.signed == False
    assert isinstance(sel, ConstraintSelection)
    assert list(sel.indices) == [0, 1, 2, 3]


def test_and_pipeline(tbl):
    val = 123
    smallest = (
        tbl.signed == False
        and tbl.max >= val
        and tbl.prev_max < val
    )
    assert isinstance(smallest, ConstraintSelection)
    assert list(smallest.indices) == [0]


def test_amp_pipeline(tbl):
    val = 123
    smallest = (
        (((tbl.signed == False) & tbl.max) >= val) & tbl.prev_max
    ) < val
    assert list(smallest.indices) == [0]


def test_filter_rows_via_iter(tbl):
    signed_rows = [t for t in tbl if t.signed == True]
    assert len(signed_rows) == 4
    assert all(bool(r.signed) is True for r in signed_rows)


def test_ne_query(tbl):
    sel = tbl.signed != False
    assert list(sel.indices) == [4, 5, 6, 7]


def test_lt_query(tbl):
    sel = tbl.bits < 16
    assert list(sel.indices) == [0, 4]


def test_le_query(tbl):
    sel = tbl.bits <= 8
    assert list(sel.indices) == [0, 4]


def test_gt_query(tbl):
    sel = tbl.bits > 8
    assert list(sel.indices) == [1, 2, 3, 5, 6, 7]


def test_ge_query(tbl):
    sel = tbl.bits >= 16
    assert list(sel.indices) == [1, 2, 3, 5, 6, 7]


def test_selection_intersection(tbl):
    sel1 = tbl.signed == False
    sel2 = tbl.bits >= 16
    combined = sel1 & sel2
    assert list(combined.indices) == [1, 2, 3]


# --------------------------------------------------------------------
# ConstraintSelection / ConstraintColumn
# --------------------------------------------------------------------

def test_selection_and_column():
    sel = ConstraintSelection(np.array([0, 1, 2]))
    col = ConstraintColumn(np.array([10, 20, 30, 40]))
    restricted = sel & col
    assert isinstance(restricted, ConstraintColumn)
    assert list(restricted.column) == [10, 20, 30]


def test_selection_and_selection():
    s1 = ConstraintSelection(np.array([0, 1, 2, 3]))
    s2 = ConstraintSelection(np.array([1, 3, 5]))
    combined = s1 & s2
    assert list(combined.indices) == [1, 3]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
