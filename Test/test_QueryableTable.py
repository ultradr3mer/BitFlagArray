from typing import NamedTuple

import numpy as np
import pytest

from GenricTable import TableCreator
from QueryableTable import (
    Table,
    QueryableTable,
    ConstraintColumn,
    ConstraintSelection,
)


# --------------------------------------------------------------------
# Row types
# --------------------------------------------------------------------

class TypeLookupRow(NamedTuple):
    signed: np.bool
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


# --------------------------------------------------------------------
# Table abstract base
# --------------------------------------------------------------------

def test_table_is_abstract():
    with pytest.raises(TypeError):
        Table("x", [], TypeLookupRow)  # type: ignore[abstract]


def test_queryable_subclass_requires_get_row_type():
    class Bad(QueryableTable[TypeLookupRow]):
        pass
    with pytest.raises(TypeError):
        Bad("x", [], TypeLookupRow)  # type: ignore[abstract]


# --------------------------------------------------------------------
# QueryableTable[TRow] — subclass API
# --------------------------------------------------------------------

class TypeLookup(QueryableTable[TypeLookupRow]):
    @classmethod
    def get_row_type(cls) -> type[TypeLookupRow]:
        return TypeLookupRow


def test_build(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    assert tbl.name == "TypeLookup"
    assert tbl.row_type is TypeLookupRow
    assert tbl.dtype.names == ('signed', 'abs_min', 'max', 'bits', 'prev_max')
    assert tbl.dtype['signed'] == np.dtype(np.bool)
    assert tbl.dtype['max'] == np.dtype(np.uint64)


def test_build_columns_are_constraint_columns(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    assert isinstance(tbl.signed, ConstraintColumn)
    assert isinstance(tbl.max, ConstraintColumn)
    assert isinstance(tbl.prev_max, ConstraintColumn)


def test_len(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    assert len(tbl) == 8


def test_getitem_int_returns_row(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    row = tbl[0]
    assert isinstance(row, TypeLookupRow)
    assert bool(row.signed) is False
    assert int(row.max) == 255


def test_iter_yields_rows(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    rows = list(tbl)
    assert len(rows) == 8
    assert all(isinstance(r, TypeLookupRow) for r in rows)
    assert bool(rows[0].signed) is False
    assert bool(rows[4].signed) is True


def test_rows_helper(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    rows = tbl.rows()
    assert len(rows) == 8
    assert isinstance(rows[0], TypeLookupRow)


# --------------------------------------------------------------------
# QueryableTable — querying
# --------------------------------------------------------------------

def test_eq_query(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    sel = tbl.signed == False
    assert isinstance(sel, ConstraintSelection)
    assert list(sel.indices) == [0, 1, 2, 3]


def test_and_pipeline(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    val = 123
    smallest = (
        tbl.signed == False
        and tbl.max >= val
        and tbl.prev_max < val
    )
    assert isinstance(smallest, ConstraintSelection)
    assert list(smallest.indices) == [0]


def test_amp_pipeline(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    val = 123
    smallest = (
        (((tbl.signed == False) & tbl.max) >= val) & tbl.prev_max
    ) < val
    assert list(smallest.indices) == [0]


def test_filter_rows_via_iter(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    signed_rows = [t for t in tbl if t.signed == True]
    assert len(signed_rows) == 4
    assert all(bool(r.signed) is True for r in signed_rows)


def test_ne_query(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    sel = tbl.signed != False
    assert list(sel.indices) == [4, 5, 6, 7]


def test_lt_query(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    sel = tbl.bits < 16
    assert list(sel.indices) == [0, 4]


def test_le_query(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    sel = tbl.bits <= 8
    assert list(sel.indices) == [0, 4]


def test_gt_query(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    sel = tbl.bits > 8
    assert list(sel.indices) == [1, 2, 3, 5, 6, 7]


def test_ge_query(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    sel = tbl.bits >= 16
    assert list(sel.indices) == [1, 2, 3, 5, 6, 7]


def test_selection_intersection(lookup_data):
    tbl = TypeLookup.build("TypeLookup", lookup_data)
    sel1 = tbl.signed == False
    sel2 = tbl.bits >= 16
    combined = sel1 & sel2
    assert list(combined.indices) == [1, 2, 3]


# --------------------------------------------------------------------
# TableCreator
# --------------------------------------------------------------------

def test_table_creator(lookup_data):
    creator = TableCreator[TypeLookupRow, TypeLookup](TypeLookup)
    tbl = creator.build("via_creator", lookup_data)
    assert isinstance(tbl, TypeLookup)
    assert tbl.name == "via_creator"
    assert tbl.row_type is TypeLookupRow
    assert len(tbl) == 8
    sel = tbl.signed == False
    assert list(sel.indices) == [0, 1, 2, 3]


def test_table_creator_simple_row():
    class SimpleTable(QueryableTable[SimpleRow]):
        @classmethod
        def get_row_type(cls) -> type[SimpleRow]:
            return SimpleRow

    creator = TableCreator[SimpleRow, SimpleTable](SimpleTable)
    tbl = creator.build("simple", [(0, 10), (1, 20), (2, 30)])
    assert len(tbl) == 3
    sel = tbl.value >= 20
    assert list(sel.indices) == [1, 2]


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
