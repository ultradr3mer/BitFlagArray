"""Zwei Varianten, Referenz-Spalten (z. B. ``Type[Any]``) in eine QueryableTable zu bringen.

A) Object-dtype-Feld: Deklaration ``np_type: CCol | np.object_`` — die
   Referenz liegt als object-Field direkt im strukturierten Array. Keine
   neue Maschinerie (Queries / Item / Range laufen unverändert), dafuer
   ist das Feld byte-opak und nur ==/!= query-bar. numpy-Eigenheit:
   object-Array == Skalar-Typ crasht, numpy-Typen als np.dtype(t) fragen.

B) Side storage: numpy-Daten bleiben pur (kein object-Field), die
   Referenzen liegen parallel in einem dict und werden bei Item-/Range-
   Zugriff wieder verheiratet (RefTable unten). Beliebig viele Ref-
   Spalten, Daten bleiben packbar; dafuer baut RefTable Item-/Range-
   Varianten selbst zusammen.
"""

from typing import get_type_hints

import numpy as np
import pytest

from clarautils.GenericTable import (
    TableFields,
    item_type_of,
    table_item_from_fields,
    table_range_from_type,
)
from clarautils.QueryableTable import QueryableTable, CCol, Constraint
from clarautils.common import get_first_or


UINTS = [np.uint8, np.uint16, np.uint32, np.uint64]
INTS = [np.int8, np.int16, np.int32, np.int64]


class RefColFields(TableFields):
    signed: CCol | np.bool_
    bits: CCol | np.uint8
    np_type: CCol | np.object_       # Referenz-Spalte: numpy-Skalar-Klasse


def build_ref_rows():
    return [(issubclass(t, np.signedinteger), np.iinfo(t).bits, t)
            for t in UINTS + INTS]


# --------------------------------------------------------------------
# A) Object-dtype-Feld — Referenz direkt im strukturierten Array
# --------------------------------------------------------------------

class ObjColTable(QueryableTable[RefColFields], RefColFields):
    """Variante A: np_type ist ein object-Field in self.data."""


@pytest.fixture
def o_tbl() -> ObjColTable:
    return ObjColTable("TypeLookup", build_ref_rows())


def test_a_dtype_has_object_field(o_tbl):
    assert o_tbl.dtype.names == ("signed", "bits", "np_type")
    assert o_tbl.dtype["np_type"] == np.dtype("O")
    assert isinstance(o_tbl.np_type, CCol)


def test_a_refs_live_inside_data(o_tbl):
    assert o_tbl.data["np_type"][0] is np.uint8


def test_a_item_keeps_reference(o_tbl):
    row = o_tbl[0]
    assert isinstance(row, o_tbl.item_type)
    assert row.np_type is np.uint8


def test_a_range_slices_refs(o_tbl):
    rng = o_tbl[4:]
    assert isinstance(rng.np_type, CCol)
    assert list(rng.np_type.column) == INTS
    assert [bool(v) for v in rng.signed.column] == [True] * 4


def test_a_query_on_ref_column(o_tbl):
    q = o_tbl.np_type == np.dtype(np.uint32)   # object-Array == Skalar-Typ crasht
    assert isinstance(q, Constraint)
    assert list(q.indices) == [2]
    assert list((o_tbl.np_type != np.dtype(np.uint8)).indices) == [1, 2, 3, 4, 5, 6, 7]


def test_a_composed_query(o_tbl):
    q = (o_tbl.signed == True) & (o_tbl.bits >= 32)
    assert list(q.indices) == [6, 7]


def test_a_get_first_returns_reference(o_tbl):
    assert o_tbl.get_first(o_tbl.np_type == np.dtype(np.int8)).np_type is np.int8
    assert o_tbl.np_type.get_first(o_tbl.bits >= 32) is np.uint32


def test_a_get_all_returns_ref_array(o_tbl):
    sel = o_tbl.np_type.get_all(o_tbl.signed == True)
    assert sel.dtype == np.dtype("O")
    assert list(sel) == INTS


def test_a_iter(o_tbl):
    rows = list(o_tbl)
    assert [r.np_type for r in rows] == UINTS + INTS


def test_a_reference_identity():
    sentinel = object()
    tbl = ObjColTable("t", [(True, 8, sentinel)])
    assert tbl.data["np_type"][0] is sentinel
    assert tbl[0].np_type is sentinel


# --------------------------------------------------------------------
# B) Side storage — RefTable komponiert QueryableTable + Ref-Spalten
# --------------------------------------------------------------------

def _is_ref_item(hint) -> bool:
    item = item_type_of(hint)
    return item is np.object_ or item is object


def numpy_fields_only(fields_cls: type) -> type:
    """Deklaration ohne Referenz-Felder (die liegen im Side storage)."""
    np_hints = {n: h for n, h in get_type_hints(fields_cls).items()
                if not _is_ref_item(h)}
    return type(f"{fields_cls.__name__}Np", (TableFields,), {"__annotations__": np_hints})


class RefCCol(CCol):
    """CCol \u00fcber ein nacktes object-Array (side storage, kein Feld-Selektor)."""

    __slots__ = ()

    def __init__(self, values, name: str):
        super().__init__(np.asarray(values, dtype=object))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.column[key]
        return RefCCol(self.column[key], self.name)


class RefTable:
    """Variante B: numpy-Daten pur, Referenzen im parallelen dict.

    Ref-Spalten sind ganz normale CCols (query-bar); Item-/Range-Varianten
    ziehen die Referenzen bei Zugriff wieder zu den numpy-Daten.
    """

    def __init__(self, name: str, rows, fields_cls: type,
                 ref_data: dict[str, list]):
        self.hints = get_type_hints(fields_cls)
        self.ref_names = [n for n in self.hints if _is_ref_item(self.hints[n])]
        self.tbl = QueryableTable(name, rows, numpy_fields_only(fields_cls))
        self.ref_cols = {n: RefCCol(refs, n) for n, refs in ref_data.items()}
        assert set(self.ref_cols) == set(self.ref_names)
        for n, col in self.ref_cols.items():
            assert len(col.column) == len(self.tbl), f"ref column {n!r}: length mismatch"
        self.item_type = table_item_from_fields(fields_cls)
        self.range_type = table_range_from_type(fields_cls, CCol,
                                                basename=self.item_type.__name__)

    def __getattr__(self, name):
        d = self.__dict__
        if name in d.get("ref_cols", ()):
            return d["ref_cols"][name]
        if "tbl" in d:
            return getattr(d["tbl"], name)
        raise AttributeError(name)

    def __len__(self):
        return len(self.tbl)

    def __iter__(self):
        return (self[i] for i in range(len(self.tbl)))

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            rec = self.tbl.data[key]
            return self.item_type(*[
                self.ref_cols[n].column[key] if n in self.ref_cols else rec[n]
                for n in self.hints])
        sub = self.tbl.data[key]
        return self.range_type(*[
            RefCCol(self.ref_cols[n].column[key], n) if n in self.ref_cols
            else self.tbl.build_column(sub, n)
            for n in self.hints])

    def get_all(self, query):
        return self[query.indices]

    def get_first(self, query):
        return self[get_first_or(query.indices, KeyError())]


@pytest.fixture
def rt() -> RefTable:
    rows = [(issubclass(t, np.signedinteger), np.iinfo(t).bits)
            for t in UINTS + INTS]
    return RefTable("TypeLookup", rows, RefColFields,
                    ref_data={"np_type": UINTS + INTS})


def test_b_data_stays_pure(rt):
    assert rt.tbl.data.dtype.names == ("signed", "bits")
    assert rt.ref_cols["np_type"].column[0] is np.uint8


def test_b_item_keeps_reference(rt):
    row = rt[0]
    assert isinstance(row, rt.item_type)
    assert row.np_type is np.uint8


def test_b_range_slices_refs(rt):
    rng = rt[4:]
    assert isinstance(rng.np_type, RefCCol)
    assert list(rng.np_type.column) == INTS
    assert [bool(v) for v in rng.signed.column] == [True] * 4


def test_b_query_numpy_and_ref_columns(rt):
    assert list((rt.np_type == np.dtype(np.uint32)).indices) == [2]
    q = (rt.signed == True) & (rt.bits >= 32)
    assert list(q.indices) == [6, 7]


def test_b_get_first_returns_reference(rt):
    assert rt.get_first(rt.np_type == np.dtype(np.int8)).np_type is np.int8
    assert rt.np_type.get_first(rt.bits >= 32) is np.uint32


def test_b_get_all(rt):
    rng = rt.get_all(rt.signed == True)
    assert list(rng.np_type.column) == INTS


def test_b_iter(rt):
    assert [r.np_type for r in rt] == UINTS + INTS


def test_b_reference_identity():
    sentinel = object()
    rt2 = RefTable("t", [(True, 8)], RefColFields, ref_data={"np_type": [sentinel]})
    assert rt2.ref_cols["np_type"].column[0] is sentinel
    assert rt2[0].np_type is sentinel


# --------------------------------------------------------------------
# beide Varianten liefern dieselben Antworten
# --------------------------------------------------------------------

def test_a_and_b_agree(o_tbl, rt):
    q_a = (o_tbl.signed == False) & (o_tbl.bits >= 32)
    q_b = (rt.signed == False) & (rt.bits >= 32)
    assert list(q_a.indices) == list(q_b.indices) == [2, 3]
    assert o_tbl.get_first(q_a).np_type is np.uint32
    assert rt.get_first(q_b).np_type is np.uint32


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
