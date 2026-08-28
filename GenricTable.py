from abc import ABC, abstractmethod
from typing import NamedTuple, Any, get_type_hints

import numpy as np
import numpy.typing as npt


#====================================================
#               COLUMN DEFINITIONS
#====================================================

class NpColDef(NamedTuple):
    name: str
    type: np.dtype[Any]


type FieldSpec = list[tuple[str, npt.DTypeLike]]


def to_col_defs(spec: FieldSpec) -> list[NpColDef]:
    """Convert a raw (name, dtype) field-spec into a list of NpColDef."""
    return [NpColDef(name, np.dtype(dt)) for name, dt in spec]


def _col_defs_from_rowtype(rowtype: type) -> list[NpColDef]:
    """Derive column definitions from a NamedTuple row type's annotations."""
    hints = get_type_hints(rowtype)
    return [NpColDef(name, np.dtype(hints[name])) for name in rowtype._fields]


def _dtype_from_fields(fields: list[NpColDef]) -> np.dtype[Any]:
    return np.dtype([(f.name, f.type) for f in fields])


#====================================================
#               TABLE BASE HIERARCHY
#====================================================

class Table[TRow, TCol](ABC):
    """Base for typed lookup tables backed by a numpy structured array.

    Subclass and override ``get_row_type()`` to bind a row NamedTuple,
    and ``_make_column()`` to specify the column wrapper type::

        class UserTable(Table[UserRow, np.ndarray]):
            @classmethod
            def get_row_type(cls) -> type[UserRow]:
                return UserRow

            @classmethod
            def _make_column(cls, ary, name) -> np.ndarray:
                return ary[name]
    """

    name: str
    fields: list[NpColDef]
    row_type: type[TRow]
    _ary: np.ndarray[Any, Any]

    def __init__(
        self,
        name: str,
        fields: list[NpColDef],
        row_type: type[TRow],
    ) -> None:
        self.name = name
        self.fields = fields
        self.row_type = row_type

    @classmethod
    @abstractmethod
    def get_row_type(cls) -> type[TRow]:
        """Gibt den Row-Typ dieser konkreten Tabelle zurück."""

    @classmethod
    @abstractmethod
    def _make_column(cls, ary: np.ndarray, name: str) -> TCol:
        """Erzeugt eine Spalte aus dem strukturierten Array und dem Feldnamen."""

    def _init_columns(self) -> None:
        for name in self._field_names:
            setattr(self, name, type(self)._make_column(self._ary, name))

    @classmethod
    def build(cls, name: str, data: npt.ArrayLike) -> "Table[TRow, TCol]":
        """Creator: leitet dtype vom Row-Typ ab, konvertiert Daten, instanziiert."""
        row_type = cls.get_row_type()
        fields = _col_defs_from_rowtype(row_type)
        dtype = _dtype_from_fields(fields)
        ary = np.asarray(data, dtype=dtype)

        tbl = cls(name, fields, row_type)
        tbl._ary = ary
        tbl._init_columns()
        return tbl

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._ary.dtype

    @property
    def _field_names(self) -> list[str]:
        return list(self.row_type._fields)

    def __iter__(self):
        for rec in self._ary:
            yield self.row_type(**{n: rec[n] for n in self.row_type._fields})

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            rec = self._ary[key]
            return self.row_type(**{n: rec[n] for n in self.row_type._fields})
        return self._ary[key]

    def __len__(self):
        return len(self._ary)

    def rows(self) -> list[Any]:
        return list(self)


class PlainTable[TRow](Table[TRow, np.ndarray]):
    """Table with plain numpy array columns."""

    @classmethod
    def _make_column(cls, ary: np.ndarray, name: str) -> np.ndarray:
        return ary[name]


#====================================================
#               TABLE CREATOR
#====================================================

class TableCreator[TRow, TTable: Table[TRow, Any]]:
    """Generischer Creator für Tabellen.

    Hält den Tabellentyp und dessen Row-Typ und erstellt Instanzen::

        creator = TableCreator[UserRow, UserTable](UserTable)
        users = creator.build("users", data)
    """

    def __init__(self, table_type: type[TTable]) -> None:
        self.table_type = table_type

    def build(self, name: str, data: npt.ArrayLike) -> TTable:
        return self.table_type.build(name, data)


#====================================================
#               DEMO
#====================================================

if __name__ == "__main__":
    class RowType(NamedTuple):
        kind: np.uint8
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8

    print(to_col_defs([('kind', np.bool), ('bits', np.uint8)]))
    print(_col_defs_from_rowtype(RowType))

    class MyPlain(PlainTable[RowType]):
        @classmethod
        def get_row_type(cls) -> type[RowType]:
            return RowType

    tbl = MyPlain.build("test", [(1, 0, 255, 8), (2, 1, 65535, 16)])
    print("plain ->", tbl.name, tbl.dtype, len(tbl))
    print("row 0 ->", tbl[0])
    print("max  ->", tbl.max)

    creator = TableCreator[RowType, MyPlain](MyPlain)
    tbl2 = creator.build("via_creator", [(1, 0, 255, 8)])
    print("creator ->", tbl2.name, tbl2[0])
