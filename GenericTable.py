from types import UnionType
from typing import NamedTuple, Any, get_type_hints, get_args, get_origin, Union, Type, List, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from common import ExceptionRaiser, get_first_or


#====================================================
#           TABLE TYPE DEFINITIONS
#====================================================

class NpColDef(NamedTuple):
    name: str
    type: np.dtype[Any]

    def __repr__(self) -> str:
        return f"(NpCol {self.name}: {self.type})"


def item_type_of(hint: Any) -> Any:
    """Item type of a ``Range | Item`` field hint.

    A table type declares each field as the union of its range (column
    container) and its item (scalar cell) type. The item is the member with
    a concrete numpy dtype; ranges (ndarray subclasses, custom column
    classes) coerce to ``object`` and are skipped. Plain non-union hints
    pass through unchanged.
    """
    if get_origin(hint) is not Union and not isinstance(hint, UnionType):
        return hint
    for member in get_args(hint):
        try:
            member_dtype = np.dtype(member)
        except (TypeError, ValueError):
            continue
        if member_dtype != object:
            return member
    raise TypeError(f"no item type in field hint: {hint!r}")


def range_type_of(hint: Any) -> Any | None:
    """Declared range (column) member of a ``Range | Item`` hint; None if absent."""
    if get_origin(hint) is not Union and not isinstance(hint, UnionType):
        return None
    item = item_type_of(hint)
    for member in get_args(hint):
        if member is not item:
            return member
    return None


def get_defs_from_table_type(table_type: type) -> list[NpColDef]:
    """Derive column definitions from a table type's annotations.

    Each field may be declared once, for table and items alike, as
    ``Range | Item``; the item type defines the column dtype.
    """
    hints = get_type_hints(table_type)
    return [NpColDef(name, np.dtype(item_type_of(hints[name]))) for name in table_type._fields]


def dtype_from_fields(fields: list[NpColDef]) -> np.dtype[Any]:
    return np.dtype([(f.name, f.type) for f in fields])


def table_type_from_fields(typename: str, fields_cls: type) -> type:
    """Build a NamedTuple table type from a plain shared fields-class.

    Companion to the shared declaration pattern: the fields-class declares
    each field once as ``Range | Item``, the table inherits the class so
    the fields exist statically on it, and this derives the table type
    (the Item variant) from the very same annotations.
    """
    return NamedTuple(typename, get_type_hints(fields_cls).items())


def table_range_from_type(table_type: type, default_range: type = np.ndarray) -> type:
    """Range variant of a table type: one column (range) per field.

    Fields that declare no range member fall back to ``default_range``
    (the family's column type).
    """
    hints = get_type_hints(table_type)
    fields = [(name, range_type_of(hints[name]) or default_range) for name in table_type._fields]
    return NamedTuple(f"{table_type.__name__}Range", fields)


#====================================================
#               TABLE BASE HIERARCHY
#====================================================

class Table[TTable]:
    """Structured table driven by its table type.

    The table type (a NamedTuple) declares each field once as
    ``Range | Item``: the table exposes the range members as columns,
    ``table[i]`` returns the Item variant (a row), ``table[i:j]`` the
    Range variant (columns over the selection). Column types are
    specified by the table type - families only provide ``build_column``.
    """

    default_range: type = np.ndarray  # column type for fields declaring no range member

    def __init__(self,
                 name: str,
                 data: npt.ArrayLike,
                 table_type: Type[TTable]):
        self.name = name
        self.table_type = table_type
        self.fields = get_defs_from_table_type(table_type)
        self.data = np.array(data, dtype=dtype_from_fields(self.fields))
        self.range_type = table_range_from_type(table_type, type(self).default_range)
        self._cols = self.create_cols_set_attr()
        self._check_col_attrs()

    def build_column(self, data: np.ndarray, name: str):
        """Range member for one field over the given (sub-)data. Families override."""
        return data[name]

    def create_cols_set_attr(self):
        cols = []
        declared_hints = get_type_hints(self.table_type)
        for f in self.fields:
            col = self.build_column(self.data, f.name)
            declared = range_type_of(declared_hints[f.name])
            if isinstance(declared, type) and not isinstance(col, declared):
                raise TypeError(
                    f"column {f.name!r}: table type declares range {declared.__name__}, "
                    f"got {type(col).__name__}")
            cols.append(col)
            setattr(self, f.name, col)
        return cols

    def _check_col_attrs(self) -> None:
        names = [f.name for f in self.fields]
        assert names == list(self.table_type._fields), \
            f"field names mismatch: {names} != {list(self.table_type._fields)}"
        for i, name in enumerate(names):
            assert getattr(self, name, None) is self._cols[i], \
                f"column attr {name!r} does not hold its column (index {i})"

    @property
    def cols(self) -> List[Any]:
        return self._cols

    @cols.setter
    def cols(self, value):
        self._cols = value

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def _field_names(self) -> list[str]:
        return list([n for n, t in self.fields])

    def __iter__(self):
        for rec in self.data:
            yield self.table_type(*rec)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.table_type(*self.data[key])      # Item Type variant
        return self.range_variant(self.data[key])         # Range Type variant

    def range_variant(self, sub_data: np.ndarray):
        """Range Type variant of the table over the selected (sub-)data."""
        return self.range_type(*[self.build_column(sub_data, f.name) for f in self.fields])

    def __len__(self):
        return len(self.data)

    def rows(self) -> list[Any]:
        """All rows as Item Type variants."""
        return [self.table_type(*rec) for rec in self.data]

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r}, len={len(self)})'


#====================================================
#               NUMPY TABLE
#====================================================

class NpColTable[TTable](Table[TTable]):
    """Plain numpy table: fields are ndarray columns."""

    def __init__(self, name: str, data: npt.ArrayLike, table_type: type[TTable]):
        super().__init__(name, data, table_type)

    if TYPE_CHECKING:
        # Column attrs come from the table type's shared field declaration;
        # for type checkers they are plain ndarrays.
        def __getattr__(self, name: str) -> np.ndarray: ...

#====================================================
#               DEMO
#====================================================

if __name__ == "__main__":
    class RowType(NamedTuple):
        kind: np.uint8
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8

    tbl1 = NpColTable(name="test",
                      data=[(1, 0, 255, 8), (2, 1, 65535, 16)],
                      table_type=RowType)

    print("plain ->", tbl1.name, tbl1.dtype, len(tbl1))
    print("item  ->", tbl1[0])
    print("range ->", tbl1[0:1])
    print("max   ->", tbl1.max)

    tbl2 = NpColTable(name="second",
                      data=[(1, 0, 255, 8), (1, 0, 255, 8), (2, 1, 65535, 16)],
                      table_type=RowType)
    print("second ->", tbl2.name, tbl2[0], "rows:", len(tbl2.rows()))
