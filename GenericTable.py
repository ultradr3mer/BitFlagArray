from types import UnionType
from typing import NamedTuple, Any, get_type_hints, get_args, get_origin, Union, List, TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np
import numpy.typing as npt

from common import ExceptionRaiser, get_first_or


#====================================================
#           TABLE TYPE DEFINITIONS
#====================================================

class TableFields:
    """Marker base for a shared ``Range | Item`` table type declaration."""


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
    return [NpColDef(name, np.dtype(item_type_of(hints[name]))) for name in hints]


def dtype_from_fields(fields: list[NpColDef]) -> np.dtype[Any]:
    return np.dtype([(f.name, f.type) for f in fields])


_item_variant_cache: "WeakKeyDictionary[type, type]" = WeakKeyDictionary()


def table_item_from_fields(fields_cls: type) -> type:
    """Item variant of a TableFields declaration: scalar members only.

    Cached per declaration class, so every table of that declaration
    shares one stable item type.
    """
    item = _item_variant_cache.get(fields_cls)
    if item is None:
        hints = get_type_hints(fields_cls)
        typename = fields_cls.__name__.removesuffix('Fields')
        item = NamedTuple(typename, [(name, item_type_of(h)) for name, h in hints.items()])
        _item_variant_cache[fields_cls] = item
    return item


def table_range_from_type(table_type: type, default_range: type = np.ndarray,
                          basename: str | None = None) -> type:
    """Range variant of a table type: one column (range) per field.

    Fields that declare no range member fall back to ``default_range``
    (the family's column type).
    """
    hints = get_type_hints(table_type)
    fields = [(name, range_type_of(hints[name]) or default_range) for name in hints]
    return NamedTuple(f"{basename or table_type.__name__}Range", fields)


#====================================================
#               TABLE BASE HIERARCHY
#====================================================

class Table[TRowNRange]:
    """Structured table driven by its table type.

    The table type is a ``TableFields`` declaration (each field once as
    ``Range | Item``) or a legacy NamedTuple passed as ``table_type``.
    The table exposes the range members as columns, ``table[i]`` returns
    the Item variant (a row), ``table[i:j]`` the Range variant (columns
    over the selection). Both variants are derived inside the framework.
    """

    default_range: type = np.ndarray  # column type for fields declaring no range member

    def __init__(self,
                 name: str,
                 data: npt.ArrayLike,
                 table_type: type | None = None):
        if table_type is None:
            table_type = self._find_table_fields()
        self.name = name
        self.table_type = table_type
        self.fields = get_defs_from_table_type(table_type)
        self.data = np.array(data, dtype=dtype_from_fields(self.fields))
        if isinstance(table_type, type) and issubclass(table_type, TableFields):
            self.item_type = table_item_from_fields(table_type)   # declaration: derive the row tuple
        else:
            self.item_type = table_type                            # legacy NamedTuple is the item itself
        self.range_type = table_range_from_type(table_type, type(self).default_range,
                                                basename=self.item_type.__name__)
        self._cols = self.create_cols_set_attr()
        self._check_col_attrs()

    def _find_table_fields(self) -> type:
        """Most-derived TableFields declaration in this table's MRO."""
        for cls in type(self).__mro__:
            # skip the framework classes (the table itself inherits the
            # declaration, so it also matches the subclass check)
            if cls is not TableFields and issubclass(cls, TableFields) and not issubclass(cls, Table):
                return cls
        raise TypeError(
            f"{type(self).__name__} needs a table type: pass table_type or "
            f"inherit a TableFields declaration")

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
        assert names == list(self.item_type._fields), \
            f"field names mismatch: {names} != {list(self.item_type._fields)}"
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
            yield self.item_type(*rec)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.item_type(*self.data[key])         # Item Type variant
        return self.range_variant(self.data[key])          # Range Type variant

    def range_variant(self, sub_data: np.ndarray):
        """Range Type variant of the table over the selected (sub-)data."""
        return self.range_type(*[self.build_column(sub_data, f.name) for f in self.fields])

    def __len__(self):
        return len(self.data)

    def rows(self) -> list[Any]:
        """All rows as Item Type variants."""
        return [self.item_type(*rec) for rec in self.data]

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r}, len={len(self)})'


#====================================================
#               NUMPY TABLE
#====================================================

class NpColTable[TRowNRange](Table[TRowNRange]):
    """Plain numpy table: fields are ndarray columns."""

    def __init__(self, name: str, data: npt.ArrayLike, table_type: type[TRowNRange]):
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
