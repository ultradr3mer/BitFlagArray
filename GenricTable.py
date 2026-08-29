from abc import ABC, abstractmethod
from types import UnionType
from typing import NamedTuple, Any, get_type_hints, get_args, get_origin, Union, TypeVar, Type, Generic, Dict, List, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from common import ExceptionRaiser, get_first_or


#====================================================
#               COLUMN DEFINITIONS
#====================================================

class NpColDef(NamedTuple):
    name: str
    type: np.dtype[Any]

    def __repr__(self) -> str:
        return f"(NpCol {self.name}: {self.type})"


def scalar_from_hint(hint: Any) -> Any:
    """Scalar member of a ``Column | scalar`` field hint.

    A shared row/table declaration annotates each field as the union of its
    column container and its scalar cell type. The scalar is the member with a
    concrete numpy dtype; column containers (ndarray subclasses, custom column
    classes) coerce to ``object`` and are skipped. Plain non-union hints pass
    through unchanged.
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
    raise TypeError(f"no scalar member in field hint: {hint!r}")


def get_defs_from_rowtype(row_type: type) -> list[NpColDef]:
    """Derive column definitions from a NamedTuple row type's annotations.

    Fields may be declared once, shared by table and row, as ``Col | scalar``
    unions; the scalar member defines the column dtype.
    """
    hints = get_type_hints(row_type)
    return [NpColDef(name, np.dtype(scalar_from_hint(hints[name]))) for name in row_type._fields]


def dtype_from_fields(fields: list[NpColDef]) -> np.dtype[Any]:
    return np.dtype([(f.name, f.type) for f in fields])


def row_type_from_fields(typename: str, fields_cls: type) -> type:
    """Build a NamedTuple row type from a plain shared fields-class.

    Companion to the shared declaration pattern: the fields-class declares
    each field once as ``Col | scalar``, the table inherits the class so
    the fields exist statically on it, and this derives the row NamedTuple
    from the very same annotations.
    """
    return NamedTuple(typename, get_type_hints(fields_cls).items())


#====================================================
#               COLUMN CONTAINER ADAPTERS
#====================================================

TCell = TypeVar('TCell')
TColContainer = TypeVar('TColContainer')


class TColContainerAdapter(ABC, Generic[TColContainer, TCell]):
    """Adapter: builds a column from a structured array.

    Holds table-level context (parent, name) and produces a column
    container of type ``TColContainer`` from the raw array data.
    """

    def __init__(self, parent: Table, name: str) -> None:
        self.parent = parent
        self.name = name

    @abstractmethod
    def init_column(self, ary: np.ndarray, cell_type: type[TCell]) -> TColContainer:
        """Create the column from the structured array."""

    def __repr__(self) -> str:
        return f'{type(self).__name__}[{self.name}] of {self.parent!r}'


class TColContainerCreator(ABC, Generic[TColContainer]):
    """Factory: creates column adapters for a given container type."""

    @abstractmethod
    def get_adapter(self, parent: Table, name: str) -> TColContainerAdapter[TColContainer, Any]:
        """Create an adapter for the given table and field name."""


class NPColAdapter(TColContainerAdapter[np.ndarray, Any]):
    def init_column(self, ary: np.ndarray, cell_type: type[Any]) -> np.ndarray:
        return ary[self.name]


class NPContainerCreator(TColContainerCreator[np.ndarray]):
    def get_adapter(self, parent: Table, name: str) -> NPColAdapter:
        return NPColAdapter(parent, name)


#====================================================
#               TABLE BASE HIERARCHY
#====================================================

class Table[TRow]:
    adapter: Dict[str, TColContainerAdapter]
    # Concrete families bind their column creator here, so the table type
    # only carries the row type: Table[DRow].
    col_creator_cls: type[TColContainerCreator]

    def __init__(self,
                 name: str,
                 data: npt.ArrayLike,
                 row_type: Type[TRow],
                 col_a_cre: type[TColContainerCreator] | TColContainerCreator | None = None):
        self.name = name
        self.row_type = row_type
        self.fields = get_defs_from_rowtype(row_type)
        self.data = np.array(data, dtype=dtype_from_fields(self.fields))
        if col_a_cre is None:
            col_a_cre = type(self).col_creator_cls
        self.col_creator = col_a_cre() if isinstance(col_a_cre, type) else col_a_cre
        self._cols = self.create_cols_set_attr()
        self._check_col_attrs()

    def create_cols_set_attr(self):
        cols = self.create_columns()
        for c, f in zip(cols, self.fields):
            setattr(self, f.name, c)
        return cols

    def create_columns(self) -> List[TColContainer]:
        adapter_dict: Dict[str, TColContainerAdapter] = {}
        cols = []
        for f in self.fields:
            adapter = self.col_creator.get_adapter(parent=self, name=f.name)
            adapter_dict[f.name] = adapter
            col = adapter.init_column(self.data, f.type)
            cols.append(col)
        self.adapter = adapter_dict
        return cols

    def _check_col_attrs(self) -> None:
        names = [f.name for f in self.fields]
        assert names == list(self.row_type._fields), \
            f"field names mismatch: {names} != {list(self.row_type._fields)}"
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
            yield self.row_type(*rec)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.row_type(*self.data[key])
        return [self.row_type(*d) for d in self.data[key]]

    def __len__(self):
        return len(self.data)

    def rows(self) -> list[Any]:
        if len(self) == 1:
            return self.row_type(*self.data)
        return list([self.row_type(*d) for d in self.data])

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r}, len={len(self)})'




#====================================================
#               NUMPY TABLE
#====================================================


class NpColTable[TRow](Table[TRow]):
    col_creator_cls = NPContainerCreator

    def __init__(self, name: str, data: npt.ArrayLike, row_type: type[TRow]):
        super().__init__(name, data, row_type)

    if TYPE_CHECKING:
        # Column attrs are derived from the row type's shared field
        # declaration; for type checkers they are plain ndarrays.
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
                     row_type=RowType)

    print("plain ->", tbl1.name, tbl1.dtype, len(tbl1))
    print("row 0 ->", tbl1[0])
    print("max  ->", tbl1.max)

    tbl2 = NpColTable(name="via_creator",
                         data=[(1, 0, 255, 8), (1, 0, 255, 8), (2, 1, 65535, 16)],
                         row_type=RowType)

    print("creator ->", tbl2.name, tbl2[0])
