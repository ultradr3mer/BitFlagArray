from abc import ABC, abstractmethod
from typing import NamedTuple, Any, get_type_hints, TypeVar, Type, Generic, Dict, List

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


def get_defs_from_rowtype(row_type: type) -> list[NpColDef]:
    """Derive column definitions from a NamedTuple row type's annotations."""
    hints = get_type_hints(row_type)
    return [NpColDef(name, np.dtype(hints[name])) for name in row_type._fields]


def dtype_from_fields(fields: list[NpColDef]) -> np.dtype[Any]:
    return np.dtype([(f.name, f.type) for f in fields])


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

class Table[TRow, TCreator: TColContainerCreator]:
    adapter: Dict[str, TColContainerAdapter]

    def __init__(self,
                 name: str,
                 data: npt.ArrayLike,
                 row_type: Type[TRow],
                 col_a_cre: TCreator):
        self.name = name
        self.row_type = row_type
        self.fields = get_defs_from_rowtype(row_type)
        self.data = np.array(data, dtype=dtype_from_fields(self.fields))
        self.col_creator = col_a_cre
        self._cols = self.create_cols_set_attr()

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


class NpColTable[TRow](Table[TRow, NPContainerCreator]):
    def __init__(self, name: str, data: npt.ArrayLike, row_type: type[TRow]):
        super().__init__(name, data, row_type, col_a_cre=NPContainerCreator())

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
