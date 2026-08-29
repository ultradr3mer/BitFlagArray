#====================================================
#               ONLY DEPRECATED STUFF
#====================================================


# from typing import TypeVar, NamedTuple, Generic, Any
# import numpy as np
# import numpy.typing as npt
#
#
# T_cols = TypeVar('T_cols', bound=Any)
#
#
#
# T_col_ary_item = T_cols # TypeVar('T_col_ary_item', bound=Any)
# type T_cols_ary[T_cols] = npt.NDArray[T_cols]
# T_col_scl_item = T_cols # TypeVar('T_col_scl_item', bound=np.ScalarType)
# type T_cols_item[T_col_scl_item] = T_col_scl_item
#
# type T_cols_ary_or_item[T_cols] = T_cols_ary[T_cols] | T_cols_item[T_cols]
#
# tbl_bool = TypeVar('tbl_bool', bound=T_cols_ary_or_item[np.bool])
# tbl_uint64 = TypeVar('tbl_uint64', bound=T_cols_ary_or_item[np.uint64])
# tbl_uint8 = TypeVar('tbl_uint8', bound=T_cols_ary_or_item[np.uint8])
#
# # T_contain = TypeVar('T_contain', bound=npt.NDArray |np.ScalarType)
# type T_ary = npt.NDArray
# type T_cols_of_container[T_contain, T_cols] = T_contain[T_cols]
# # def try1():
#
# class TblAndColumn(NamedTuple, Generic[tbl_bool,tbl_uint64,tbl_uint8]):
#     kind: tbl_bool
#     abs_min: tbl_uint64
#     max: tbl_uint64
#     bits: tbl_uint8
#
# class MyTable(TblAndColumn[npt.NDArray[np.bool],npt.NDArray[np.uint64],npt.NDArray[np.uint8]]):
#     pass
#
# class MyRow(TblAndColumn[np.bool,np.uint64,np.uint8]):
#     pass
#
#
# test_tbl = MyTable(kind=np.array([True]), abs_min=np.array([1]), max=np.array([1]), bits=np.array([1]))
#
# print(test_tbl)
#
# test_row = MyRow(kind=True, abs_min=1, max=1, bits=1)
#
# print(test_row)
#
# type FieldSpec = list[tuple[str, npt.DTypeLike]]
#
#
# def to_col_defs(spec: FieldSpec) -> list[NpColDef]:
#     """Convert a raw (name, dtype) field-spec into a list of NpColDef."""
#     return [NpColDef(name, np.dtype(dt)) for name, dt in spec]



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

