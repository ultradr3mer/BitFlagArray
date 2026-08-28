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