import operator
from dataclasses import dataclass
from typing import Any, Literal, List, Dict

import numpy as np
import numpy.typing as npt

from GenricTable import Table, TColContainerAdapter, TColContainerCreator
from common import ExceptionRaiser, get_first_or

#====================================================
#               QUERY INFRASTRUCTURE
#====================================================

Undefined = Literal

_pending_selections: list["ConstraintSelection"] = []


@dataclass(frozen=True)
class ConstraintSelection:
    indices: npt.NDArray[np.intp]

    def __and__(self, other):
        if isinstance(other, ConstraintSelection):
            return ConstraintSelection(
                np.intersect1d(self.indices, other.indices)
            )
        if isinstance(other, ConstraintColumn):
            return other[self]
        return NotImplemented


class ConstraintColumn:
    __slots__ = ("table", "column", "parent_indices")

    def __init__(
        self,
        data: np.ndarray,
        selector: str | None = None,
        *,
        _column: np.ndarray | None = None,
        _parent_indices: npt.NDArray[np.intp] | None = None,
    ):
        if _column is not None:
            self.column = _column
            self.table = data
            self.parent_indices = _parent_indices
            return
        self.column = data if selector is None else data[selector]
        self.table = data
        self.parent_indices = None

    def _resolve(self, local_indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        return local_indices if self.parent_indices is None else self.parent_indices[local_indices]

    def __getitem__(self, key):
        if isinstance(key, ConstraintSelection):
            return ConstraintColumn(
                self.table,
                _column=self.column[key.indices],
                _parent_indices=key.indices,
            )
        return self.column[key]

    def _compare(self, value: object, op) -> ConstraintSelection:
        col = self
        if _pending_selections:
            col = col[_pending_selections.pop()]
        if value is Undefined:
            mask = np.ones(len(col.column), dtype=bool)
        else:
            mask = op(col.column, value)
        return ConstraintSelection(col._resolve(np.flatnonzero(mask)))

    def __eq__(self, value: object) -> ConstraintSelection:  # type: ignore[override]
        return self._compare(value, operator.eq)

    def __ne__(self, value: object) -> ConstraintSelection:  # type: ignore[override]
        return self._compare(value, operator.ne)

    def __lt__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.lt)

    def __le__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.le)

    def __gt__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.gt)

    def __ge__(self, value: object) -> ConstraintSelection:
        return self._compare(value, operator.ge)

    def __hash__(self) -> int:
        return id(self)


CCol = ConstraintColumn


#====================================================
#               CONSTRAINT COLUMN ADAPTER
#====================================================

class ConstraintColAdapter(TColContainerAdapter[ConstraintColumn, Any]):
    """Adapter for ConstraintColumn columns."""

    def init_column(self, ary: np.ndarray, cell_type: type[Any]) -> ConstraintColumn:
        return ConstraintColumn(ary, self.name)


class ConstraintColCreator(TColContainerCreator[ConstraintColumn]):
    """Factory for ConstraintColumn column adapters."""

    def get_adapter(self, parent: Table, name: str) -> ConstraintColAdapter:
        return ConstraintColAdapter(parent, name)


#====================================================
#               QUERYABLE TABLE
#====================================================

class QueryableTable[TRow](Table[TRow, ConstraintColCreator]):
    def __init__(self, name: str, data: npt.ArrayLike, row_type: type[TRow]):
        super().__init__(name, data, row_type, col_a_cre=ConstraintColCreator())


#====================================================
#               WITH SPECIAL COL
#====================================================

type SpecialColumnType = Any
TContainer = npt.NDArray[SpecialColumnType]|List[SpecialColumnType]

class QTblSpecialCol[TRow, SpecialColumnType](QueryableTable):
    not_found_except: ExceptionRaiser[SpecialColumnType] = ExceptionRaiser(KeyError, "Not Found")
    result_dict: Dict[str, type]
    def __init__(self, name: str, data: npt.ArrayLike, result_dict: TContainer, row_type: type[TRow]):
        super().__init__(name, data, row_type)
        self.ref_column = result_dict

    def get_first(self, key) -> SpecialColumnType:
        all_items = self.get_all(key)
        return get_first_or(all_items, self.not_found_except.do_raise())

    def get_all(self, key) -> npt.NDArray[SpecialColumnType] | SpecialColumnType:
        return self.ref_column[key]

#====================================================
#               DEMO
#====================================================
def build_type_table():
    kind = {'u': False, 'i': True}
    sizes = [1, 2, 4, 8]
    types = [np.dtype(f"{k}{s}") for k in kind for s in sizes]
    rows = []
    for t in types:
        s = 1 if t.kind == 'i' else -1
        rows.append((kind[t.kind], -np.iinfo(t).min, np.iinfo(t).max, np.iinfo(t).bits))
    return rows

if __name__ == "__main__":
    from typing import NamedTuple

    class TypeLookupRow(NamedTuple):
        signed: np.bool
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8

    qtbl = QueryableTable(name="TypeLookup",data=build_type_table(), row_type=TypeLookupRow)
    print("direct  ->", qtbl.name, qtbl.dtype, "len:", len(qtbl))
    print("row 0:", qtbl[0])

    sel = qtbl.signed == False
    print("signed == False ->", sel)

    val = 255
    smallest = (
        qtbl.signed == False and qtbl.max >= val
    )
    print("smallest unsigned for 123 ->", smallest)

    smallest_alt = (
        (((qtbl.signed == False) & qtbl.max) >= val) & qtbl.prev_max
    ) < val
    print("pipeline ->", smallest_alt)

    signed_rows = [t for t in qtbl if t.signed == True]
    print("signed rows:", len(signed_rows))
