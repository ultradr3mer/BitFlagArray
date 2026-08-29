import operator
from typing import Any, Literal, List, Dict, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from GenericTable import Table
from common import ExceptionRaiser, get_first_or

#====================================================
#               QUERY INFRASTRUCTURE
#====================================================

Undefined = Literal  # sentinel: `column == Undefined` builds an always-true constraint

class Query:
    """Composed, unevaluated condition over table columns.

    Constraints (``col == v``, ``col >= v``, ...) combine via ``&`` / ``|``
    into one query. Nothing touches the table data until ``indices``,
    ``get_first`` or ``get_all`` evaluates it.
    """

    __slots__ = ("left", "right", "combine")

    def __init__(self, left, right, combine):
        self.left = left
        self.right = right
        self.combine = combine

    def __and__(self, other) -> "Query":
        return Query(self, other, np.intersect1d)

    def __or__(self, other) -> "Query":
        return Query(self, other, np.union1d)

    @property
    def indices(self) -> npt.NDArray[np.intp]:
        """Evaluate the whole query; row indices that match."""
        return self.combine(self.left.indices, self.right.indices)

    def evaluate(self) -> npt.NDArray[np.intp]:
        return self.indices


class Constraint(Query):
    """Single unevaluated condition: one comparison on one column."""

    __slots__ = ("column", "op", "value")

    def __init__(self, column, op, value):
        self.column = column
        self.op = op
        self.value = value

    @property
    def indices(self) -> npt.NDArray[np.intp]:
        if self.value is Undefined:  # no constraint -> all rows
            mask = np.ones(len(self.column.column), dtype=bool)
        else:
            mask = self.op(self.column.column, self.value)
        return np.flatnonzero(mask)


class ConstraintColumn:
    """Column of a table; comparisons build lazy Constraints on it."""

    __slots__ = ("table", "name", "column")

    def __init__(self, data: np.ndarray, selector: str | None = None):
        self.table = data
        self.name = selector
        self.column = data if selector is None else data[selector]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.column[key]                          # Item Type
        return ConstraintColumn(self.table[key], self.name)  # Range Type variant

    def __repr__(self) -> str:
        return f"CCol({self.name}): {self.column!r}"

    def __eq__(self, value: object) -> Constraint:  # type: ignore[override]
        return Constraint(self, operator.eq, value)

    def __ne__(self, value: object) -> Constraint:  # type: ignore[override]
        return Constraint(self, operator.ne, value)

    def __lt__(self, value: object) -> Constraint:
        return Constraint(self, operator.lt, value)

    def __le__(self, value: object) -> Constraint:
        return Constraint(self, operator.le, value)

    def __gt__(self, value: object) -> Constraint:
        return Constraint(self, operator.gt, value)

    def __ge__(self, value: object) -> Constraint:
        return Constraint(self, operator.ge, value)

    def __hash__(self) -> int:
        return id(self)

    def get_all(self, query: Query) -> np.ndarray:
        """Values of this column for all rows matching the query."""
        return self.column[query.indices]

    def get_first(self, query: Query) -> Any:
        """Value of this column for the first row matching the query."""
        return get_first_or(self.get_all(query), KeyError())


CCol = ConstraintColumn


#====================================================
#               QUERYABLE TABLE
#====================================================

class QueryableTable[TTable](Table[TTable]):
    """Table whose fields are lazy queryable ConstraintColumns."""

    default_range = ConstraintColumn

    def build_column(self, data: np.ndarray, name: str) -> ConstraintColumn:
        return ConstraintColumn(data, name)

    def get_all(self, query: Query):
        """Range Type variant of the table matching the query."""
        return self[query.indices]

    def get_first(self, query: Query):
        """Item Type variant: first row matching the query."""
        first = get_first_or(query.indices, KeyError())
        return self.table_type(*self.data[first])

    if TYPE_CHECKING:
        # Column attrs are derived from the table type's shared field
        # declaration; for type checkers they are constraint columns.
        def __getattr__(self, name: str) -> CCol: ...


#====================================================
#               WITH SPECIAL COL
#====================================================

type SpecialColumnType = Any
TContainer = npt.NDArray[SpecialColumnType]|List[SpecialColumnType]

class QTblSpecialCol[TTable, SpecialColumnType](QueryableTable[TTable]):
    result_dict: Dict[str, type]
    def __init__(self, name: str, data: npt.ArrayLike, result_dict: TContainer, table_type: type[TTable]):
        super().__init__(name, data, table_type)
        self.ref_column = result_dict

    def get_all(self, key) -> npt.NDArray[SpecialColumnType] | SpecialColumnType:
        if isinstance(key, Query):
            key = key.indices
        return self.ref_column[key]

    def get_first(self, key) -> SpecialColumnType:
        all_items = self.get_all(key)
        return get_first_or(all_items, KeyError())

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
        signed: np.bool_
        abs_min: np.uint64
        max: np.uint64
        bits: np.uint8

    qtbl = QueryableTable(name="TypeLookup", data=build_type_table(), table_type=TypeLookupRow)
    print("direct  ->", qtbl.name, qtbl.dtype, "len:", len(qtbl))
    print("item    ->", qtbl[0])
    print("range   ->", qtbl[4:6])

    q = qtbl.signed == False
    print("signed == False ->", q.indices)

    val = 255
    smallest = (qtbl.signed == False) & (qtbl.max >= val)
    print("get_first ->", qtbl.get_first(smallest))
    print("get_all max ->", qtbl.get_all(smallest).max.column)

    # every column can select values; queries compose before they evaluate
    print("max.get_first   ->", qtbl.max.get_first(qtbl.signed == True))
    print("or-query item   ->", qtbl.get_first((qtbl.bits <= 8) | (qtbl.max >= 2**32)))

    signed_rows = [t for t in qtbl if t.signed == True]
    print("signed rows:", len(signed_rows))
