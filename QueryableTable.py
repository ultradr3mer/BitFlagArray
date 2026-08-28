from dataclasses import dataclass
from typing import TypeVar, Generic, Type, List, Iterable, Any, NamedTuple, overload, Literal, TYPE_CHECKING, Tuple

import numpy as np
import numpy.typing as npt


# ====================================================
#                   QUERYABLE TABLE
# ====================================================

T_constr_item = TypeVar('T_constr_item', bound=Any)

Undefined = Literal

# Stack for the `and`-trick: `bool(selection)` (called by `and`) pushes the
# selection here; the next comparison on a column pops it and applies it as a
# restriction. Enables: `selection and column >= value`.
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

    def __bool__(self) -> bool:
        # Called by `and`. Stash self so the following column comparison can
        # pick it up as a restriction. Returning True makes `and` evaluate and
        # return its right-hand side.
        _pending_selections.append(self)
        return True

import operator

class ConstraintColumn(Generic[T_constr_item]):
    def __init__(
        self,
        data: npt.NDArray[T_constr_item],
        selector: str | None = None,
        *,
        _column: npt.NDArray[T_constr_item] | None = None,
        _parent_indices: npt.NDArray[np.intp] | None = None,
    ):
        if _column is not None:
            self.column = _column
            self.table = data
            self.parent_indices = _parent_indices
            return

        if selector is None:
            self.column = data
        else:
            self.column = data[selector]
        self.table = data
        self.parent_indices = None

    def _resolve(self, local_indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        if self.parent_indices is None:
            return local_indices
        return self.parent_indices[local_indices]

    def __getitem__(self, key):
        if isinstance(key, ConstraintSelection):
            return ConstraintColumn(
                self.table,
                _column=self.column[key.indices],
                _parent_indices=key.indices,
            )
        return self.column[key]

    def _compare(self, value: object, op) -> ConstraintSelection:
        # `and`-trick: if a pending selection was stashed by `__bool__`,
        # restrict this column to it before comparing.
        col = self
        if _pending_selections:
            col = col[_pending_selections.pop()]

        if value is Undefined:
            mask = np.ones(len(col.column), dtype=bool)
        else:
            mask = op(col.column, value)

        local = np.flatnonzero(mask)
        return ConstraintSelection(col._resolve(local))

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

# dtype spec: list of (name, dtype) tuples.
TFieldSpec = list[tuple[str, Any]]

# Module-level cache for generated row NamedTuples per LookupTable subclass.
_row_types: dict[type, type] = {}


class QueryableTable:
    """Base for dtype-first lookup tables.

    Subclass via::

        class MyTable(LookupTable[[
            ('signed', np.bool),
            ('max',    np.uint64),
        ]]):
            pass
    """

    #Inherit Generic table

    def __init__(self):
        self._array = None

    @classmethod
    def _row_type(cls) -> type:
        cached = _row_types.get(cls)
        #TODO
        pass

    def __iter__(self):
        # TODO
        pass

    def __getitem__(self, key):
        # TODO
        pass

    def __len__(self):
        # TODO
        pass


