# BitFlagArray

A Python library for **packed bit-level storage** in NumPy arrays with a
**typed, queryable table layer** for structured data.

The project has three pillars:

| Module              | Role                                                            |
|---------------------|-----------------------------------------------------------------|
| **`BitFlagArray`**  | Pack, unpack, slice and mask individual bits inside integers.  |
| **`GenricTable`**   | Generic typed table framework — define a row, get a table.     |
| **`QueryableTable`**| Queryable tables with constraint-column selection pipelines.   |

---

## Quick start

```bash
pip install numpy
```

```python
import numpy as np
from typing import NamedTuple
from QueryableTable import QueryableTable

class TypeLookupRow(NamedTuple):
    signed:   np.bool
    abs_min:  np.uint64
    max:      np.uint64
    bits:     np.uint8
    prev_max: np.int64

class TypeLookup(QueryableTable[TypeLookupRow]):
    @classmethod
    def get_row_type(cls) -> type[TypeLookupRow]:
        return TypeLookupRow

tbl = TypeLookup.build("TypeLookup", [
    (False, 0, 255,                8, -1),
    (False, 0, 65535,              16, 255),
    (True,  0, 127,                 8, -1),
    (True,  0, 32767,              16, 127),
])

# Query: smallest unsigned type that can hold 123
val = 123
smallest = (
    tbl.signed == False
    and tbl.max >= val
    and tbl.prev_max < val
)
print(smallest)  # ConstraintSelection(indices=array([0]))
```

---

## GenricTable — the generic table framework

### The core idea

You define a **row** as a `NamedTuple`. Everything else — the NumPy structured
dtype, the column array, iteration, indexing — is derived automatically.

```python
from typing import NamedTuple
from GenricTable import PlainTable

class UserRow(NamedTuple):
    id:    np.uint32
    name:  np.str_       # any numpy scalar type works
    score: np.uint64

class UserTable(PlainTable[UserRow]):
    @classmethod
    def get_row_type(cls) -> type[UserRow]:
        return UserRow

users = UserTable.build("users", [
    (1, "Alice", 95),
    (2, "Bob",   42),
    (3, "Carol", 87),
])
```

### What goes in

`build(name, data)` accepts **anything `np.asarray` can convert**:

```python
# List of tuples (most common)
UserTable.build("u", [(1, "A", 95), (2, "B", 42)])

# List of lists
UserTable.build("u", [[1, "A", 95], [2, "B", 42]])

# Existing NumPy structured array
UserTable.build("u", np.array([(1, "A", 95)], dtype=...))

# Another table's array
UserTable.build("u", other_users._ary)
```

The dtype is derived from the row type's annotations:

```python
UserRow.__annotations__
# {'id': numpy.uint32, 'name': numpy.str_, 'score': numpy.uint64}

UserTable.dtype
# dtype([('id', '<u4'), ('name', '<U'), ('score', '<u8')])
```

### How you access it

A `PlainTable` exposes each field as a **plain NumPy array column**:

```python
users.id     # array([1, 2, 3], dtype=uint32)
users.score  # array([95, 42, 87], dtype=uint64)
users.name   # array(['Alice', 'Bob', 'Carol'], dtype='<U5')
```

Iterate over rows:

```python
for row in users:
    print(row.id, row.name, row.score)
# 1 Alice 95
# 2 Bob 42
# 3 Carol 87
```

Index a single row:

```python
users[0]  # UserRow(id=1, name='Alice', score=95)
```

Get all rows as a list:

```python
users.rows()  # [UserRow(1,'Alice',95), UserRow(2,'Bob',42), ...]
```

Length, dtype, repr:

```python
len(users)        # 3
users.dtype       # structured dtype
users.name        # "users"
repr(users)       # 'UserTable(name='users', len=3)'
```

### The class hierarchy

```
Table[TRow, TCreator]  (ABC)              ← common: build(), __iter__, __getitem__, dtype
├── PlainTable[TRow]                      ← columns = np.ndarray
└── QueryableTable[TRow]  (in QueryableTable.py)
                                          ← columns = ConstraintColumn
```

- `Table[TRow, TCreator]` — abstract base. Holds the structured array,
  derives dtype from the row type, provides `build()`, `__iter__`,
  `__getitem__`, `__len__`, `dtype`.
- `PlainTable[TRow]` — columns are plain `np.ndarray` slices.
- `QueryableTable[TRow]` — columns are `ConstraintColumn`s (see below).

### The column-adapter pattern

Column construction is formalized through a creator/adapter pair so that
adding a new column type only requires two small classes:

```
TColContainerCreator           (abstract factory)
    └── get_adapter(parent, name) -> TColContainerAdapter

TColContainerAdapter          (abstract adapter)
    └── init_column(ary, cell_type) -> column
```

| Creator               | Adapter          | Column type       |
|-----------------------|------------------|-------------------|
| `NPContainerCreator` | `NPColAdapter`   | `np.ndarray`      |
| `ConstraintColCreator`| `ConstraintColAdapter` | `ConstraintColumn` |

Each intermediate base (`PlainTable` / `QueryableTable`) sets
`_col_creator_cls` to its creator. `Table.create_columns()` uses that
class to build every field's column.

### TableCreator — the generic factory

`TableCreator` binds a table type and its row type so the return type
is known to type checkers:

```python
from GenricTable import TableCreator

creator = TableCreator[UserRow, UserTable](UserTable)
users   = creator.build("users", data)  # type: UserTable
```

---

## QueryableTable — constraint columns and selection pipelines

### What changes vs PlainTable

A `QueryableTable` exposes each field as a **`ConstraintColumn`** instead
of a raw `np.ndarray`. A `ConstraintColumn` supports comparison operators
that return a **`ConstraintSelection`** — a set of row indices that match
the constraint.

```python
from QueryableTable import QueryableTable, ConstraintColumn, ConstraintSelection

class TypeLookup(QueryableTable[TypeLookupRow]):
    @classmethod
    def get_row_type(cls) -> type[TypeLookupRow]:
        return TypeLookupRow

tbl = TypeLookup.build("TypeLookup", data)

tbl.signed            # ConstraintColumn
tbl.signed == False   # ConstraintSelection(indices=array([0,1,2,3]))
tbl.max >= 1000       # ConstraintSelection(indices=array([...]))
```

### Selection pipelines

The power of `ConstraintSelection` is **chaining**. Two equivalent forms:

**`and`-form** (idiomatic, no parens needed because `and` binds looser
than comparisons):

```python
smallest = (
    tbl.signed == False
    and tbl.max >= 123
    and tbl.prev_max < 123
)
# ConstraintSelection(indices=array([0]))
```

**`&`-form** (explicit, parens required because `&` binds tighter):

```python
smallest = (
    (((tbl.signed == False) & tbl.max) >= 123) & tbl.prev_max
) < 123
```

Both produce the same `ConstraintSelection`.

### How the pipeline works

```
signed == False          →  ConstraintSelection([0,1,2,3])   # step 1: select
& max                    →  restrict `max` to those rows     # step 2: restrict
>= 123                   →  ConstraintSelection([0,1,2])     # step 3: refine
& prev_max               →  restrict `prev_max` to those     # step 4: restrict
< 123                    →  ConstraintSelection([0])         # step 5: refine
```

Internally:

1. `column == value` produces a `ConstraintSelection` (global row indices).
2. `selection & column` restricts the column to those indices
   (`column[selection.indices]`) and returns a **restricted column** that
   remembers its parent indices.
3. The next comparison on the restricted column maps its local mask back to
   global indices (`parent_indices[local_mask]`).

### The `and`-trick

Python's `and` doesn't call `__and__`. It evaluates `bool(left)` and
returns the right side if truthy. `ConstraintSelection.__bool__` pushes
`self` onto a pending stack and returns `True`; the next column
comparison pops it and applies it as a restriction. This makes the
natural `and`-chain work without parens.

**Limitations of the `and`-trick:**

- `is` is not overloadable — use `== Undefined` for "any value".
- The right side of `and` must be a comparison (`>=`, `<`, ...). A bare
  `selection and column` (no comparison) leaves a dangling stack entry.
- Not thread-safe (module-global pending stack).

### Selection operators

| Operator | Meaning                              |
|----------|--------------------------------------|
| `==`     | equal                                |
| `!=`     | not equal                            |
| `<`      | less than                            |
| `<=`     | less than or equal                   |
| `>`      | greater than                         |
| `>=`     | greater than or equal                |
| `&`      | selection ∩ selection (intersection) |
| `&`      | selection & column    (restriction)   |
| `and`    | same as `&` (via the trick)          |

### Using a selection

Once you have a `ConstraintSelection`, use `.indices` to index any
column or array:

```python
sel = tbl.signed == False and tbl.max >= 1000

tbl.max[sel.indices]           # the matching max values
tbl._ary[sel.indices]         # full structured records
tbl[sel.indices[0]]            # first matching row
```

### Filtering rows via iteration

```python
signed_rows = [row for row in tbl if row.signed == True]
```

Each `row` is a `TypeLookupRow` NamedTuple with typed scalar fields.

---

## BitFlagArray — packed bit storage

`BitFlagArray` provides efficient storage and manipulation of individual
**bits** within integers. It treats each integer in a NumPy array as a
container of flag bits and supports slicing, masking, and reconstructing
at the bit level.

Typical use case: store a set of boolean flags compactly inside a single
integer column, then query and modify individual flags.

```python
from BitFlagArray import NBitAryOnly

# 8-bit value with bits 0..7
val = NBitAryOnly(0b10110010, bit_count=8)

# Read bits 2..5
slice_2_5 = val[2:6]  # the 4 bits at positions 2,3,4,5
```

`BitFlagArray` integrates with the type-detection utilities in
`common.py` (`get_type_for_bit_count`, `get_type_for_scalar`,
`get_as_signed`, `get_as_unsigned`) which use the lookup-table pattern
to pick the smallest NumPy dtype that can hold a given value or bit
count.

---

## Module reference

### `GenricTable.py`

| Name                    | Kind     | Description                                     |
|-------------------------|----------|-------------------------------------------------|
| `NpColDef`              | NamedTuple | `(name, type)` column definition              |
| `FieldSpec`             | type alias | `list[tuple[str, DTypeLike]]`                |
| `to_col_defs(spec)`     | function | Convert a field-spec to `list[NpColDef]`        |
| `TColContainerAdapter`  | ABC      | Adapter base: `init_column(ary, cell_type)`     |
| `TColContainerCreator`  | ABC      | Factory base: `get_adapter(parent, name)`       |
| `NPColAdapter`          | class    | Adapter for `np.ndarray` columns                |
| `NPContainerCreator`    | class    | Factory for numpy columns                       |
| `Table[TRow, TCreator]`| ABC      | Abstract table base                             |
| `PlainTable[TRow]`      | class    | Table with `np.ndarray` columns                  |
| `TableCreator[TRow, TTable]` | class | Generic table factory                        |

### `QueryableTable.py`

| Name                    | Kind     | Description                                     |
|-------------------------|----------|-------------------------------------------------|
| `ConstraintSelection`   | dataclass | `indices` + `&` + `and`-trick                 |
| `ConstraintColumn`      | class    | Column with comparison operators → selection    |
| `ConstraintColAdapter`  | class    | Adapter for `ConstraintColumn`                  |
| `ConstraintColCreator`  | class    | Factory for constraint columns                  |
| `QueryableTable[TRow]`  | class    | Table with `ConstraintColumn` columns           |

---

## Design philosophy

1. **Row-first**: define the row, get the table. No manual dtype lists.
2. **One source of truth**: the `NamedTuple` row type's annotations drive
   the dtype, the column names, and the iteration shape.
3. **Composable columns**: the adapter/creator pattern lets new column
   types (e.g. lazy, compressed, remote) be added with two small classes.
4. **Natural querying**: `column op value` reads like a predicate;
   `and` chains read like a SQL `WHERE` clause.
5. **NumPy-native**: data lives in a single structured array; columns are
   views, not copies.

---

## Running the tests

```bash
pytest Test/test_GenricTable.py Test/test_QueryableTable.py -v
```

32 tests cover column definitions, table construction, iteration,
indexing, all comparison operators, both pipeline forms (`and` and `&`),
selection intersection, and the `TableCreator` factory.
