# clarautils

Utility-Paket von Clara: gepackte Bit-Daten (`BitFlagArray`/`Bitty`), typisierte Tabellen über numpy-Strukturarrays, Bit-Selektion.
Quelle: `F:\source\BitFlagArray`, installiert per `pip install -e F:\source\BitFlagArray` (editable — Änderungen wirken sofort).

## Import
- Top-Level: `from clarautils import Bitty, SliceView, put_bits, TableFields, Table, CCol, QueryableTable, QTblSpecialCol, Query, Undefined, Multislice, get_bits, get_number, get_type_for_scalar, ...` (volle Liste: `__init__.py` / `__all__`)
- Modul-qualifiziert: `from clarautils.commonTyping import get_type_for_array`, `from clarautils.BitFlagArray import select_bits`, ...

## Module
| Modul | Inhalt |
|---|---|
| `common.py` | `get_first_or`, `ExceptionRaiser`/`ExcRaiser`, `AccessibleAry`, `iter_bits` — **keine** `get_type_*` (die in commonTyping) |
| `commonTyping.py` | `INTEGER_TYPES`, `get_type_for_scalar/array/bit_count`, `get_as_signed/unsigned`, `DTableFields`, `TypeTable` |
| `GenericTable.py` | Table-Framework: `TableFields`, `Table`, `NpColTable`, item/range-Typen-Ableitung |
| `QueryableTable.py` | lazy `Query`/`Constraint`, `ConstraintColumn`=`CCol`, `QueryableTable`, `QTblSpecialCol`, `Undefined` |
| `commonEncoding.py` | `CommonNBitAry`/`CommonNBitSc`, `get_number`/`get_bits`, Hex/Bit-Konvertierung |
| `BitFlagArray.py` | `BitFlagArray`/`Bitty`, `SliceView`, Bit-Selektion (`select_bits*`), LRU-Cache |
| `Mulitslice.py` | `Multislice` (Schreibweise "Mulitslice" ist Legacy — **nicht umbenennen**) |

## Kernkonzepte
- **table type** zerfällt in **Range** (Spalten-Container, z. B. `CCol`) und **Item** (Skalar, z. B. `np.bool_`); Generics-Parameter: `TRowNRange`.
- Felder **genau einmal** deklarieren — als `Range | Item`-Union auf einer `TableFields`-Unterklasse; die Deklaration IST der table type:
  ```python
  class DTableFields(TableFields):
      signed: CCol | np.bool_
  class TypeTable(QTblSpecialCol[DTableFields, Type[Any]], DTableFields): ...
  ```
  Framework generiert `item_type` (NamedTuple) und `range_type` automatisch — nichts per Hand ableiten.
- `tbl[int]` → Item (Zeile); `tbl[slice]`/`tbl[indices]` → Range (Spalten über Selektion).
- Queries sind **lazy**: Spalten-Vergleiche bauen `Constraint`, `&`/`|` komponieren `Query`; Auswertung erst bei `.indices`/`get_first`/`get_all`.
- `Undefined`-Sentinel (QueryableTable): `column == Undefined` → immer-wahre Bedingung (bei `get_type_for_*` = beide Vorzeichen-Familien).
- `signed`-Tri-State in commonTyping: `False` (nur unsigned), `True` (nur signed), `Undefined` (beide).
- Bit-Konvention **MSB-first** (Bit 0 = MSB); Shifts immer signed rechnen (`.astype(np.intp)`) — unsigned wrappt.

## Regeln
- Fehlermeldungen **verbatim** (Tippfehler sind Absicht): `"value to big"` / `"to many bits requested"`.
- `get_type_for_bit_count` nimmt Bit-Anzahlen, `get_type_for_scalar` Werte — nicht vertauschen.
- Neue öffentliche Namen in `clarautils/__init__.py` UND `__all__` ergänzen.
- Interne Imports im Paket relativ (`from .common import ...`).
- Annotation `signed: "bool | Undefined"` muss ein **String** sein — `bool | Literal` crasht zur Laufzeit.
