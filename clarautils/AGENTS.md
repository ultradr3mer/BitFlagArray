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
| `BitInfo.py` | `BitInfo` — zusammengeführte Bit-Abfrage (`from_value`/`from_string`, Modi `bits`/`flags`/`indices`/`count`); `get_bits`/`get_bit_flags` in commonEncoding delegieren dorthin |
| `BitFlagArray.py` | `BitFlagArray`/`Bitty`, `SliceView`, Bit-Selektion (`select_bits*`), LRU-Cache |
| `Mulitslice.py` | `Multislice` (Schreibweise "Mulitslice" ist Legacy — **nicht umbenennen**) |
| `RankedBit.py` | `RankedBit` (rank-first Iteration), `BitGroupWalker` (Odometer über Gruppen), `bits_rank_first*` |

## Kernkonzepte
- **table type** zerfällt in **Range** (Spalten-Container, z. B. `CCol`) und **Item** (Skalar, z. B. `np.bool_`); Tabellen sind nicht generisch (kein Typ-Parameter, kein `name`-Argument).
- Felder **genau einmal** deklarieren — als `Range | Item`-Union auf einer `TableFields`-Unterklasse; die Deklaration IST der table type:
  ```python
  class DTableFields(TableFields):
      signed: CCol | np.bool_
  class TypeTable(QTblSpecialCol, DTableFields): ...
  ```
  Framework generiert `item_type` (NamedTuple) und `range_type` automatisch — nichts per Hand ableiten.
- `tbl[int]` → Item (Zeile); `tbl[slice]`/`tbl[indices]` → Range (Spalten über Selektion).
- Queries sind **lazy**: Spalten-Vergleiche bauen `Constraint`, `&`/`|` komponieren `Query`; Auswertung erst bei `.indices`/`get_first`/`get_all`.
- `Undefined`-Sentinel (QueryableTable): `column == Undefined` → immer-wahre Bedingung (bei `get_type_for_*` = beide Vorzeichen-Familien).
- `signed`-Tri-State in commonTyping: `False` (nur unsigned), `True` (nur signed), `Undefined` (beide).
- Bit-Konvention **MSB-first** (Bit 0 = MSB); Shifts immer signed rechnen (`.astype(np.intp)`) — unsigned wrappt.
- **RankedBit**: Reihenfolge rank-first (erst alle Werte mit 1 Bit, dann 2, ...), innerhalb eines Rangs lexikographisch wie `itertools.combinations`; `bit_count` = Popcount der Maske (= mask_rank); empty (`bit_value=0`) hat `global_index` -1; `get_next(max_rank_idx)` grenzt Ranks ein (0-basierter Rang-Index), sonst Lauf bis volle Maske; `BitGroupWalker` verlangt paarweise disjunkte Masken, iteriert product-artig (innerste Gruppe zuerst) und liefert das OR als Combined-Zahl.

## Regeln
- Fehlermeldungen **verbatim** (Tippfehler sind Absicht): `"value to big"` / `"to many bits requested"`.
- `get_type_for_bit_count` nimmt Bit-Anzahlen, `get_type_for_scalar` Werte — nicht vertauschen.
- Neue öffentliche Namen in `clarautils/__init__.py` UND `__all__` ergänzen.
- Interne Imports im Paket relativ (`from .common import ...`); Ausnahme `RankedBit.py`: zusätzlicher absoluter Fallback im `try/except ImportError`, damit Direktaufruf `python clarautils/RankedBit.py` aus der IDE läuft.
- Brute-Force-Referenz `get_comb_idx` (`(lex_index, naechste_kombination | None)`) liegt in `Test/test_RankedBit.py` — bewusst nicht im Paket.
- Index ist NICHT linear in den Positionen (LGS geprüft — polynomial mit Grad = Rang). Deshalb Anker-Modell: `_INDEX_CACHE` je Bit-Anzahl n (Positionen, nicht Bits — Masks gleicher Größe teilen den Cache); ein Durchlauf notiert die Umbruch-Anker (Positionen + Index), Lookup = Anker + Rest der letzten Spalte (`_gidx_of_positions`/`_positions_of_gidx`). `get_next` zählt nur die letzte Position hoch und springt bei Umbruch zum nächsten Anker. `_lex_index`/`_lex_unrank`/`_next_positions` sind entfernt.
- Annotation `signed: "bool | Undefined"` muss ein **String** sein — `bool | Literal` crasht zur Laufzeit.
