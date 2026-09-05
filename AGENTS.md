# AGENTS

## Projekt
BitFlagArray — gepackte Bit-Daten (BitFlagArray/Bitty), typisierte Tabellen über numpy-Strukturarrays, Bit-Selektion.

## Kommunikation
- Der Nutzer schreibt Deutsch — auf Deutsch antworten, knapp und direkt.
- Docstrings/Kommentare minimal halten — lange Docstrings löscht der Nutzer.

## Packaging
- Die 7 Kern-Module liegen im Paket **`clarautils/`** (imports: `from clarautils import Bitty`, `from clarautils.commonTyping import ...`). Interne Imports im Paket sind **relativ mit absolutem Fallback** (`try: from .common import ...` / `except ImportError:`) — so laufen Paket-Import (pytest/`python -m`) UND IDE-Direktaufruf.
- Installierbar per `pyproject.toml` (setuptools, `numpy` als einzige Abhängigkeit): `pip install -e .` im Ziel-venv — Änderungen hier wirken sofort.
- `clarautils/__init__.py` re-exportiert die Kern-API — neue öffentliche Namen dort UND in `__all__` ergänzen.
- Repo-Rest (showcase, printing/DebugPrint, GainCoder, Tests) importiert `clarautils.<Modul>` absolut. Achtung: `association.py`/`reverseEncoding.py` sind gelöscht (Commit 3beb11e).

## Modul-Landkarte
| Modul | Inhalt |
|---|---|
| `clarautils/common.py` | `get_first_or`, `ExceptionRaiser`/`ExcRaiser`, `iter_bits` — **keine** `get_type_*` mehr |
| `clarautils/commonTyping.py` | `INTEGER_TYPES`, `get_type_for_scalar/array/bit_count`, `get_as_signed/unsigned`, `DTableFields`, `ExcRaiser`-Instanzen |
| `clarautils/GenericTable.py` | Table-Framework (`TableFields`, item/range-Varianten, `Table`) — hieß früher `GenricTable.py` |
| `clarautils/QueryableTable.py` | lazy `Query`/`Constraint`, `ConstraintColumn`=`CCol`, `QueryableTable`, `QTblSpecialCol`, `Undefined` |
| `clarautils/commonEncoding.py` | `CommonNBitAry`/`CommonNBitSc`, `get_number`/`get_bits` |
| `clarautils/BitInfo.py` | `BitInfo` — `from_value`/`from_string` mit Modi `bits`/`flags`/`indices`/`count` (B_COUNT läuft in der FLAGS-Pipeline: Bitlänge je Eintrag, durch `bit_count` maskiert; Array-Input normalisiert via `get_as_unsigned(value, fit=True)`, `acc_floats` wird dorthin durchgereicht); `get_bits`/`get_bit_flags` delegieren hierher |
| `clarautils/BitFlagArray.py` | `BitFlagArray`/`Bitty`, `SliceView`, LRU-Cache, **`BitFlagIndex`/`BittyIndex`** (Index-Baum über `group_by_bit`, per `FluentBuilder`: `index_by(key).then_by(*keys).with_leafs(key).index_by_key/slice/fullindex().build()`; Keys: `slice`/`int`/`list[int]` mit `group_by_bit`-Semantik, `list[int]` = EIN Level über mehrere Bits; `with_leafs(key)` fügt KEIN Level hinzu — die tiefste `then_by`-Gruppe wird direkt Blatt, `key` selektiert `leaf.data` (`view.b[key]`), Leaf-Wert wird lokal im Blatt aufgelöst; Blätter materialisiert mit `key_path` + absoluten `item_indices`; `dispose_bitty=True` verwirft die Bitty nach `build` + invalidiert den Cache) |
| `clarautils/Mulitslice.py` | `Multislice` (Schreibweise "Mulitslice" ist Legacy — **nicht umbenennen**) |
| `clarautils/RankedBit.py` | `RankedBit` (rank-first: erst 1-Bit-Werte, dann 2, ...; lex-Ordnung innerhalb Rang wie `itertools.combinations`), `BitGroupWalker` (Odometer über mehrere RankedBit-Masken, Combined = OR), `RankIndexMin` (Ranking über `BittyIndex`: `row_from_cols`/`cols_from_row` via key/int-Abstieg + Leaf-Daten) — `index_in_rank`/`_from_global_index`/`get_next` laufen über den Index (`get_next` = `_from_global_index(gidx+1)`, Floors/Ceilings kombinatorisch via `rank_states`/`comb`, voller Rang k==n als Spezialfall); Anker-Funktionen (`_get_rank_index` etc.) und `_lex_rank`/`_lex_unrank` sind entfernt |
| `Multislice.md` | Benchmarks + Faustregeln zur Bit-Selektion |
| `README.md` | **veraltet** (PlainTable/TableCreator-Ära) — nicht vertrauen |

## Kernkonzepte / Terminologie (vom Nutzer festgelegt)
- "Row type" heißt jetzt **table type**; er zerfällt in **Range** (Spalten-Container, z. B. `CCol`) und **Item** (Skalar, z. B. `np.bool_`). Die Tabellen sind nicht generisch (kein Typ-Parameter, kein `name`-Argument mehr).
- Felder werden **genau einmal** deklariert — als `Range | Item`-Union auf einer `TableFields`-Unterklasse. Die Deklaration IST der table type:
  ```python
  class DTableFields(TableFields):
      signed: CCol | np.bool_
  class TypeTable(QTblSpecialCol, DTableFields): ...
  ```
  Das Framework generiert intern `item_type` (skalar-getyptes NamedTuple, gecacht pro Deklaration) und `range_type` (z. B. `DTableRange`). Es gibt KEIN `table_type_from_fields` mehr — nichts per Hand ableiten.
- `tbl[int]` → Item-Variante (Zeile); `tbl[slice]`/`tbl[indices]` → Range-Variante (Spalten über die Selektion).
- Spaltentypen bestimmt die Deklaration (Validierung beim Bau → `TypeError` bei Mismatch). Familien liefern nur `build_column` + `default_range`.
- Query-Modell ist **lazy**: Vergleiche auf Spalten bauen `Constraint`-Blätter, `&`/`|` komponieren zu `Query`; Auswertung erst bei `.indices`/`get_first`/`get_all`. Der alte `and`-Trick/`_pending_selections` ist entfernt — nicht wieder einführen.
- `Undefined = Literal` (QueryableTable): `column == Undefined` → immer-wahre Bedingung (bei `get_type_for_*` = "beide Vorzeichen-Familien").
- `_find_table_fields` (MRO-Scan) muss Framework-Klassen überspringen — Tabellen erben die Deklaration und matchen sonst selbst.
- Alte Namen (`DRow`, `row_type`, `ConstraintSelection`, Adapter/Creator-Klassen, `GenricTable`) existieren nicht mehr.

## commonTyping-Regeln
- `signed`-Tri-State: `False` (Legacy-Default, nur unsigned), `True` (nur signed), `Undefined` (beide Familien).
- `_get_type_for_bounds`: `abs_min` grenzt die negative Seite ab (-128 passt in int8!); `low=0` deaktiviert das (Legacy-unsigned-Pfad prüft nur `max`).
- Fehlermeldungen **verbatim** (Tippfehler sind Absicht): `Exception("value to big")` / `Exception("to many bits requested")`.
- Annotation `signed: "bool | Undefined"` muss ein **String** sein — `bool | Literal` crasht zur Laufzeit.
- Funktionen, die `INTEGER_TYPES` mehrfach nutzen: lokal `type_tbl = INTEGER_TYPES` aliasen.
- `acc_floats`-Parameter (`get_as_signed`/`get_as_unsigned`/`get_as_fitting`): go-to für Integer-Normalisierung — castet ganzzahlige Floats (Skalar + Array), nicht-ganzzahlige → `TypeError("Expected an integer array")`.

## Mulitslice / Bit-Selektion
- Konvention MSB-first (Bit 0 = MSB), wie BitFlagArray; Runs landen im Ziel in Auswahl-Reihenfolge.
- Shifts **immer signed** rechnen (`.astype(np.intp)`) — unsigned wrappt (-4 → 252).
- Attributnamen: `src_start`/`src_stop` (nicht `start`/`stop`).
- `get_type_for_bit_count(np.max(...))` ist ein Wert-als-Bitcount-Bug — für Werte `get_type_for_scalar` nutzen.
- Multislice skaliert mit Run-Anzahl, nicht mit Bitlänge; Umschlagpunkt zu per-Index bei mittlerer Run-Länge ~3–4 Bits (Details: `Multislice.md`).

## Tests & Aufruf
- Windows/pwsh, Python 3.14 in `.venv`:
  `& .\.venv\Scripts\python.exe -m pytest Test -q`
- **Bewusst defekt** (bei Bedarf mit dem Nutzer abklären): `Test/test_common.py`, `Test/test_DebugPrint.py` (stale `from common import get_type_for_*`); `Test/test_association.py`, `Test/test_reverseEncoding.py` (Module in 3beb11e gelöscht); `Test/test_RankedBit.py` (importiert die auskommentierten Anker-Funktionen); `Test/test_BitFlagArray.py::test_stack_items*` (`stack_items` ist auskommentiert).
- `Test/test_perf*.py` sind Benchmarks (~50 s); `Test/conftest.py` hat eine autouse-Fixture, die BitFlagArray importiert und den Cache räumt.
- Referenz ohne die defekten + Perf: **275 passed** (Stand BitFlagIndex-Einführung; die 2 `stack_items`-Failures sind in der Defekt-Liste).
- Schnellster Smoke-Test: die `__main__`-Demos der Module laufen lassen: `& .\.venv\Scripts\python.exe -m clarautils.GenericTable` (analog `clarautils.QueryableTable`, `clarautils.commonTyping`, `clarautils.Mulitslice`).

## Workflow
- **Nicht** committen/pushen — der Nutzer committet selbst ("update"-Nachrichten).
- Vorsicht bei `git stash`: zwei Stashes des Nutzers vorhanden (einer wurde in einer Session versehentlich gedroppt und wiederhergestellt — stash-Operationen doppelt prüfen).
- `rg` ist nicht installiert — grep-Tool/`Select-String` nutzen.
- Benchmarks für Multislice liegen außerhalb des Repos (Temp), reproduzierbar per `Multislice.md`-Zahlen.
