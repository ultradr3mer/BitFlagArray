# LAST_SESSION — Stand 13.09.2026, ~15:10

## Thema dieser Session
1. `BitFlagGroupView` / `build_groups` / `IdxGroupView` — Gruppierung von Bits in NamedTuple-Felder
2. `MultiIndex` / `ItemBitIndex` — generische Index-Wrapper (common.py)
3. opencode-Plugin `reflect-watcher` (5-Min-Reflection-Trigger)

## Was funktioniert (verifiziert)

### Bit-Gruppierung (clarautils/BitFlagArray.py)
- `build_groups(ary, group_lens, t_target=None)` (~Zeile 920): `int`-Lens → Ceil-Division über alle Bits; Slice-Offsets via `get_slices_from_diffs(np.cumsum(group_lens) - 1)` (kumulativ minus 1, NICHT die Längen direkt!)
- `BitFlagGroupView` — Deklarations-Basis (Marker-Pattern wie `TableFields`):
  `__init_subclass__` baut per funktionaler NamedTuple-API `cls.group_type` aus `get_type_hints`
- `IdxGroupView` — generischer Container (Default-Target von `build_groups`): `groups`-Liste, `__len__`/`__iter__`, late binding `None → IdxGroupView`
- `NBitArray.g(group_lengths, t_target=None)` — Convenience-Methode auf allen NBitArrays
- `GTest(BitFlagGroupView)` mit `sign/exponent/mantissa` — Demo läuft

### MultiIndex / ItemBitIndex (clarautils/common.py:7)
- `class MultiIndex[T_index: Tuple[Any, ...]](NamedTuple)` — generisch, PEP 695, geht in Python 3.14
- `class ItemBitIndex(MultiIndex[Tuple[int, ...]])` — Subclassing ok, `item`/`bit` als `@property` auf `index` (Defaults wie `item: int = index[0]` gehen NIE — Klassen-Konstante vs. Instanz)
- Semantik: `ItemBitIndex((3, 7))` ist ein **1-Tupel**, `x[0] == (3, 7)`, NICHT `x[0] == 3`! `x.item == 3`, `x.bit == 7` via Properties
- Exportiert in `__init__.py` + `__all__`

### Plugin (.opencode/plugins/reflect-watcher.ts)
- Bun-Build ok. Alle 5 min: Worktree-Snapshot (mtime:size), Diff gegen Baseline; nur wenn Agent idle
- Bei Änderung: Toast + Prompt an **plan**-Agent (`agentID: "plan"`, Fallback ohne) mit:
  Dateiliste, Inaktivitätsdauer, Reflektions-Auftrag, 2-3 Follow-up-Vorschläge
- **Wirkt erst nach opencode-Neustart** (Plugins laden nur beim Start)

### Gelernt / Fallgruben diese Session
- **NamedTuple-Subclassing in 3.14**: Felder im Subclass-Rumpf werden IGNORIERT — nur Methoden/Properties vererben sich. Generische Params via `[T_index: ...]` gehen.
- **SliceView-Iteration ist UNENDLICH**: `__getitem__(int)` wirft nie `IndexError` (liefert immer neuen View) → `enumerate(view)`/`for x in view` hängt! Immer `range(len(view))` / `get_item_count()` nutzen.
- **Late Binding statt Vorwärts-Stubs**: `class X: ...`-Stubs + Redefinition später = Zombie-Klassen. Besser: Default `None`, zur Aufrufzeit auflösen (`build_groups`-Pattern).
- Annotations mit Vorwärtsreferenzen als **String** (`"Type[...] | None"`).
- `\` am Klassende ist KEINE "Fortsetzt später"-Ankündigung — es spliced die nächste physische Zeile.
- `exampe_data`-Import braucht das relative/Fallback-`try/except`-Muster, sonst bricht der Paket-Import.

## AKTUELL KAPUTT (live editiert, ~15:05 — Nutzer war noch dran)

`IdxGroupView.__getitem__` (clarautils/BitFlagArray.py:895) — Runtime-Fehler:
1. **Zeile 901: `keytype` ist undefiniert** → `c[:, 0]` wirft `NameError: name 'keytype' is not defined`
   (gewollt vermutlich `(int, np.integer)`)
2. **Zeile 896-897: `issubclass(idx, int)`** — prüft Klassen; `isinstance` für Werte. Bedingung matched nie sinnvoll.
3. **Zeile 902: `enumerate(self.groups[grp])`** — UNENDLICHE Iteration über SliceView (siehe Fallgrube oben) → HANG. Ersatz: `range(self.groups[grp].get_item_count())`
4. **Zeile 903: `Bitty.stack_bit_arys(**self.groups[i.item].b[b.bit])`** — `i.item` ist `idx` (das Tupel selbst), nicht der Loop-Index; `**` auf ein SliceView ergibt keine kwargs. Diese Zeile ist toter Code (nach Zeile 902 unreachable, wenn 902 hängt eh nie erreichbar).
5. **`ItemBitIndex(i_itm, bit)`** — falsche Kardinalität! `ItemBitIndex` nimmt EIN Tuple (`ItemBitIndex((i_itm, bit))`), nicht zwei Argumente.
6. `ItemBitIndex((1, 4))` als Index wirft `IndexError: tuple index out of range` — der `isinstance(idx, ItemBitIndex)`-Zweig fehlt (er wurde beim Live-Edit überschrieben).

### Letzte bekannte GUTE Version von `__getitem__` (vor den Live-Edits, alles grün):
```python
def __getitem__(self, idx):
    if isinstance(idx, ItemBitIndex):
        grp, bit = idx.index
        view = self.groups[grp].b[bit]
        return NBitAryOnly(view.get_array(), view.get_bit_count())
    if not isinstance(idx, (str, bytes)) and isinstance(idx, Iterable):
        items = list(idx)
        if items and all(isinstance(x, ItemBitIndex) for x in items):
            return [self[x] for x in items]
    if isinstance(idx, tuple):
        grp, bit = idx[0], idx[1]
        if isinstance(grp, (int, np.integer)):
            view = [ItemBitIndex((i_itm, bit)) for i_itm in range(self.groups[grp].get_item_count())]
            return [self[row] for row in view]
        return [self[ItemBitIndex((g, bit))] for g in range(len(self.groups))[grp]]
    return self.groups[idx]
```

## Test-Referenz
`& .\.venv\Scripts\python.exe -m pytest Test -q` (ohne die defekten + Perf):
- **275 passed** = Referenz-Grün
- Bekannt defekt (dokumentiert, nicht fixen): `test_common`, `test_DebugPrint`, `test_association`, `test_reverseEncoding`, `test_RankedBit` (Import-Fehler), `test_stack_items*` (2 Failures, `stack_items` auskommentiert)
- Perf-Benchmarks: `test_perf*.py` (~50 s, ignorieren)

## Offene Follow-ups
1. `__getitem__` reparieren (Punkte 1-6 oben) — dann `c[(1, 4)]`, `c[:, 0]`, `ItemBitIndex`-Zugriff verifizieren
2. `.opencode/` ist ungetrackt (`?? .opencode/`) — entscheiden: committen oder ignorieren
3. Idea aus Zeile 903 (Kommentar 897): pro Item `stack_bit_arys` über die Bit-Selection — sinnvoll erst nach Fix; `Bitty.stack_bit_arys(*views)` (positionals, keine `**`)
4. `README.md` ist veraltet (AGENTS.md warnt) — Update steht noch aus
5. `git status`: `clarautils/BitFlagArray.py`, `__init__.py`, `common.py` modifiziert — Nutzer committet selbst ("update")
