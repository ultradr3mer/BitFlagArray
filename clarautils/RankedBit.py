from itertools import combinations
from math import comb
from typing import List, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt

try:
    from .commonEncoding import get_bit_flags, normalize_flags
    from .commonTyping import get_as_fitting, get_as_unsigned
except ImportError:  # direktaufruf: python clarautils/RankedBit.py
    from clarautils.commonEncoding import get_bit_flags, normalize_flags
    from clarautils.commonTyping import get_as_fitting, get_as_unsigned


def bits_combs_by_rank(bit_mask: int) -> npt.NDArray:
    flags = get_bit_flags(bit_mask)
    item_count = flags.size
    return get_as_fitting([[r_idx, sum(c)] for r_idx in range(0, item_count) for c in combinations(flags, r_idx + 1)])


def bits_rank_first(bit_count: int, min_r: int = 0, max_r: "int | None" = None) -> np.ndarray:
    return bits_rank_first_from_flags(1 << np.arange(bit_count), min_r, max_r)


def bits_rank_first_from_flags(bit_flags, min_r: int = 0, max_r: "int | None" = None) -> np.ndarray:
    max_r = len(bit_flags) if max_r is None else max_r
    return get_as_unsigned([sum(p)
                            for r in range(min_r, max_r)
                            for p in combinations(bit_flags, r + 1)],
                           fit=True)


def get_comb_idx(comb_items, value_mask, value_rank) -> "Tuple[int, Tuple[int, ...] | None]":
    # brute-force referenz: (lex-index, naechste kombination), None wenn keine folgt
    next_return = False
    return_idx = -1
    for i, c in enumerate(combinations(comb_items, value_rank)):
        if next_return:
            return return_idx, c
        if sum(c) == value_mask:
            return_idx = i
            next_return = True
    return return_idx, None


def rank_states_per_rank(mask_rank: int, rank_slice: "slice | int | npt.ArrayLike") -> np.ndarray:
    ranks_idx = np.atleast_1d(np.arange(mask_rank)[rank_slice])
    return np.array([comb(mask_rank, i + 1) for i in ranks_idx])


def rank_states(mask_rank: int, val_rank: int) -> int:
    # Anzahl Zustaende mit Rang < val_rank (Floor fuer Rang val_rank)
    return sum(comb(mask_rank, i) for i in range(1, val_rank))


def _lex_index(positions, mask_rank: int) -> int:
    # Rang innerhalb eines Rangs, Ordnung wie itertools.combinations
    pos = [int(v) for v in np.asarray(positions)]
    k = len(pos)
    if k == 0:
        return 0
    idx = comb(mask_rank, k) - comb(mask_rank - pos[0], k)
    for i in range(1, k):
        idx += comb(mask_rank - 1 - pos[i - 1], k - i) - comb(mask_rank - pos[i], k - i)
    return idx


def _lex_unrank(mask_rank: int, k: int, idx: int) -> np.ndarray:
    pos = np.empty(k, dtype=np.int64)
    prev = -1
    for i in range(k):
        r = k - i
        v = prev + 1
        while True:
            cnt = comb(mask_rank - 1 - v, r - 1)
            if idx < cnt:
                break
            if cnt == 0:
                raise ValueError("lex index out of range")
            idx -= cnt
            v += 1
        pos[i] = v
        prev = v
    return pos


def _next_positions(pos: np.ndarray, mask_rank: int) -> "np.ndarray | None":
    # naechste Kombination in lex-Ordnung, None wenn die letzte erreicht ist
    k = pos.size
    i = k - 1
    while i >= 0 and pos[i] == mask_rank - k + i:
        i -= 1
    if i < 0:
        return None
    new = pos.copy()
    new[i] += 1
    new[i + 1:] = np.arange(int(new[i]) + 1, int(new[i]) + 1 + (k - i - 1))
    return new


class RankedBit(NamedTuple):
    class RankInfo(NamedTuple):
        rank_index: int
        index_floor: int

        def __repr__(self):
            return f"RankInfo(rank={self.rank_index}, floor={self.index_floor})"

    class CombInfo(NamedTuple):
        position: "np.ndarray | Tuple[int, ...]"
        mask_rank: int

        @property
        def bitwise_index(self) -> np.ndarray:
            pos = np.asarray(self.position)
            return pos - np.arange(pos.size)

        @property
        def index_in_rank(self) -> int:
            return _lex_index(self.position, self.mask_rank)

        @property
        def pos_str(self) -> str:
            return ", ".join(str(int(v)) for v in np.asarray(self.position))

        def __repr__(self):
            return f"CombInfo(pos=[{self.pos_str}], idx={self.index_in_rank}, n={self.mask_rank})"

    class Info(NamedTuple):
        rank_info: "RankedBit.RankInfo"
        comb_info: "RankedBit.CombInfo"

        @property
        def global_index(self) -> int:
            if self.rank_info.rank_index == 0:
                return -1
            return self.rank_info.index_floor + self.comb_info.index_in_rank

        def __repr__(self):
            return (f"Info(rank={self.rank_info.rank_index}, floor={self.rank_info.index_floor}, "
                    f"pos=[{self.comb_info.pos_str}], idx={self.comb_info.index_in_rank}, "
                    f"gidx={self.global_index})")

    bit_mask: int
    bit_value: int
    bit_count: int

    def expand(self) -> Tuple[int, npt.NDArray, int, npt.NDArray]:
        if self.bit_value & ~self.bit_mask:
            raise ValueError("val not in mask")
        mask_flags: npt.NDArray = get_bit_flags(self.bit_mask) if self.bit_mask else np.array([], dtype=np.int64)
        mask_rank = mask_flags.size
        value_flags: npt.NDArray = get_bit_flags(self.bit_value) if self.bit_value else np.array([], dtype=mask_flags.dtype)
        rank = value_flags.size
        if not (np.diff(mask_flags) > 0).all() or not (np.diff(value_flags) > 0).all():
            raise ValueError("val and mask mut not contain same flag twice")
        return mask_rank, mask_flags, rank, value_flags

    def get_comb_info(self) -> "RankedBit.CombInfo":
        mask_rank, mask_flags, val_rank, value_flags = self.expand()
        bitwise_pos = np.where(mask_flags & self.bit_value)[0]
        return RankedBit.CombInfo(bitwise_pos, mask_rank)

    def get_rank_info(self) -> "RankedBit.RankInfo":
        mask_rank, mask_flags, val_rank, value_flags = self.expand()
        return RankedBit.RankInfo(val_rank, rank_states(mask_rank, val_rank))

    def get_info(self) -> "RankedBit.Info":
        return RankedBit.Info(self.get_rank_info(), self.get_comb_info())

    @property
    def global_index(self) -> int:
        return self.get_info().global_index

    def _with_value(self, value: int) -> "RankedBit":
        return RankedBit(self.bit_mask, value, self.bit_count)

    def get_next(self, max_rank_idx: "int | None" = None) -> "RankedBit":
        # max_rank_idx: 0-basierter Rang-Index (r -> r+1 Bits); None laeuft bis zur vollen Maske
        mask_rank, mask_flags, val_rank, value_flags = self.expand()
        max_rank = mask_rank if max_rank_idx is None else min(max_rank_idx + 1, mask_rank)
        if val_rank > max_rank:
            raise StopIteration
        if val_rank == 0:
            if max_rank < 1:
                raise StopIteration
            return self._with_value(int(mask_flags[0]))
        pos = np.where(mask_flags & self.bit_value)[0]
        nxt = _next_positions(pos, mask_rank)
        if nxt is not None:
            return self._with_value(int(np.bitwise_or.reduce(mask_flags[nxt])))
        if val_rank >= max_rank:
            raise StopIteration
        return self._with_value(int(np.bitwise_or.reduce(mask_flags[:val_rank + 1])))

    def _from_global_index(self, gidx: int) -> "RankedBit":
        mask_rank, mask_flags, val_rank, value_flags = self.expand()
        if gidx < -1 or gidx >= rank_states(mask_rank, mask_rank + 1):
            raise IndexError("ranked bit index out of range")
        if gidx == -1:
            return self._with_value(0)
        k = 1
        while gidx >= comb(mask_rank, k):
            gidx -= comb(mask_rank, k)
            k += 1
        return self._with_value(int(np.bitwise_or.reduce(mask_flags[_lex_unrank(mask_rank, k, gidx)])))

    def __add__(self, steps: int) -> "RankedBit":
        if not isinstance(steps, int):
            return NotImplemented
        return self._from_global_index(self.global_index + steps)

    def __sub__(self, steps: int) -> "RankedBit":
        if not isinstance(steps, int):
            return NotImplemented
        return self._from_global_index(self.global_index - steps)

    def iter_next(self, max_rank_idx: "int | None" = None):
        cur = self
        while True:
            try:
                cur = cur.get_next(max_rank_idx)
            except StopIteration:
                return
            yield cur

    @staticmethod
    def _build_int(value: "int | npt.ArrayLike") -> int:
        if isinstance(value, int):
            return value
        ipt = np.bitwise_or.reduce(value)
        if not np.sum(value) == ipt:
            raise ValueError("val and mask mut not contain same flag twice")
        return int(ipt)

    @classmethod
    def empty(cls, bit_count: int) -> "RankedBit":
        return cls(bit_mask=(1 << bit_count) - 1, bit_value=0, bit_count=bit_count)

    @classmethod
    def from_value(cls, val: "int | npt.ArrayLike", mask: "int | npt.ArrayLike | None" = None) -> "RankedBit":
        v = cls._build_int(val)
        m = cls._build_int(mask) if mask is not None else (1 << v.bit_length()) - 1
        if v < 0 or m < 0 or v & ~m:
            raise ValueError("val not in mask")
        return cls(bit_mask=m, bit_value=v, bit_count=m.bit_count())

    @classmethod
    def from_bits(cls, bits: npt.ArrayLike, indices: npt.ArrayLike = None) -> "RankedBit":
        b = get_as_unsigned(bits, fit=True)
        i = get_as_unsigned(indices, fit=True) if indices is not None else np.arange(b.size)
        _, mask, value = normalize_flags(i, b)
        m, v = int(mask), int(value)
        if v & ~m:
            raise ValueError("val not in mask")
        return cls(m, v, m.bit_count())

    def __repr__(self):
        mask_rank, mask_flags, rank, value_flags = self.expand()
        width = int(self.bit_mask).bit_length()
        v, m = self.bit_value, self.bit_mask
        bit_chars = ("1" if (v >> i) & 1 else "0" if (m >> i) & 1 else "." for i in range(width - 1, -1, -1))
        return f"RankedBit(bits='{''.join(bit_chars)}', mask=0b{m:0{width}b}, rank={rank}/{mask_rank})"


class BitGroupWalker:
    # odometer ueber mehrere RankedBit-Masken ("Ziffern"), innerste Gruppe zuerst
    def __init__(self, *groups: "RankedBit | int", max_rank_idx: "int | None" = None):
        if len(groups) == 1 and not isinstance(groups[0], (RankedBit, int)):
            groups = tuple(groups[0])
        self.groups: List[RankedBit] = []
        seen = 0
        for g in groups:
            mask = g.bit_mask if isinstance(g, RankedBit) else int(g)
            if mask <= 0:
                raise ValueError("empty group mask")
            if seen & mask:
                raise ValueError("overlapping group masks")
            seen |= mask
            self.groups.append(RankedBit(mask, 0, mask.bit_count()))
        self.max_rank_idx = max_rank_idx
        self._done = False

    @property
    def state(self) -> Tuple[RankedBit, ...]:
        return tuple(self.groups)

    @property
    def combined(self) -> int:
        v = 0
        for g in self.groups:
            v |= g.bit_value
        return v

    def __len__(self):
        if not self.groups:
            return 0
        total = 1
        for g in self.groups:
            n_g = g.expand()[0]
            cap = n_g if self.max_rank_idx is None else min(self.max_rank_idx + 1, n_g)
            total *= rank_states(n_g, cap + 1)
        return total

    def get_next(self, max_rank_idx: "int | None" = None) -> int:
        if self._done:
            raise StopIteration
        mri = self.max_rank_idx if max_rank_idx is None else max_rank_idx
        gs = self.groups
        if not gs:
            raise StopIteration
        if all(g.bit_value == 0 for g in gs):
            for i in range(len(gs)):
                gs[i] = gs[i].get_next(mri)
            return self.combined
        i = len(gs) - 1
        while i >= 0:
            try:
                gs[i] = gs[i].get_next(mri)
                break
            except StopIteration:
                gs[i] = RankedBit(gs[i].bit_mask, 0, gs[i].bit_count)
                i -= 1
        else:
            self._done = True
            raise StopIteration
        for j in range(i + 1, len(gs)):
            if gs[j].bit_value == 0:
                gs[j] = gs[j].get_next(mri)
        return self.combined

    def __next__(self) -> int:
        return self.get_next()

    def __iter__(self):
        return self


if __name__ == '__main__':
    demo = RankedBit.from_bits([1, 0, 1], [1, 2, 3])
    print(demo, "->", demo.get_info())

    empty = RankedBit.empty(4)
    print("rank-first ab leer:", [b.bit_value for b in empty.iter_next()])
    print("nur bis Rang 2:   ", [b.bit_value for b in empty.iter_next(max_rank_idx=1)])
    print("plus/minus:       ", (empty + 6).bit_value, (empty + 6 - 2).bit_value)

    walker = BitGroupWalker(0b0011, 0b1100, max_rank_idx=0)
    print("walker 1 bit/grp: ", [v for v in walker])
    walker = BitGroupWalker(0b0011, 0b1100)
    print("walker komplett:  ", [v for v in walker])
