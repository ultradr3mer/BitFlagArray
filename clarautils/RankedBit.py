from itertools import combinations
from math import comb
from typing import Dict, List, NamedTuple, Tuple, Literal
from clarautils import BitInfo

import numpy as np
import numpy.typing as npt

from BitFlagArray import Bitty, NBitAryOnly

from commonEncoding import get_bit_flags, normalize_flags
from commonTyping import get_as_fitting, get_as_unsigned


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

def rank_states_per_rank(mask_rank: int, rank_slice: "slice | int | npt.ArrayLike") -> np.ndarray:
    ranks_idx = np.atleast_1d(np.arange(mask_rank)[rank_slice])
    return np.array([comb(mask_rank, i + 1) for i in ranks_idx])


def rank_states(mask_rank: int, val_rank: int) -> int:
    # Anzahl Zustaende mit Rang < val_rank (Floor fuer Rang val_rank)
    return sum(comb(mask_rank, i) for i in range(1, val_rank))

###############################
# Umbruch-Anker: Positionen vs Index (n=6, rang 3)
###############################
# A = anker/segmentstart: letzte spalte beginnt direkt hinter der vorletzten
#
#      komb     pos      idx
#      1,2,3    0,1,2      0   A   rang-start
#      1,2,4    0,1,3      1
#      1,2,5    0,1,4      2
#      1,2,6    0,1,5      3       letzte spalte am ende -> umbruch
#      1,3,4    0,2,3      4   A
#      1,3,5    0,2,4      5
#      1,3,6    0,2,5      6       umbruch
#      1,4,5    0,3,4      7   A
#      1,4,6    0,3,5      8       umbruch
#      1,5,6    0,4,5      9   A
#      2,3,4    1,2,3     10   A
#      2,3,5    1,2,4     11
#      2,3,6    1,2,5     12       umbruch
#      2,4,5    1,3,4     13   A
#      2,4,6    1,3,5     14       umbruch
#      2,5,6    1,4,5     15   A
#      3,4,5    2,3,4     16   A
#      3,4,6    2,3,5     17       umbruch
#      3,5,6    2,4,5     18   A
#      4,5,6    3,4,5     19   A   rang-ende
#
# lookup: idx = anker-idx + (letzte pos - vorletzte pos - 1); rang 1: ein anker, wertigkeit 1
# tabelle im code: _get_rank_index(n) — gepinnt in test_rank_index_anchors_match_table


# class RankIndexMin(NamedTuple):
#     index_floors: Bitty
#     mask_rank: int
#     val_rank: int
class RankIndexMin:
    _INSTANCE_CACHE: Dict[Tuple[int, int], RankIndexMin] = {}

    def __init__(
        self,
        # index_floors: Bitty,
        mask_rank: int,
        val_rank: int,
    ) -> None:
        self.index_floors: Bitty = None
        self.mask_rank = mask_rank
        self.val_rank = val_rank

    @staticmethod
    def idx_from_pos(pos: np.ndarray) -> np.ndarray:
        pos_flr = np.arange(pos.shape[1])
        if not (np.min(pos, axis=0) == pos_flr).all():
            raise ValueError("pos not in range -> r_min save required")
        return pos - pos_flr

    @staticmethod
    def pos_from_idx(idx: np.ndarray) -> np.ndarray:
        return idx + np.arange(idx.shape[1])

    @classmethod
    def create(cls, mask_rank: int, val_rank: int) -> 'RankIndexMin':
        # comb_items = gen_labels(mask_rank)
        full_tbl = get_as_unsigned([i for i in combinations(range(mask_rank), val_rank)], fit=True)

        total_combinations = comb(mask_rank, val_rank)
        bty = Bitty.empty((total_combinations, mask_rank))
        instance = RankIndexMin(index_floors=bty, mask_rank=mask_rank, val_rank=val_rank)

        full_idx = cls.idx_from_pos(full_tbl)
        bit_used_binf = BitInfo.from_value(np.max(full_idx, axis=0), mode=BitInfo.Mode.B_COUNT)
        bit_used = [int(m).bit_count() for m in np.max(full_idx, axis=0)]
        if not (bit_used_binf == bit_used).all():
            raise ValueError("bit_used_binf and bit_used do not match")

        n_bit_cols = [NBitAryOnly(c,b) for c, b in zip(full_idx, bit_used)]
        bty = Bitty.stack_items(*n_bit_cols)

        return

    @classmethod
    def get_or_create(cls, mask_rank: int, val_rank: int) -> RankIndexMin:
        key = (mask_rank, val_rank)

        instance = _INSTANCE_CACHE.get(key)
        if instance is None:
            instance = cls.create(mask_rank, val_rank)
            _INSTANCE_CACHE[key] = instance

        return instance

if __name__ == '__main__':
    demo = RankIndexMin.get_or_create(5,3)
    # mask / value rank
#
# class _RankIndex(NamedTuple):
#
#     # pro bit-anzahl: umbruch-anker (positionen + index), floors je rang, total
#     anchor_pos: List[Tuple[int, ...]]
#     anchor_gidx: List[int]
#     anchor_row: Dict[Tuple[int, ...], int]
#     rank_floors: List[int]
#     total: int
#
#
# _INDEX_CACHE: Dict[int, _RankIndex] = {}
#
#
# def _build_rank_index(mask_rank: int) -> _RankIndex:
#     # ein durchlauf: anker = segmentstart, letzte spalte beginnt direkt hinter der vorletzten
#     anchor_pos, anchor_gidx = [], []
#     rank_floors = []
#     g = 0
#     for k in range(1, mask_rank + 1):
#         rank_floors.append(g)
#         for pos in combinations(range(mask_rank), k):
#             prev = pos[-2] if len(pos) > 1 else -1
#             if pos[-1] == prev + 1:
#                 anchor_pos.append(pos)
#                 anchor_gidx.append(g)
#             g += 1
#     return _RankIndex(anchor_pos, anchor_gidx, {p: i for i, p in enumerate(anchor_pos)}, rank_floors, g)
#
#
# def _get_rank_index(mask_rank: int) -> _RankIndex:
#     # cache je bit-anzahl (n), nicht je maske — gleiche groesse teilt sich die tabelle
#     ri = _INDEX_CACHE.get(mask_rank)
#     if ri is None:
#         ri = _INDEX_CACHE[mask_rank] = _build_rank_index(mask_rank)
#     return ri
#
#
# def _gidx_of_positions(mask_rank: int, pos) -> int:
#     # anker des segments suchen, dann letzte spalte aufzaehlen (rang 1: wertigkeit 1)
#     ri = _get_rank_index(mask_rank)
#     p = tuple(int(v) for v in np.asarray(pos))
#     prev = p[-2] if len(p) > 1 else -1
#     anchor = p[:-1] + (prev + 1,)
#     return ri.anchor_gidx[ri.anchor_row[anchor]] + (p[-1] - prev - 1)
#
#
# def _positions_of_gidx(mask_rank: int, gidx: int) -> Tuple[int, ...]:
#     # umkehrung: anker per bisect, letzte spalte = anker-ende + rest
#     ri = _get_rank_index(mask_rank)
#     if not 0 <= gidx < ri.total:
#         raise IndexError("ranked bit index out of range")
#     row = bisect_right(ri.anchor_gidx, gidx) - 1
#     anchor = ri.anchor_pos[row]
#     return anchor[:-1] + (anchor[-1] + gidx - ri.anchor_gidx[row],)
#

class RankInfo(NamedTuple):
    rank_index: int
    index_floor: int

    def __repr__(self):
        return f"RankInfo(rank={self.rank_index}, floor={self.index_floor})"

class RankCombInfo(NamedTuple):
    position: "np.ndarray | Tuple[int, ...]"
    mask_rank: int

    @property
    def bitwise_index(self) -> np.ndarray:
        pos = np.asarray(self.position)
        return pos - np.arange(pos.size)

    @property
    def index_in_rank(self) -> int:
        pos = np.asarray(self.position)
        if pos.size == 0:
            return 0
        ri = _get_rank_index(self.mask_rank)
        return _gidx_of_positions(self.mask_rank, pos) - ri.rank_floors[pos.size - 1]

    @property
    def pos_str(self) -> str:
        return ", ".join(str(int(v)) for v in np.asarray(self.position))

    def __repr__(self):
        return f"CombInfo(pos=[{self.pos_str}], idx={self.index_in_rank}, n={self.mask_rank})"

class RankedBitInfo(NamedTuple):
    rank_info: RankInfo
    comb_info: RankCombInfo

    @property
    def global_index(self) -> int:
        if self.rank_info.rank_index == 0:
            return -1
        return self.rank_info.index_floor + self.comb_info.index_in_rank

    def __repr__(self):
        return (f"Info(rank={self.rank_info.rank_index}, floor={self.rank_info.index_floor}, "
                f"pos=[{self.comb_info.pos_str}], idx={self.comb_info.index_in_rank}, "
                f"gidx={self.global_index})")


class RankedBit(NamedTuple):
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

    def get_comb_info(self) -> RankCombInfo:
        mask_rank, mask_flags, val_rank, value_flags = self.expand()
        bitwise_pos = np.where(mask_flags & self.bit_value)[0]
        return RankCombInfo(bitwise_pos, mask_rank)

    def get_rank_info(self) -> RankInfo:
        mask_rank, mask_flags, val_rank, value_flags = self.expand()
        return RankInfo(val_rank, rank_states(mask_rank, val_rank))

    def get_info(self) -> RankedBitInfo:
        return RankInfo(self.get_rank_info(), self.get_comb_info())

    @property
    def global_index(self) -> int:
        return self.get_info().global_index

    def _with_value(self, value: int) -> "RankedBit":
        return RankedBit(self.bit_mask, value, self.bit_count)

    def get_next(self, max_rank_idx: "int | None" = None) -> "RankedBit":
        # # max_rank_idx: 0-basierter Rang-Index (r -> r+1 Bits); None laeuft bis zur vollen Maske
        # # letzte spalte hochzaehlen; bei umbruch (oder ab leer) den naechsten anker ueber den index
        # mask_rank, mask_flags, val_rank, value_flags = self.expand()
        # ri = _get_rank_index(mask_rank)
        # max_rank = max(0, mask_rank if max_rank_idx is None else min(max_rank_idx + 1, mask_rank))
        # if val_rank > max_rank:
        #     raise StopIteration
        # if val_rank > 0:
        #     pos = np.where(mask_flags & self.bit_value)[0]
        #     if int(pos[-1]) < mask_rank - 1:
        #         nxt = pos.copy()
        #         nxt[-1] += 1
        #         return self._with_value(int(np.bitwise_or.reduce(mask_flags[nxt])))
        #     gidx = _gidx_of_positions(mask_rank, pos)
        # else:
        #     gidx = -1
        # ng = gidx + 1
        # ceiling = ri.rank_floors[max_rank] if max_rank < mask_rank else ri.total
        # if ng >= ceiling:
        #     raise StopIteration
        # return self._with_value(int(np.bitwise_or.reduce(mask_flags[np.array(_positions_of_gidx(mask_rank, ng))])))
        pass

    def _from_global_index(self, gidx: int) -> "RankedBit":
        # mask_rank, mask_flags, val_rank, value_flags = self.expand()
        # if gidx < -1 or gidx >= _get_rank_index(mask_rank).total:
        #     raise IndexError("ranked bit index out of range")
        # if gidx == -1:
        #     return self._with_value(0)
        # pos = _positions_of_gidx(mask_rank, gidx)
        # return self._with_value(int(np.bitwise_or.reduce(mask_flags[np.array(pos)])))
        pass

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
    def from_flags_masks(cls, val: "int | npt.ArrayLike", mask: "int | npt.ArrayLike | None" = None) -> "RankedBit":
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
    demo = RankedBit.from_flags_masks([8,16,64], [4,8,16,32,64,128])

    empty = RankedBit.empty(4)
    print("rank-first ab leer:", [b.bit_value for b in empty.iter_next()])
    print("nur bis Rang 2:   ", [b.bit_value for b in empty.iter_next(max_rank_idx=1)])
    print("plus/minus:       ", (empty + 6).bit_value, (empty + 6 - 2).bit_value)

    ri = _get_rank_index(6)
    anker = [(p, g - ri.rank_floors[2]) for p, g in zip(ri.anchor_pos, ri.anchor_gidx) if len(p) == 3]
    print("umbruch-anker n=6 rang 3:", anker)

    walker = BitGroupWalker(0b0011, 0b1100, max_rank_idx=0)
    print("walker 1 bit/grp: ", [v for v in walker])
    walker = BitGroupWalker(0b0011, 0b1100)
    print("walker komplett:  ", [v for v in walker])
