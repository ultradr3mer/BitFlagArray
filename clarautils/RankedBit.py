from itertools import combinations
from math import comb
from multiprocessing.managers import public_methods
from typing import Tuple, List, Dict, NamedTuple

import numpy as np
import numpy.typing as npt

from clarautils import get_as_unsigned, get_type_for_bit_count, get_as_fitting
from clarautils.commonEncoding import get_bit_flags, normalize_flags

def bits_combs_by_rank(bit_mask: int)-> npt.NDArray:
    flags = get_bit_flags(bit_mask)
    item_count = flags.size
    return get_as_fitting([[r_idx, sum(c)] for r_idx in range(0, item_count) for c in combinations(flags, r_idx+1)])

def bits_rank_first(bit_count, min_r=0, max_r=None)-> None:
    max_r = bit_count if max_r is None else max_r
    flags = np.array(1 << np.arange(bit_count), dtype=np.uint8)
    # if by_rank:
    #     return { r: [sum(p) for p in combinations(flags, r + 1)]
    #                 for r in range(min_r, max_r) }
    # else:

def bits_rank_first_from_flags(bit_flags, min_r=0, max_r=None) -> np.ndarray:
    return get_as_unsigned([sum(p)
                            for r in range(min_r, max_r)
                            for p in combinations(bit_flags, r + 1)],
                           fit=True)

def get_comb_idx(comb_items, value_mask, value_rank) -> int:
    for i, c in enumerate(combinations(comb_items, value_rank)):
        if sum(c) == value_mask:
            return i
    return -1

def rank_states_per_rank(mask_rank: int, rank_slice: slice | int | []):
    ranks_idx = np.arange(mask_rank)[rank_slice]
    result = [comb(mask_rank, i+1) for i in ranks_idx]
    return result

def rank_states(mask_rank: int, val_rank: int) -> int: #Tuple[int,List[int]]:
    comb_by_rank = [comb(mask_rank, i+1) for i in range(val_rank+1)]
    return sum(comb_by_rank) #, comb_by_rank

class RankedBit(NamedTuple):
    class RankInfo(NamedTuple): # TODO __repr__
        rank_index: int
        index_floor: int
    class CombInfo(NamedTuple): # TODO __repr__
        position: np.ndarray

        @property
        def bitwise_index(self) -> np.ndarray:
            return self.position - np.arange(len(self.position))

        @property
        def index_in_rank(self) -> int:
            return np.sum(self.bitwise_index)

    class Info(NamedTuple): # TODO __repr__ DIese klassen sind eine gute gelegenheit eine visiualierung zu bauen
        rank_info: RankedBit.RankInfo
        comb_info: RankedBit.CombInfo

        @property
        def global_index(self) -> int:
            return self.rank_info.index_floor + self.comb_info.index_in_rank


    bit_mask: int
    bit_value: int
    bit_count: int

    def expand(self) -> Tuple[int, npt.NDArray,int, npt.ArrayLike]:
        mask_flags: np.ndarray = get_bit_flags(self.bit_mask)
        mask_rank = mask_flags.size
        value_flags: np.ndarray = get_bit_flags(self.bit_value)
        rank = value_flags.size
        if not (np.diff(mask_flags) > 0).all() or not (np.diff(value_flags).all() > 0):
            raise ValueError("val and mask mut not contain same flag twice")
        return mask_rank, mask_flags, rank, value_flags

    @staticmethod
    def get_index_for_bit_idx_combination(bit_idx: npt.NDArray):
        diff_from_nullpos = bit_idx - np.arange(len(bit_idx))
        return np.sum(diff_from_nullpos)

    def get_comb_info(self) -> RankedBit.CombInfo:
        mask_rank, mask_flags, val_rank, value_flags = self.expand()

        bitwise_pos = np.where(mask_flags & self.bit_value)[0] ## WILL BECOME IMPORTANT FOR ITERATING
        comb_info = RankedBit.CombInfo(bitwise_pos) # Die methode stimmt noch nicht
        # bitwise_idx = bitwise_pos - np.arange(len(bitwise_pos)) ## ggf get_comb_idx(mask_flags, self.bit_value, val_rank) angleichen
        # combined_id = np.sum(bitwise_idx)
        index_in_comb = get_comb_idx(mask_flags, self.bit_value, val_rank)
        if comb_info.index_in_rank != index_in_comb:
            raise ValueError("val and comb idx differ!")

        return comb_info

    def get_rank_info(self) -> RankedBit.RankInfo:
        mask_rank, mask_flags, val_rank, value_flags = self.expand()

        rank_floor = rank_states(mask_rank, val_rank)
        return RankedBit.RankInfo(val_rank, rank_floor)

    def get_info(self) -> RankedBit.Info:
        r_info = self.get_rank_info()
        c_info = self.get_comb_info()
        return RankedBit.Info(r_info, c_info)

    @staticmethod
    def _build_int(value: int | npt.ArrayLike) -> int:
        if isinstance(value, int):
            return value
        else:
            ipt = np.bitwise_or.reduce(value)
            if not np.sum(value) == ipt:
                raise ValueError("val and mask mut not contain same flag twice")
            return ipt

    @classmethod
    def empty(cls, bit_count: int) -> 'RankedBit':
        return RankedBit(bit_mask=(1 << bit_count)-1, value=0, bit_count=bit_count)

    @classmethod
    def from_value(cls, val: int | npt.ArrayLike, mask: int | npt.ArrayLike = None) -> 'RankedBit':
        v = cls._build_int(val)
        m = cls._build_int(mask)
        b_cnt = m.bit_count()
        return RankedBit(bit_mask=m, value=v,  bit_count=b_cnt)


    @classmethod
    def from_bits(cls, bits: npt.ArrayLike, indices: npt.ArrayLike = None) -> 'RankedBit':
        b = get_as_unsigned(bits,fit=True)
        i = get_as_unsigned(indices,fit=True) if indices is not None else np.arange(b.size)
        t = get_type_for_bit_count(np.max(i))
        bit_count, mask, value = normalize_flags(i, b)
        return RankedBit(mask, value, bit_count)

    def __repr__(self):
        mask_rank, mask_bit, rank, value_bit = self.expand()
        mask_nrs = [f"{i}-{n}" if n >= 0 else "_" for i, n in np.ndenumerate(mask_bit)]
        mask = ", ".join(mask_nrs)
        return f"[{rank}][{mask}][{value_bit}]"


testBit = RankedBit.from_bits([1,0,1],[1,2,3])
print(testBit)
print(testBit.get_info())


testBit = RankedBit.from_bits([1,1,1],[0,1,2])
print(testBit)
print(testBit.get_info())

testBit = RankedBit.from_bits([1,1,0],[0,1,2])
print(testBit)
print(testBit.get_info())


testBit = RankedBit.from_bits([1,0,0],[0,1,2])
print(testBit)
print(testBit.get_info())

class BitGroupWalker:
    bit_groups: Dict[int, List[int]] = {}


if __name__ == '__main__':
    # TODO: SChreibe eine klasse vlt NamedTuple für diese werte
    # test = [16,32,64]
    # mask_rank=len(test)
    # test_mask = sum(test)
    # rnk_bits = bits_combs_by_rank(test_mask)
    # rnk_bit from flags
    # die kombination scheint immerwieder aufzutreten und ich brauche
    # die klasse damit es einheitlch ist und um konversion an einem ort zu machen

    ## TEST ITEATE RANK FIRST
    test = [16, 32, 64]
    mask_rank = len(test)
    test_mask = sum(test)
    rnk_bits = bits_combs_by_rank(test_mask)
    print(rnk_bits)

    ## COMPUTE PREV COMBINATION COUNT FOR RANKS
    comb_sum_by_rank = np.array([rank_states(mask_rank, r_idx) for r_idx in range(mask_rank)])
    print(comb_sum_by_rank)


    ## FIND COMBINATION IDX AND COUNT OF PREV COMB
    def build_row(r_idx, v):
        comb_idx = get_comb_idx(test, v, r_idx)
        comb_idx_floor = comb_sum_by_rank[r_idx]
        return [r_idx, v, comb_idx, comb_idx_floor,
                comb_idx + comb_idx_floor]  ## VOLLSTAENDIGE SKALR REPRESENTATION


    rnk_vals = np.array([build_row(r_idx, v) for r_idx, v in rnk_bits])
    print(rnk_vals)

