from itertools import combinations
from math import comb

import numpy as np
import pytest

from clarautils.RankedBit import (
    BitGroupWalker,
    RankedBit,
    _get_rank_index,
    bits_combs_by_rank,
    bits_rank_first,
    bits_rank_first_from_flags,
    rank_states,
    rank_states_per_rank,
)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

MASKS = [0b1111, 0b1011, 0b111]


def sub_values(mask):
    return [v for v in range(1, mask + 1) if v & mask == v]


def flags_of(mask):
    return [1 << i for i in range(mask.bit_length()) if (mask >> i) & 1]


def get_comb_idx(comb_items, value_mask, value_rank):
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


# ------------------------------------------------------------------ #
#  Konstruktoren
# ------------------------------------------------------------------ #

def test_from_bits():
    rb = RankedBit.from_bits([1, 0, 1], [1, 2, 3])
    assert (rb.bit_mask, rb.bit_value, rb.bit_count) == (0b111, 0b101, 3)
    rb2 = RankedBit.from_bits([1, 1, 0])
    assert (rb2.bit_mask, rb2.bit_value, rb2.bit_count) == (0b111, 0b110, 3)


def test_empty_and_from_value():
    e = RankedBit.empty(3)
    assert (e.bit_mask, e.bit_value, e.bit_count) == (0b111, 0, 3)
    fv = RankedBit.from_flags_masks(0b101)
    assert (fv.bit_mask, fv.bit_value, fv.bit_count) == (0b111, 0b101, 3)
    fv2 = RankedBit.from_flags_masks([1, 4], 0b111)
    assert (fv2.bit_mask, fv2.bit_value, fv2.bit_count) == (0b111, 0b101, 3)


def test_from_value_not_in_mask():
    with pytest.raises(ValueError):
        RankedBit.from_flags_masks(0b101, 0b1000)
    with pytest.raises(ValueError):
        RankedBit.from_flags_masks([1, 1])
    with pytest.raises(ValueError):
        RankedBit(0b1111, 0b10011, 4).expand()


# ------------------------------------------------------------------ #
#  Index-Kreuzcheck gegen Brute-Force (get_comb_idx)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("mask", MASKS)
def test_index_in_rank_matches_get_comb_idx(mask):
    for v in sub_values(mask):
        rb = RankedBit.from_flags_masks(v, mask)
        val_rank = rb.expand()[2]
        comb_idx, _ = get_comb_idx(flags_of(mask), v, val_rank)
        assert rb.get_comb_info().index_in_rank == comb_idx


@pytest.mark.parametrize("mask", MASKS)
def test_rank_info_floor(mask):
    n = mask.bit_count()
    for v in sub_values(mask):
        r_info = RankedBit.from_flags_masks(v, mask).get_rank_info()
        assert r_info.rank_index == v.bit_count()
        assert r_info.index_floor == sum(comb(n, i) for i in range(1, v.bit_count()))


@pytest.mark.parametrize("mask", MASKS)
def test_global_index_bijection(mask):
    seq = [int(v) for v in bits_rank_first_from_flags(flags_of(mask))]
    gidx = [RankedBit.from_flags_masks(v, mask).global_index for v in seq]
    assert gidx == list(range(len(seq)))


def test_rank_index_anchors_match_table():
    # umbruch-anker (0-basiert) fuer n=6, rang 3 — die wrap-zeilen der tabelle im modulkopf
    ri = _get_rank_index(6)
    ankers = [(p, g - ri.rank_floors[2]) for p, g in zip(ri.anchor_pos, ri.anchor_gidx) if len(p) == 3]
    assert ankers == [
        ((0, 1, 2), 0), ((0, 2, 3), 4), ((0, 3, 4), 7), ((0, 4, 5), 9),
        ((1, 2, 3), 10), ((1, 3, 4), 13), ((1, 4, 5), 15),
        ((2, 3, 4), 16), ((2, 4, 5), 18), ((3, 4, 5), 19),
    ]


# ------------------------------------------------------------------ #
#  Iteration: get_next / iter_next / Rank-Grenze
# ------------------------------------------------------------------ #

def test_get_next_lex_order_4bit():
    seq = [b.bit_value for b in RankedBit.from_flags_masks(0, 0b1111).iter_next()]
    assert seq == [1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15]
    # lex und nicht numerisch: 9 kommt vor 6
    assert seq[4:10] == [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]


@pytest.mark.parametrize("mask", MASKS)
def test_get_next_matches_bits_rank_first(mask):
    expected = [int(v) for v in bits_rank_first_from_flags(flags_of(mask))]
    assert [b.bit_value for b in RankedBit.from_flags_masks(0, mask).iter_next()] == expected


@pytest.mark.parametrize("mask", MASKS)
def test_get_next_matches_brute_force_next(mask):
    flags = flags_of(mask)
    for v in sub_values(mask):
        rb = RankedBit.from_flags_masks(v, mask)
        _, nxt_comb = get_comb_idx(flags, v, v.bit_count())
        if nxt_comb is not None:
            assert rb.get_next().bit_value == sum(nxt_comb)
        elif v == mask:
            with pytest.raises(StopIteration):
                rb.get_next()
        else:
            # letzter im rang -> erster zustand des naechsten rangs
            assert rb.get_next().bit_value == sum(flags[:v.bit_count() + 1])


def test_get_next_boundaries():
    full = RankedBit.from_flags_masks(0b1111, 0b1111)
    with pytest.raises(StopIteration):
        full.get_next()
    first = RankedBit.from_flags_masks(1, 0b1111)
    assert first.get_next().bit_value == 2


def test_get_next_max_rank_idx():
    mask = 0b1111
    seq1 = [b.bit_value for b in RankedBit.from_flags_masks(0, mask).iter_next(max_rank_idx=0)]
    assert seq1 == [1, 2, 4, 8]
    seq2 = [b.bit_value for b in RankedBit.from_flags_masks(0, mask).iter_next(max_rank_idx=1)]
    assert seq2 == [1, 2, 4, 8, 3, 5, 9, 6, 10, 12]
    with pytest.raises(StopIteration):
        RankedBit.from_flags_masks(0b1100, mask).get_next(max_rank_idx=1)
    with pytest.raises(StopIteration):
        RankedBit.from_flags_masks(0b1111, mask).get_next(max_rank_idx=1)


def test_add_sub():
    empty = RankedBit.from_flags_masks(0, 0b1111)
    assert (empty + 0).bit_value == 0
    assert (empty + 1).bit_value == 1
    assert (empty + 15).bit_value == 0b1111
    assert (empty + 6 - 2).bit_value == 0b1000
    assert [(empty + g).global_index for g in range(1, 15)] == list(range(14))
    with pytest.raises(IndexError):
        empty + 16
    with pytest.raises(IndexError):
        empty - 1
    with pytest.raises(IndexError):
        RankedBit.from_flags_masks(0b1111, 0b1111) + 1


# ------------------------------------------------------------------ #
#  Reprs
# ------------------------------------------------------------------ #

def test_reprs():
    rb = RankedBit.from_bits([1, 0, 1], [1, 2, 3])
    assert repr(rb) == "RankedBit(bits='101', mask=0b111, rank=2/3)"
    gapped = RankedBit.from_flags_masks(0b1001, 0b1011)
    assert "bits='1.01'" in repr(gapped)
    info = rb.get_info()
    assert repr(info.rank_info) == "RankInfo(rank=2, floor=3)"
    assert repr(info.comb_info) == "CombInfo(pos=[0, 2], idx=1, n=3)"
    assert repr(info) == "Info(rank=2, floor=3, pos=[0, 2], idx=1, gidx=4)"


# ------------------------------------------------------------------ #
#  Modul-Helper
# ------------------------------------------------------------------ #

def test_bits_rank_first():
    np.testing.assert_array_equal(
        bits_rank_first(4),
        [1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15])


def test_bits_rank_first_big():
    # uint8-Overflow der alten Version
    vals = bits_rank_first(10)
    assert (1 << 9) in vals.tolist()
    assert int(vals.max()) == (1 << 10) - 1


def test_bits_rank_first_from_flags_default():
    # max_r=None-Crash der alten Version
    flags = [1, 2, 4, 8]
    np.testing.assert_array_equal(
        bits_rank_first_from_flags(flags, 1, 3),
        [3, 5, 9, 6, 10, 12, 7, 11, 13, 14])
    assert len(bits_rank_first_from_flags(flags)) == 15


def test_bits_combs_by_rank():
    np.testing.assert_array_equal(bits_combs_by_rank(0b101), [[0, 1], [0, 4], [1, 5]])


def test_rank_states():
    assert rank_states(4, 1) == 0
    assert rank_states(4, 2) == 4
    assert rank_states(4, 3) == 10
    assert rank_states(4, 5) == 15


def test_rank_states_per_rank():
    np.testing.assert_array_equal(rank_states_per_rank(4, slice(0, 2)), [4, 6])
    np.testing.assert_array_equal(rank_states_per_rank(4, 2), [4])
    np.testing.assert_array_equal(rank_states_per_rank(4, np.array([0, 2])), [4, 4])


def test_get_comb_idx_reference():
    assert get_comb_idx([1, 2, 4, 8], 3, 2) == (0, (1, 4))
    assert get_comb_idx([1, 2, 4, 8], 10, 2) == (4, (4, 8))
    assert get_comb_idx([1, 2, 4, 8], 12, 2) == (5, None)
    assert get_comb_idx([1, 2, 4, 8], 15, 4) == (0, None)
    assert get_comb_idx([1, 2, 4, 8], 16, 1) == (-1, None)


# ------------------------------------------------------------------ #
#  BitGroupWalker
# ------------------------------------------------------------------ #

def test_walker_matches_product():
    expected = [a | b for a in (1, 2, 3) for b in (4, 8, 12)]
    walker = BitGroupWalker(0b0011, 0b1100)
    assert [v for v in walker] == expected


def test_walker_state_and_combined():
    walker = BitGroupWalker(0b0011, 0b1100)
    assert next(walker) == 0b0101
    assert [g.bit_value for g in walker.state] == [1, 4]
    assert walker.combined == 0b0101


def test_walker_len():
    assert len(BitGroupWalker(0b0011, 0b1100)) == 9
    assert len(BitGroupWalker(0b0011, 0b1100, max_rank_idx=0)) == 4


def test_walker_max_rank_idx():
    walker = BitGroupWalker(0b0011, 0b1100)
    got = []
    while True:
        try:
            got.append(walker.get_next(max_rank_idx=0))
        except StopIteration:
            break
    assert got == [0b0101, 0b1001, 0b0110, 0b1010]


def test_walker_three_groups():
    walker = BitGroupWalker(0b001, 0b010, 0b100)
    assert [v for v in walker] == [0b111]
    assert len(walker) == 1


def test_walker_exhausted_stays_stopped():
    walker = BitGroupWalker(0b01)
    assert next(walker) == 1
    with pytest.raises(StopIteration):
        next(walker)
    with pytest.raises(StopIteration):
        next(walker)


def test_walker_validation():
    with pytest.raises(ValueError):
        BitGroupWalker(0b0011, 0b0110)
    with pytest.raises(ValueError):
        BitGroupWalker(0b101, 0b011)
    with pytest.raises(ValueError):
        BitGroupWalker(0b00, 0b11)


def test_walker_accepts_rankedbit_and_list():
    # bit_value wird ignoriert, nur die Maske definiert die Gruppe
    rb = RankedBit.from_flags_masks(0b10, 0b11)
    walker = BitGroupWalker(rb, 0b1100)
    assert [v for v in walker][:2] == [0b0101, 0b1001]
    walker2 = BitGroupWalker([rb, 0b1100])
    assert len(walker2) == 9
