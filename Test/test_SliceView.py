import numpy as np
import pytest

from clarautils.commonEncoding import get_bitmask
from clarautils.BitFlagArray import Bitty


def test_b_forces_bit_slice_next(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    v = bitty.b[1:4]
    assert v.get_bit_count() == 3
    np.testing.assert_array_equal(v.get_bitwise(), bit_data[:, 1:4])


def test_i_forces_item_slice_next(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    v = bitty.i[1:4]
    assert len(v) == 3
    np.testing.assert_array_equal(v.get_bitwise(), bit_data[1:4])


def test_alternating_slices_flip_mode(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    v = bitty.b[1:4][2:5]
    assert v.get_bit_count() == 3
    assert len(v) == 3
    np.testing.assert_array_equal(v.get_bitwise(), bit_data[2:5, 1:4])


def test_b_property_resets_mode_to_bit(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    v = bitty.b[1:4].b[0:2]
    assert v.get_bit_count() == 2
    np.testing.assert_array_equal(v.get_bitwise(), bit_data[:, 1:4][:, 0:2])


def test_single_bit_index_via_int(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    v = bitty.b[-2]
    assert v.get_bit_count() == 1
    np.testing.assert_array_equal(v.get_bitwise(), bit_data[:, 4:5])


def test_bool_mask_item_selection(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    mask = np.array([False, True, True, False, False, True])
    v = bitty.i[mask]
    assert len(v) == 3
    np.testing.assert_array_equal(v.get_bitwise(), bit_data[[1, 2, 5]])


def test_bool_mask_size_mismatch_raises(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    mask = np.array([False, True, True])
    with pytest.raises(NotImplementedError):
        bitty.i[mask]


def test_step_not_supported(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    with pytest.raises(NotImplementedError):
        bitty.b[0:6:2]


def test_write_sets_bits_in_place(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    sel = bitty.b[1:4][1:4]
    sel.write(np.full_like(sel, sel.get_max_item()))
    expected = bit_data.copy()
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(bitty.get_bitwise(), expected)


def test_write_zero_clears_bits(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    sel = bitty.b[2:5]
    sel.write(np.zeros_like(sel.get_array()))
    expected = bit_data.copy()
    expected[:, 2:5] = 0
    np.testing.assert_array_equal(bitty.get_bitwise(), expected)


def test_setitem_writes_through(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    bitty.b[1:4][1:4] = np.full((3,), 7, dtype=bitty.get_array().dtype)
    expected = bit_data.copy()
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(bitty.get_bitwise(), expected)


def test_get_max_item(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    assert bitty.b[1:4].get_max_item() == get_bitmask(3)
    assert bitty.b[:].get_max_item() == get_bitmask(6)
    assert bitty.b[1:4][1:4].get_max_item() == get_bitmask(3)


def test_remove_bitsplit(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    split = 1
    rest = bitty.rm_b(split)
    np.testing.assert_array_equal(rest[:split], bitty.b[:split])
    np.testing.assert_array_equal(rest[split:], bitty.b[split+1:])


def test_remove_itemsplit(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    split = 2
    rest = bitty.rm_i(split)
    np.testing.assert_array_equal(rest.i[:split], bitty.i[:split])
    np.testing.assert_array_equal(rest.i[split:], bitty.i[split+1:])


def test_group_by_2(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    g_idx = 2
    groups = bitty.group_by_bit(g_idx)
    np.testing.assert_array_equal(groups[1], bitty.i[[0,3,4]][[0,1,3,4,5]])
    np.testing.assert_array_equal(groups[0], bitty.i[[1,2,5]][[0,1,3,4,5]])

def test_group_by_3(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    g_idx = 3
    groups = bitty.group_by_bit(g_idx)
    np.testing.assert_array_equal(groups[1], bitty.i[[1,2,3,4,5]][[0,1,2,4,5]])
    np.testing.assert_array_equal(groups[0], bitty.i[[0]][[0,1,2,4,5]])

def test_group_by_23(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    g_idx = [2,3]
    groups = bitty.group_by_bit(g_idx)
    np.testing.assert_array_equal(groups[1], bitty.i[[1,2,5]][[0,1,4,5]])
    np.testing.assert_array_equal(groups[2], bitty.i[[0]][[0,1,4,5]])
    np.testing.assert_array_equal(groups[3], bitty.i[[3, 4]][[0,1,4,5]])

def test_get_item_indices_offset_slice():
    b = Bitty(np.array([0, 1, 2, 3], dtype=np.uint16), max_bit=2)
    hi, lo = b.split_i(b.b[0] == 1)
    odd, even = hi.split_i(hi.b[1] == 1)
    assert odd.get_item_indices() == [3]
    assert even.get_item_indices() == [2]
    assert hi.get_item_indices() == [2, 3]
    assert lo.get_item_indices() == [0, 1]


def test_get_bit_indices_offset_slice():
    b = Bitty(np.array([0, 1, 2, 3], dtype=np.uint16), max_bit=2)
    assert b.b[0].get_bit_indices() == [0]
    assert b.b[1].get_bit_indices() == [1]
    assert b.b[1:2].get_bit_indices() == [1]
    assert b.b[0][0].get_bit_indices() == [0]
    assert b.i[2:4].b[1].get_bit_indices() == [1]
