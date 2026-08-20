import numpy as np
import pytest

from commonEncoding import get_bitmask
from BitFlagArray.BitFlagArray import Bitty


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
