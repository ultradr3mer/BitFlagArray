import numpy as np
import pytest

from BitFlagArray import (
    NBitAryOnly,
    limit_to_bit_count, select_bits, put_bits,
    merge_slices, normalize_key, get_indices, get_slice_item_count,
    slice_union, slice_intersection,
)


def test_slice_union():
    assert slice_union(slice(1, 4), slice(2, 6)) == slice(1, 6)
    assert slice_union(slice(2, 5), slice(0, 3)) == slice(0, 5)


def test_slice_intersection():
    assert slice_intersection(slice(1, 4), slice(2, 6)) == slice(2, 4)
    assert slice_intersection(slice(1, 2), slice(3, 4)) == slice(0, 0)


def test_get_slice_item_count():
    assert get_slice_item_count(slice(2, 7)) == 5
    assert get_slice_item_count(np.array([1, 2, 3])) == 3


def test_normalize_key_slice():
    assert normalize_key(slice(1, 4), 6) == slice(1, 4, 1)


def test_normalize_key_slice_open():
    assert normalize_key(slice(None), 6) == slice(0, 6, 1)


def test_normalize_key_slice_negative():
    assert normalize_key(slice(None, -2), 6) == slice(0, 4, 1)


def test_normalize_key_int():
    k = normalize_key(3, 6)
    assert k == slice(3, 4, 1)


def test_normalize_key_int_negative():
    assert normalize_key(-2, 6) == slice(4, 5, 1)


def test_normalize_key_list():
    k = normalize_key([0, 2, 4], 6)
    assert get_indices(k, 3) == [0, 2, 4]


def test_normalize_key_list_negative():
    k = normalize_key([0, -1, -2], 6)
    assert get_indices(k, 3) == [0, 5, 4]


def test_normalize_key_non_bool_ndarray():
    k = normalize_key(np.array([0, 2, 4], dtype=np.uint8), 6)
    assert get_indices(k, 3) == [0, 2, 4]


def test_normalize_key_non_bool_ndarray_negative():
    k = normalize_key(np.array([0, -1, -2], dtype=np.int8), 6)
    assert get_indices(k, 3) == [0, 5, 4]


def test_normalize_key_bool_mask():
    mask = np.array([False, True, True, False, False, True])
    k = normalize_key(mask, 6)
    assert get_indices(k, 3) == [1, 2, 5]


def test_normalize_key_step_unsupported():
    with pytest.raises(NotImplementedError):
        normalize_key(slice(0, 6, 2), 6)


def test_merge_slices_both_slice():
    assert merge_slices(slice(2, 8), slice(1, 4)) == slice(3, 6, 1)


def test_merge_slices_slice_global_array_local():
    result = merge_slices(slice(2, 8), np.array([1, 2, 3]))
    assert get_indices(result, 8) == [3, 4, 5]


def test_merge_slices_array_global_slice_local():
    result = merge_slices(np.array([10, 20, 30, 40]), slice(1, 3))
    assert get_indices(result, 40) == [20, 30]


def test_merge_slices_array_global_array_local():
    result = merge_slices(np.array([10, 20, 30, 40]), np.array([0, 2]))
    assert get_indices(result, 40) == [10, 30]


def test_select_bits_slice_high_bits():
    data = NBitAryOnly(np.array([0b101010, 0b010101], dtype=np.uint8), 6)
    sel = select_bits(data, slice(1, 4))
    assert sel.get_bit_count() == 3
    np.testing.assert_array_equal(
        sel.get_bitwise(),
        np.array([[0, 1, 0],
                  [1, 0, 1]]),
    )


def test_select_bits_list_picks_msb_positions():
    data = NBitAryOnly(np.array([0b101010], dtype=np.uint8), 6)
    sel = select_bits(data, [0, 2, 4])
    assert sel.get_bit_count() == 3
    np.testing.assert_array_equal(sel.get_bitwise(), np.array([[1, 1, 1]]))


def test_select_bits_list_all_zero_positions():
    data = NBitAryOnly(np.array([0b101010], dtype=np.uint8), 6)
    sel = select_bits(data, [1, 3, 5])
    assert sel.get_bit_count() == 3
    np.testing.assert_array_equal(sel.get_bitwise(), np.array([[0, 0, 0]]))


def test_select_bits_list_out_of_range_raises():
    data = NBitAryOnly(np.array([0b101010], dtype=np.uint8), 6)
    with pytest.raises(ValueError):
        select_bits(data, [0, 7])


def test_select_bits_unsupported_key_type():
    data = NBitAryOnly(np.array([0b101010], dtype=np.uint8), 6)
    with pytest.raises(NotImplementedError):
        select_bits(data, 1.5)


def test_put_bits_places_value_at_positions():
    val = np.array([0b111], dtype=np.uint8)
    np.testing.assert_array_equal(put_bits(val, np.array([0, 2, 4]), 6), np.array([42]))
    np.testing.assert_array_equal(put_bits(val, np.array([1, 3, 5]), 6), np.array([21]))


def test_limit_to_bit_count_truncates():
    out = limit_to_bit_count(np.array([0b11111], dtype=np.uint8), 3)
    assert out.get_bit_count() == 3
    np.testing.assert_array_equal(out.get_array(), np.array([0b111], dtype=np.uint8))


def test_limit_to_bit_count_slice():
    out = limit_to_bit_count(np.array([0b101010], dtype=np.uint8), slice(1, 4))
    assert out.get_bit_count() == 3
    np.testing.assert_array_equal(out.get_array(), np.array([0b010], dtype=np.uint8))


def test_limit_to_bit_count_overflow_raises():
    with pytest.raises(Exception):
        limit_to_bit_count(np.array([0b1000], dtype=np.uint8), 2, check_overflow=True)


def test_limit_to_bit_count_no_check_silent_truncation():
    out = limit_to_bit_count(np.array([0b1000], dtype=np.uint8), 2, check_overflow=False)
    np.testing.assert_array_equal(out.get_array(), np.array([0b00], dtype=np.uint8))
