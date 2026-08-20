import numpy as np
import pytest

from common import get_type_for_bit_count, get_type_for_scalar, iter_bits
from commonEncoding import (
    get_bitmask, get_bit_count, get_number, get_bits,
    bits_to_hex, hex_to_bits,
)
from BitFlagArray.BitFlagArray import (
    BitFlagArray, NBitAryTpl, SliceView, Bitty,
    limit_to_bit_count, select_bits, arrange_bits, put_bits,
    merge_slices, normalize_key, get_slice_item_count,
    slice_union, slice_intersection,
)

test = np.array([[1, 0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0, 1],
                 [1, 1, 0, 1, 0, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1],
                 [0, 1, 0, 1, 0, 1]])

def bits_of(obj):
    """2D bit representation (MSB-first) of any NBitArray / SliceView."""
    return obj.get_bitwise()

def test_basic():
    bitty = Bitty.stack_bit(test)

    assert bitty.get_bit_count() == 6
    assert len(bitty) == 6
    assert bitty.get_item_count() == 6
    np.testing.assert_array_equal(bits_of(bitty), test)

    test1 = bitty.b[1:4]
    assert test1.get_bit_count() == 3
    np.testing.assert_array_equal(bits_of(test1), test[:, 1:4])

    test2 = test1.i[1:4]
    np.testing.assert_array_equal(bits_of(test2), test[1:4, 1:4])

    test2b = test1[[1, 2, 3]]
    np.testing.assert_array_equal(bits_of(test2b), test[[1, 2, 3]][:, 1:4])

    selection = bitty.b[1:4][1:4]
    np.testing.assert_array_equal(bits_of(selection), test[1:4, 1:4])

    selection.write(np.full_like(selection, selection.get_max_item()))
    np.testing.assert_array_equal(bits_of(selection), np.ones((3, 3), dtype=int))

    expected = test.copy()
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(bits_of(bitty), expected)

def test_slicing():
    split = -2
    bitty = Bitty.stack_bit(test)

    test1 = bitty.b[:split]
    assert test1.get_bit_count() == 4
    np.testing.assert_array_equal(bits_of(test1), test[:, :4])

    test2 = bitty.b[split:]
    assert test2.get_bit_count() == 2
    np.testing.assert_array_equal(bits_of(test2), test[:, 4:])

    new_bitty = Bitty.stack_bit_arys(test1, test2)
    assert new_bitty.get_bit_count() == 6
    np.testing.assert_array_equal(bits_of(new_bitty), test)

def test_advanced():
    split = -2
    bitty = Bitty.stack_bit(test)

    group_indices = [(bitty.b[split] == b) for b in range(2)]
    np.testing.assert_array_equal(group_indices[0], test[:, 4] == 0)
    np.testing.assert_array_equal(group_indices[1], test[:, 4] == 1)

    grop1 = bitty.i[group_indices[0]]
    np.testing.assert_array_equal(bits_of(grop1), test[test[:, 4] == 0])

    new_bt1 = Bitty.stack_bit_arys(grop1[:split], grop1[split + 1:])
    np.testing.assert_array_equal(
        bits_of(new_bt1),
        np.array([[0, 1, 0, 1, 1],
                  [1, 1, 0, 1, 1],
                  [0, 1, 0, 1, 1]]),
    )

    grop2 = bitty.i[group_indices[1]]
    np.testing.assert_array_equal(bits_of(grop2), test[test[:, 4] == 1])

    new_bt2 = Bitty.stack_bit_arys(grop2[:split], grop2[split + 1:])
    np.testing.assert_array_equal(
        bits_of(new_bt2),
        np.array([[1, 0, 1, 0, 0],
                  [1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1]]),
    )

    empty = Bitty.empty((6, 6))
    assert empty.get_bit_count() == 6
    assert len(empty) == 6
    np.testing.assert_array_equal(empty.get_array(), np.zeros(6, dtype=empty.get_array().dtype))

def test_construct_from_uint_array():
    ary = np.array([1, 2, 3, 7], dtype=np.uint8)
    bitty = BitFlagArray(ary, max_bit=4)
    assert bitty.get_bit_count() == 4
    assert len(bitty) == 4
    np.testing.assert_array_equal(bitty.get_array(), ary)
    np.testing.assert_array_equal(
        bits_of(bitty),
        np.array([[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 1],
                  [0, 1, 1, 1]]),
    )

def test_construct_max_bit_caps_bit_count():
    ary = np.array([255], dtype=np.uint8)
    bitty = BitFlagArray(ary, max_bit=3)
    assert bitty.get_bit_count() == 3
    np.testing.assert_array_equal(bits_of(bitty), np.array([[1, 1, 1]]))

def test_construct_signed_positive_converted_to_unsigned():
    ary = np.array([1, 2, 3], dtype=np.int8)
    bitty = BitFlagArray(ary, max_bit=4)
    assert bitty.get_array().dtype == np.uint8
    assert bitty.get_bit_count() == 4
    np.testing.assert_array_equal(bitty.get_array(), np.array([1, 2, 3], dtype=np.uint8))

def test_construct_from_nbitarray_inherits_bit_count():
    src = BitFlagArray(np.array([5, 6], dtype=np.uint8), max_bit=4)
    wrapped = BitFlagArray(src)
    assert wrapped.get_bit_count() == 4
    np.testing.assert_array_equal(wrapped.get_array(), src.get_array())

def test_construct_from_nbitarray_with_max_bit_clamps():
    src = BitFlagArray(np.array([5, 6], dtype=np.uint8), max_bit=4)
    wrapped = BitFlagArray(src, max_bit=2)
    assert wrapped.get_bit_count() == 2

def test_construct_rejects_non_integer():
    with pytest.raises(ValueError):
        BitFlagArray(np.array([1.5, 2.5], dtype=np.float64))

def test_construct_rejects_2d():
    with pytest.raises(ValueError):
        BitFlagArray(np.array([[1, 2], [3, 4]], dtype=np.uint8))

def test_construct_rejects_negative_signed():
    with pytest.raises(ValueError):
        BitFlagArray(np.array([-1, 2], dtype=np.int8))

def test_stack_bit():
    bitty = Bitty.stack_bit(test)
    assert bitty.get_bit_count() == 6
    np.testing.assert_array_equal(
        bitty.get_array(),
        np.array([0b101010, 0b010101, 0b110101, 0b111111, 0b011111, 0b010101],
                 dtype=bitty.get_array().dtype),
    )
    np.testing.assert_array_equal(bits_of(bitty), test)

def test_stack_bit_axis1():
    t = test.T.copy()
    bitty = Bitty.stack_bit(t, axis=1)
    assert bitty.get_bit_count() == 6
    np.testing.assert_array_equal(bits_of(bitty), t)
    np.testing.assert_array_equal(
        bitty.get_array(),
        np.array([0b101100, 0b011111, 0b100110, 0b011111, 0b100110, 0b011111],
                 dtype=bitty.get_array().dtype),
    )

def test_stack_n_bits_int_bit_count():
    a = np.array([0b101, 0b110], dtype=np.uint8)
    c = np.array([0b11, 0b00], dtype=np.uint8)
    stacked = Bitty.stack_n_bits(a, c, bit_count=3)
    assert stacked.get_bit_count() == 6
    np.testing.assert_array_equal(stacked.get_array(), np.array([43, 48], dtype=stacked.get_array().dtype))

def test_stack_n_bits_list_bit_count():
    a = np.array([0b101, 0b110], dtype=np.uint8)
    c = np.array([0b11, 0b00], dtype=np.uint8)
    stacked = Bitty.stack_n_bits(a, c, bit_count=[3, 2])
    assert stacked.get_bit_count() == 5
    np.testing.assert_array_equal(stacked.get_array(), np.array([23, 24], dtype=stacked.get_array().dtype))

def test_stack_n_bits_unsupported_bit_count_type():
    a = np.array([0b101, 0b110], dtype=np.uint8)
    with pytest.raises(TypeError):
        Bitty.stack_n_bits(a, a, bit_count="three")

def test_stack_bit_arys_preserves_order():
    hi = BitFlagArray(np.array([0b101, 0b110], dtype=np.uint8), max_bit=3)
    lo = BitFlagArray(np.array([0b11, 0b00], dtype=np.uint8), max_bit=2)
    stacked = Bitty.stack_bit_arys(hi, lo)
    assert stacked.get_bit_count() == 5
    np.testing.assert_array_equal(stacked.get_array(), np.array([23, 24], dtype=stacked.get_array().dtype))

def test_stack_bit_arys_mismatched_item_count():
    a = BitFlagArray(np.array([1, 2, 3], dtype=np.uint8), max_bit=4)
    b = BitFlagArray(np.array([4, 5], dtype=np.uint8), max_bit=4)
    with pytest.raises(ValueError):
        Bitty.stack_bit_arys(a, b)

def test_stack_items_concatenates():
    a = BitFlagArray(np.array([1, 2, 3], dtype=np.uint8), max_bit=4)
    b = BitFlagArray(np.array([4, 5, 6], dtype=np.uint8), max_bit=4)
    stacked = Bitty.stack_items(a, b)
    assert stacked.get_bit_count() == 4
    assert len(stacked) == 6
    np.testing.assert_array_equal(stacked.get_array(), np.array([1, 2, 3, 4, 5, 6], dtype=stacked.get_array().dtype))

def test_stack_items_rejects_mismatched_bit_count():
    a = BitFlagArray(np.array([1, 2, 3], dtype=np.uint8), max_bit=4)
    b = BitFlagArray(np.array([4, 5, 6], dtype=np.uint8), max_bit=5)
    with pytest.raises(ValueError):
        Bitty.stack_items(a, b)

def test_empty_shape():
    empty = Bitty.empty((4, 6))
    assert empty.get_bit_count() == 6
    assert len(empty) == 4
    assert empty.get_array().dtype == get_type_for_bit_count(6)
    np.testing.assert_array_equal(empty.get_array(), np.zeros(4, dtype=empty.get_array().dtype))

def test_b_forces_bit_slice_next():
    bitty = Bitty.stack_bit(test)
    v = bitty.b[1:4]
    assert v.get_bit_count() == 3
    np.testing.assert_array_equal(bits_of(v), test[:, 1:4])

def test_i_forces_item_slice_next():
    bitty = Bitty.stack_bit(test)
    v = bitty.i[1:4]
    assert len(v) == 3
    np.testing.assert_array_equal(bits_of(v), test[1:4])

def test_alternating_slices_flip_mode():
    bitty = Bitty.stack_bit(test)
    v = bitty.b[1:4][2:5]
    assert v.get_bit_count() == 3
    assert len(v) == 3
    np.testing.assert_array_equal(bits_of(v), test[2:5, 1:4])

def test_b_property_resets_mode_to_bit():
    bitty = Bitty.stack_bit(test)
    v = bitty.b[1:4].b[0:2]
    assert v.get_bit_count() == 2
    np.testing.assert_array_equal(bits_of(v), test[:, 1:4][:, 0:2])

def test_single_bit_index_via_int():
    bitty = Bitty.stack_bit(test)
    v = bitty.b[-2]
    assert v.get_bit_count() == 1
    np.testing.assert_array_equal(bits_of(v), test[:, 4:5])

def test_bool_mask_item_selection():
    bitty = Bitty.stack_bit(test)
    mask = np.array([False, True, True, False, False, True])
    v = bitty.i[mask]
    assert len(v) == 3
    np.testing.assert_array_equal(bits_of(v), test[[1, 2, 5]])

def test_bool_mask_size_mismatch_raises():
    bitty = Bitty.stack_bit(test)
    mask = np.array([False, True, True])
    with pytest.raises(NotImplementedError):
        bitty.i[mask]

def test_step_not_supported():
    bitty = Bitty.stack_bit(test)
    with pytest.raises(NotImplementedError):
        bitty.b[0:6:2]

def test_write_sets_bits_in_place():
    bitty = Bitty.stack_bit(test)
    sel = bitty.b[1:4][1:4]
    sel.write(np.full_like(sel, sel.get_max_item()))
    expected = test.copy()
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(bits_of(bitty), expected)

def test_write_zero_clears_bits():
    bitty = Bitty.stack_bit(test)
    sel = bitty.b[2:5]
    sel.write(np.zeros_like(sel.get_array()))
    expected = test.copy()
    expected[:, 2:5] = 0
    np.testing.assert_array_equal(bits_of(bitty), expected)

def test_setitem_writes_through():
    bitty = Bitty.stack_bit(test)
    bitty.b[1:4][1:4] = np.full((3,), 7, dtype=bitty.get_array().dtype)
    expected = test.copy()
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(bits_of(bitty), expected)

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
    assert isinstance(k, np.ndarray)
    assert k.tolist() == [3]

def test_normalize_key_int_negative():
    assert normalize_key(-2, 6).tolist() == [4]

def test_normalize_key_list():
    assert normalize_key([0, 2, 4], 6).tolist() == [0, 2, 4]

def test_normalize_key_list_negative():
    assert normalize_key([0, -1, -2], 6).tolist() == [0, 5, 4]

def test_normalize_key_non_bool_ndarray():
    k = normalize_key(np.array([0, 2, 4], dtype=np.uint8), 6)
    assert k.tolist() == [0, 2, 4]

def test_normalize_key_non_bool_ndarray_negative():
    k = normalize_key(np.array([0, -1, -2], dtype=np.int8), 6)
    assert k.tolist() == [0, 5, 4]

def test_normalize_key_bool_mask():
    mask = np.array([False, True, True, False, False, True])
    assert normalize_key(mask, 6).tolist() == [1, 2, 5]

def test_normalize_key_step_unsupported():
    with pytest.raises(NotImplementedError):
        normalize_key(slice(0, 6, 2), 6)

def test_merge_slices_both_slice():
    assert merge_slices(slice(2, 8), slice(1, 4)) == slice(3, 6, 1)

def test_merge_slices_slice_global_array_local():
    np.testing.assert_array_equal(
        merge_slices(slice(2, 8), np.array([1, 2, 3])),
        np.array([3, 4, 5]),
    )

def test_merge_slices_array_global_slice_local():
    np.testing.assert_array_equal(
        merge_slices(np.array([10, 20, 30, 40]), slice(1, 3)),
        np.array([20, 30]),
    )

def test_merge_slices_array_global_array_local():
    np.testing.assert_array_equal(
        merge_slices(np.array([10, 20, 30, 40]), np.array([0, 2])),
        np.array([10, 30]),
    )

def test_select_bits_slice_high_bits():
    data = NBitAryTpl(np.array([0b101010, 0b010101], dtype=np.uint8), 6)
    sel = select_bits(data, slice(1, 4))
    assert sel.get_bit_count() == 3
    np.testing.assert_array_equal(
        sel.get_bitwise(),
        np.array([[0, 1, 0],
                  [1, 0, 1]]),
    )

def test_select_bits_list_picks_msb_positions():
    data = NBitAryTpl(np.array([0b101010], dtype=np.uint8), 6)
    sel = select_bits(data, [0, 2, 4])
    assert sel.get_bit_count() == 3
    np.testing.assert_array_equal(sel.get_bitwise(), np.array([[1, 1, 1]]))

def test_select_bits_list_all_zero_positions():
    data = NBitAryTpl(np.array([0b101010], dtype=np.uint8), 6)
    sel = select_bits(data, [1, 3, 5])
    assert sel.get_bit_count() == 3
    np.testing.assert_array_equal(sel.get_bitwise(), np.array([[0, 0, 0]]))

def test_select_bits_list_out_of_range_raises():
    data = NBitAryTpl(np.array([0b101010], dtype=np.uint8), 6)
    with pytest.raises(ValueError):
        select_bits(data, [0, 7])

def test_select_bits_unsupported_key_type():
    data = NBitAryTpl(np.array([0b101010], dtype=np.uint8), 6)
    with pytest.raises(NotImplementedError):
        select_bits(data, 1.5)

def test_arrange_bits_inverts_put_bits():
    data = NBitAryTpl(np.array([0b101010], dtype=np.uint8), 6)
    indices = [0, 2, 4]
    arranged = arrange_bits(data, indices)
    back = put_bits(arranged, np.array(indices), 6)
    np.testing.assert_array_equal(back, data.get_array())

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

def test_get_max_item():
    bitty = Bitty.stack_bit(test)
    assert bitty.b[1:4].get_max_item() == get_bitmask(3)
    assert bitty.b[:].get_max_item() == get_bitmask(6)
    assert bitty.b[1:4][1:4].get_max_item() == get_bitmask(3)

def test_array_protocol_returns_array():
    bitty = Bitty.stack_bit(test)
    arr = np.asarray(bitty)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, bitty.get_array())

def test_eq_compares_arrays():
    a = BitFlagArray(np.array([1, 2, 3], dtype=np.uint8), max_bit=4)
    b = BitFlagArray(np.array([1, 2, 3], dtype=np.uint8), max_bit=4)
    c = BitFlagArray(np.array([1, 2, 4], dtype=np.uint8), max_bit=4)
    np.testing.assert_array_equal(a == b, np.array([True, True, True]))
    np.testing.assert_array_equal(a == c, np.array([True, True, False]))

def test_repr_contains_bit_length():
    bitty = Bitty.stack_bit(test)
    r = repr(bitty)
    assert "bit_length=6" in r

def test_len_and_item_count():
    bitty = Bitty.stack_bit(test)
    assert len(bitty) == 6
    assert bitty.get_item_count() == 6

def test_get_bitmask_no_start():
    assert int(get_bitmask(6)) == 0b111111
    assert int(get_bitmask(0)) == 0

def test_get_bitmask_with_start():
    assert int(get_bitmask(4, 2)) == 0b111100

def test_get_bit_count():
    assert get_bit_count(0) == 0
    assert get_bit_count(1) == 1
    assert get_bit_count(7) == 3
    assert get_bit_count(8) == 4

def test_get_number_2d():
    values, bc = get_number(test, axis=0)
    assert bc == 6
    np.testing.assert_array_equal(
        values,
        np.array([0b101010, 0b010101, 0b110101, 0b111111, 0b011111, 0b010101],
                 dtype=values.dtype),
    )

def test_get_number_int():
    bits = get_number(0b1010)
    np.testing.assert_array_equal(bits, np.array([1, 0, 1, 0]))

def test_get_bits_ndarray():
    bits = get_bits(np.array([0b101, 0b110], dtype=np.uint8), count=3)
    np.testing.assert_array_equal(bits, np.array([[1, 0, 1], [1, 1, 0]]))

def test_get_bits_str():
    assert get_bits("1010") == [1, 0, 1, 0]

def test_bits_to_hex_and_back():
    for s in ("", "1", "1010", "1111", "10011001", "10101010"):
        assert hex_to_bits(bits_to_hex(s), len(s)) == s

def test_bits_to_hex_padding():
    assert bits_to_hex("1") == "8"
    assert hex_to_bits(bits_to_hex("1"), 1) == "1"

def test_get_type_for_bit_count():
    assert get_type_for_bit_count(8) == np.uint8
    assert get_type_for_bit_count(16) == np.uint16
    assert get_type_for_bit_count(32) == np.uint32
    assert get_type_for_bit_count(64) == np.uint64

def test_get_type_for_bit_count_too_many_raises():
    with pytest.raises(Exception):
        get_type_for_bit_count(65)

def test_get_type_for_scalar():
    assert get_type_for_scalar(255) == np.uint8
    assert get_type_for_scalar(256) == np.uint16

def test_iter_bits():
    data = np.array([0b10110000], dtype=np.uint8)
    assert list(iter_bits(data)) == [1, 0, 1, 1, 0, 0, 0, 0]
