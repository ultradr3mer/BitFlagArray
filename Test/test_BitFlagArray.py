import numpy as np
import pytest

from BitFlagArray import BitFlagArray, Bitty
from common import get_type_for_bit_count
from commonEncoding import get_number_old


def test_construct_from_uint_array():
    ary = np.array([1, 2, 3, 7], dtype=np.uint8)
    bitty = BitFlagArray(ary, max_bit=4)
    assert bitty.get_bit_count() == 4
    assert len(bitty) == 4
    np.testing.assert_array_equal(bitty.get_array(), ary)
    np.testing.assert_array_equal(
        bitty.get_bitwise(),
        np.array([[0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 1],
                  [0, 1, 1, 1]]),
    )


def test_construct_max_bit_caps_bit_count():
    ary = np.array([255], dtype=np.uint8)
    bitty = BitFlagArray(ary, max_bit=3)
    assert bitty.get_bit_count() == 3
    np.testing.assert_array_equal(bitty.get_bitwise(), np.array([[1, 1, 1]]))


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


def test_stack_bit_axis_1(long_data):
    bitty = Bitty.stack_bit(long_data, axis=1)
    assert bitty.get_bit_count() == 16
    expected = np.array(
        [get_number_old(row.tolist()) for row in long_data],
        dtype=bitty.get_array().dtype,
    )
    np.testing.assert_array_equal(bitty.get_array(), expected)


def test_stack_bit_axis_0(long_data):
    bitty = Bitty.stack_bit(long_data, axis=0)
    assert bitty.get_bit_count() == 6
    expected = np.array(
        [get_number_old(long_data[:, j].tolist()) for j in range(long_data.shape[1])],
        dtype=bitty.get_array().dtype,
    )
    np.testing.assert_array_equal(bitty.get_array(), expected)


def test_stack_bit_axis1(bit_data):
    t = bit_data.T.copy()
    bitty = Bitty.stack_bit(t, axis=1)
    assert bitty.get_bit_count() == 6
    np.testing.assert_array_equal(bitty.get_bitwise(), t)
    np.testing.assert_equal(
        bitty.get_array()[0],
        get_number_old([1, 0, 1, 1, 0, 0]),
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


def test_array_protocol_returns_array(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    arr = np.asarray(bitty)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, bitty.get_array())


def test_eq_compares_arrays():
    a = BitFlagArray(np.array([1, 2, 3], dtype=np.uint8), max_bit=4)
    b = BitFlagArray(np.array([1, 2, 3], dtype=np.uint8), max_bit=4)
    c = BitFlagArray(np.array([1, 2, 4], dtype=np.uint8), max_bit=4)
    np.testing.assert_array_equal(a == b, np.array([True, True, True]))
    np.testing.assert_array_equal(a == c, np.array([True, True, False]))


def test_repr_contains_bit_length(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    r = repr(bitty)
    assert "bit_length=6" in r


def test_len_and_item_count(bit_data):
    bitty = Bitty.stack_bit(bit_data)
    assert len(bitty) == 6
    assert bitty.get_item_count() == 6
