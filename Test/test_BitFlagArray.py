import numpy as np
import pytest

from clarautils.BitFlagArray import BitFlagArray, Bitty
from clarautils.commonEncoding import DefinedBit, get_bitwise_entropy
from clarautils.commonTyping import get_type_for_bit_count


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


def test_stack_bit_axis_1(long_data, long_numbers_axis1):
    bitty = Bitty.stack_bit(long_data, axis=1)
    assert bitty.get_bit_count() == 16
    np.testing.assert_array_equal(
        bitty.get_array(),
        np.array(long_numbers_axis1, dtype=bitty.get_array().dtype),
    )


def test_stack_bit_axis_0(long_data, long_numbers_axis0):
    bitty = Bitty.stack_bit(long_data, axis=0)
    assert bitty.get_bit_count() == 6
    np.testing.assert_array_equal(
        bitty.get_array(),
        np.array(long_numbers_axis0, dtype=bitty.get_array().dtype),
    )


def test_stack_bit_axis1(bit_data):
    t = bit_data.T.copy()
    bitty = Bitty.stack_bit(t, axis=1)
    assert bitty.get_bit_count() == 6
    np.testing.assert_array_equal(bitty.get_bitwise(), t)
    np.testing.assert_equal(bitty.get_array()[0], 0b101100)


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


def test_get_defined_bits_mixed():
    bitty = BitFlagArray(np.array([0b0001, 0b0011, 0b0101], dtype=np.uint8), max_bit=4)
    assert bitty.get_defined_bits() == [DefinedBit(0, 0), DefinedBit(3, 1)]


def test_get_bitwise_entropy_method():
    bitty = BitFlagArray(np.array([0b1010, 0b0101], dtype=np.uint8), max_bit=4)
    entropy = bitty.get_bitwise_entropy()
    assert entropy.shape == (4,)
    assert entropy.dtype == np.float32
    np.testing.assert_allclose(entropy, np.ones(4), rtol=1e-6)


def test_get_bitwise_mean_method():
    bitty = BitFlagArray(np.array([0b1010, 0b0001, 0b1111], dtype=np.uint8), max_bit=4)
    np.testing.assert_allclose(bitty.get_bitwise_mean(), [0.5, 0.25, 1.0])
    np.testing.assert_allclose(bitty.get_bitwise_mean(axis=0), [2 / 3, 1 / 3, 2 / 3, 2 / 3])


def test_get_bitwise_entropy_method_on_view():
    bitty = BitFlagArray(np.array([0b1010, 0b0101], dtype=np.uint8), max_bit=4)
    part = bitty.b[0:2]
    np.testing.assert_allclose(part.get_bitwise_entropy(),
                               get_bitwise_entropy(part), rtol=1e-6)


def test_get_defined_bits_all_items_equal():
    bitty = BitFlagArray(np.array([5, 5, 5], dtype=np.uint8), max_bit=4)
    assert bitty.get_defined_bits() == [DefinedBit(0, 0), DefinedBit(1, 1), DefinedBit(2, 0), DefinedBit(3, 1)]


def test_get_defined_bits_single_item():
    bitty = BitFlagArray(np.array([0b101], dtype=np.uint8), max_bit=3)
    assert bitty.get_defined_bits() == [DefinedBit(0, 1), DefinedBit(1, 0), DefinedBit(2, 1)]


def test_get_defined_bits_no_items():
    bitty = BitFlagArray(np.array([], dtype=np.uint8), max_bit=4)
    assert bitty.get_defined_bits() == []


def test_get_defined_bits_matches_bitwise_reference():
    rng = np.random.default_rng(42)
    for bc in (1, 3, 8, 12, 16, 32):
        vals = rng.integers(0, 2 ** bc, size=50, dtype=np.uint64)
        for c in rng.choice(bc, size=max(1, bc // 4), replace=False):
            m = np.uint64(1 << (bc - 1 - int(c)))
            if int(rng.integers(2)):
                vals |= m
            else:
                vals &= ~m
        bitty = Bitty(vals.astype(get_type_for_bit_count(bc)), max_bit=bc)
        bits = bitty.get_bitwise()
        expected = [(c, int(bits[0, c])) for c in range(bc) if (bits[:, c] == bits[0, c]).all()]
        assert bitty.get_defined_bits() == expected


def test_get_defined_bits_on_views():
    bitty = BitFlagArray(np.array([0b0001, 0b0011, 0b0101, 0b0111], dtype=np.uint8), max_bit=4)
    without_defined = bitty.rm_b([0, 3])
    assert without_defined.get_defined_bits() == []
    subset = bitty.i[[0, 1]]
    assert subset.get_defined_bits() == [DefinedBit(0, 0), DefinedBit(1, 0), DefinedBit(3, 1)]
