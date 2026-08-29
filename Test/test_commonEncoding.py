from dataclasses import dataclass

import numpy as np
import pytest
from clarautils.commonEncoding import (
    get_bitmask, get_bit_count, get_number, get_bits,
    bits_to_hex, hex_to_bits, arrange_bits, get_bitwise_entropy,
    get_bitwise_mean, get_defined_bits, DefinedBit,
)
from clarautils.BitFlagArray import NBitAryOnly, put_bits, Bitty


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


def test_get_number_axis_0(bit_data):
    n_bit_ary = get_number(bit_data, axis=0)
    assert n_bit_ary.bit_count == 6
    np.testing.assert_array_equal(
        n_bit_ary.array,
        np.array([0b101100, 0b011111, 0b100110, 0b011111, 0b100110, 0b011111],
                 dtype=n_bit_ary.dtype),
    )


def test_get_number_axis_1(bit_data):
    n_bit_ary = get_number(bit_data, axis=1)
    assert n_bit_ary.bit_count == 6
    np.testing.assert_array_equal(
        n_bit_ary.array,
        np.array([0b101010, 0b010101, 0b110101, 0b111111, 0b011111, 0b010101],
                 dtype=n_bit_ary.dtype),
    )

def test_get_number_zero_list(bit_data):
    n_bit_sc = get_number([0]) #[c[0] for c in [(0, 0)]]))
    assert n_bit_sc.bit_count == 1
    np.testing.assert_array_equal(
        n_bit_sc.value,
        np.array([0]))

def test_get_number_3_list(bit_data):
    n_bit_sc = get_number([1, 1])
    assert n_bit_sc.bit_count == 2
    np.testing.assert_array_equal(
        n_bit_sc.value,
        np.array([3]))

def test_get_number_long_axis_0(long_data, long_numbers_axis0):
    n_bit_ary = get_number(long_data, axis=0)
    assert n_bit_ary.bit_count == 6
    assert len(n_bit_ary.array) == long_data.shape[1]
    expected = long_numbers_axis0
    np.testing.assert_array_equal(
        n_bit_ary.array,
        np.array(expected, dtype=n_bit_ary.dtype),
    )


def test_get_number_long_axis_1(long_data, long_numbers_axis1):
    n_bit_ary = get_number(long_data, axis=1)
    assert n_bit_ary.bit_count == 16
    assert len(n_bit_ary.array) == long_data.shape[0]
    np.testing.assert_array_equal(
        n_bit_ary.array,
        np.array(long_numbers_axis1, dtype=n_bit_ary.array.dtype),
    )


def test_get_number_1d_list():
    n_bit_sc = get_number([1, 0, 1, 0])
    assert n_bit_sc.bit_count == 4
    assert n_bit_sc.value == 0b1010


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


def test_arrange_bits_inverts_put_bits():
    data = NBitAryOnly(np.array([0b101010], dtype=np.uint8), 6)
    indices = [0, 2, 4]
    arranged = arrange_bits(data, indices)
    back = put_bits(arranged, np.array(indices), 6)
    np.testing.assert_array_equal(back, data.get_array())


def test_get_bitwise_entropy_plain_2arg():
    v = np.array([0b00, 0b01, 0b01, 0b11], dtype=np.uint8)
    entropy = get_bitwise_entropy(v, 2)
    assert entropy.shape == (2,)
    assert entropy.dtype == np.float32
    expected = -0.25 * np.log2(0.25) - 0.75 * np.log2(0.75)
    np.testing.assert_allclose(entropy, [expected, expected], rtol=1e-6)


def test_get_bitwise_entropy_msb_first():
    v = np.array([0b10, 0b11], dtype=np.uint8)
    entropy = get_bitwise_entropy(v, 2)
    np.testing.assert_allclose(entropy, [0.0, 1.0], rtol=1e-6)


def test_get_bitwise_entropy_bitty():
    b = Bitty(np.array([0b1010, 0b0101], dtype=np.uint8), max_bit=4)
    entropy = get_bitwise_entropy(b)
    assert entropy.shape == (4,)
    np.testing.assert_allclose(entropy, np.ones(4), rtol=1e-6)


def test_get_bitwise_entropy_nbit_ary_only():
    data = NBitAryOnly(np.array([0b01, 0b01], dtype=np.uint8), 2)
    np.testing.assert_allclose(get_bitwise_entropy(data), [0.0, 0.0], rtol=1e-6)


def test_get_bitwise_entropy_slice_view():
    b = Bitty(np.array([0b1010, 0b0101], dtype=np.uint8), max_bit=4)
    sv = b.b[0:2]
    entropy = get_bitwise_entropy(sv)
    assert entropy.shape == (2,)
    np.testing.assert_allclose(entropy, np.ones(2), rtol=1e-6)


def test_get_bitwise_entropy_empty():
    result = get_bitwise_entropy(np.array([], dtype=np.uint8), 5)
    np.testing.assert_array_equal(result, np.zeros(5))
    assert result.dtype == np.float32


def test_get_bitwise_entropy_requires_bit_count():
    with pytest.raises(ValueError):
        get_bitwise_entropy(np.array([1, 2], dtype=np.uint8))


def test_get_bitwise_mean_per_item():
    v = np.array([0b1010, 0b0001, 0b1111], dtype=np.uint8)
    np.testing.assert_allclose(get_bitwise_mean(v, 4), [0.5, 0.25, 1.0])


def test_get_bitwise_mean_per_pos():
    v = np.array([0b10, 0b11], dtype=np.uint8)
    np.testing.assert_allclose(get_bitwise_mean(v, 2, axis=0), [1.0, 0.5])


def test_get_bitwise_mean_bitty_matches_unpack():
    rng = np.random.default_rng(7)
    vals = rng.integers(0, 2 ** 12, size=200, dtype=np.uint64).astype(np.uint16)
    b = Bitty(vals, max_bit=12)
    np.testing.assert_allclose(get_bitwise_mean(b), np.mean(b.get_bitwise(), axis=1))
    np.testing.assert_allclose(get_bitwise_mean(b, axis=0), np.mean(b.get_bitwise(), axis=0))


def test_get_bitwise_mean_nbit_ary_only():
    data = NBitAryOnly(np.array([0b01, 0b11], dtype=np.uint8), 2)
    np.testing.assert_allclose(get_bitwise_mean(data), [0.5, 1.0])


def test_get_bitwise_mean_empty():
    result = get_bitwise_mean(np.array([], dtype=np.uint8), 5)
    assert result.shape == (0,)
    result0 = get_bitwise_mean(np.array([], dtype=np.uint8), 5, axis=0)
    np.testing.assert_array_equal(result0, np.zeros(5))


def test_get_bitwise_mean_requires_bit_count():
    with pytest.raises(ValueError):
        get_bitwise_mean(np.array([1, 2], dtype=np.uint8))


def test_get_bitwise_mean_invalid_axis():
    with pytest.raises(ValueError):
        get_bitwise_mean(np.array([1, 2], dtype=np.uint8), 2, axis=2)


def test_get_defined_bits_plain_2arg():
    v = np.array([0b0001, 0b0011, 0b0101], dtype=np.uint8)
    assert get_defined_bits(v, 4) == [DefinedBit(0, 0), DefinedBit(3, 1)]


def test_get_defined_bits_nbit_ary():
    data = NBitAryOnly(np.array([0b01, 0b01], dtype=np.uint8), 2)
    assert get_defined_bits(data) == [DefinedBit(0, 0), DefinedBit(1, 1)]


def test_get_defined_bits_empty():
    assert get_defined_bits(np.array([], dtype=np.uint8), 4) == []


def test_get_defined_bits_requires_bit_count():
    with pytest.raises(ValueError):
        get_defined_bits(np.array([1, 2], dtype=np.uint8))

