from dataclasses import dataclass

import numpy as np
import pytest

from commonEncoding import (
    get_bitmask, get_bit_count, get_number, get_bits,
    bits_to_hex, hex_to_bits,
)


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
                 dtype=values.dtype),
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


def test_get_number_long_axis_1(long_data):
    n_bit_ary = get_number(long_data, axis=1)
    assert n_bit_ary.bit_count == 16
    assert len(n_bit_ary.array) == long_data.shape[0]
    expected = [get_number_old(row.tolist()) for row in long_data]
    np.testing.assert_array_equal(
        n_bit_ary.array,
        np.array(expected, dtype=n_bit_ary.array.dtype),
    )


def test_get_number_int():
    bits = get_number([1, 0, 1, 0])
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
