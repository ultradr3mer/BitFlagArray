import numpy as np
import pytest

from Mulitslice import Multislice
from commonEncoding import get_number, get_bits


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def test_bits():
    return np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                     [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                     [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])


@pytest.fixture
def test_n_ary(test_bits):
    return get_number(test_bits, axis=1)


# ------------------------------------------------------------------ #
#  Segmentation
# ------------------------------------------------------------------ #

def test_segmentation_two_runs():
    multi = Multislice([1, 2, 3, 5, 6, 7])
    np.testing.assert_array_equal(multi.src_start, [1, 5])
    np.testing.assert_array_equal(multi.stop, [4, 8])
    np.testing.assert_array_equal(multi.lengths, [3, 3])
    assert multi.total_len == 6


def test_segmentation_single_contiguous():
    multi = Multislice([2, 3, 4, 5])
    np.testing.assert_array_equal(multi.src_start, [2])
    np.testing.assert_array_equal(multi.stop, [6])
    np.testing.assert_array_equal(multi.lengths, [4])
    assert multi.total_len == 4


def test_segmentation_single_element():
    multi = Multislice([3])
    np.testing.assert_array_equal(multi.src_start, [3])
    np.testing.assert_array_equal(multi.stop, [4])
    np.testing.assert_array_equal(multi.lengths, [1])
    assert multi.total_len == 1


def test_segmentation_all_singletons():
    multi = Multislice([0, 2, 4, 6])
    np.testing.assert_array_equal(multi.src_start, [0, 2, 4, 6])
    np.testing.assert_array_equal(multi.stop, [1, 3, 5, 7])
    np.testing.assert_array_equal(multi.lengths, [1, 1, 1, 1])
    assert multi.total_len == 4


def test_segmentation_uneven_runs():
    multi = Multislice([0, 1, 5, 6, 7, 8, 10])
    np.testing.assert_array_equal(multi.src_start, [0, 5, 10])
    np.testing.assert_array_equal(multi.stop, [2, 9, 11])
    np.testing.assert_array_equal(multi.lengths, [2, 4, 1])
    assert multi.total_len == 7


# ------------------------------------------------------------------ #
#  get_slices
# ------------------------------------------------------------------ #

def test_get_slices_two_runs():
    multi = Multislice([1, 2, 3, 5, 6, 7])
    slices = multi.get_slices()
    assert slices == [slice(1, 4), slice(5, 8)]


def test_get_slices_single_run():
    multi = Multislice([0, 1, 2])
    assert multi.get_slices() == [slice(0, 3)]


def test_get_slices_singletons():
    multi = Multislice([0, 2, 4])
    assert multi.get_slices() == [slice(0, 1), slice(2, 3), slice(4, 5)]


# ------------------------------------------------------------------ #
#  select_bits  (MSB-first convention like BitFlagArray)
# ------------------------------------------------------------------ #

def test_select_bits_two_runs(test_n_ary):
    multi = Multislice([1, 2, 3, 5, 6, 7])
    result = multi.select_bits(test_n_ary.array, test_n_ary.bit_count)

    # Expected: bits [1,2,3,5,6,7] extracted MSB-first from 11-bit values
    # Row 0: [0,1,0,0,1,0] -> 18
    # Row 1: [1,0,1,1,0,1] -> 45
    # Row 2: [0,0,1,0,1,0] -> 10
    # Row 3: [1,0,0,0,0,1] -> 33
    # Row 4: [0,1,0,1,0,0] -> 20
    expected = np.array([18, 45, 10, 33, 20], dtype=result.dtype)
    np.testing.assert_array_equal(result, expected)


def test_select_bits_contiguous(test_n_ary):
    multi = Multislice([0, 1, 2])
    result = multi.select_bits(test_n_ary.array, test_n_ary.bit_count)

    # Bits 0,1,2 from 11-bit values -> top 3 bits
    # Row 0: [1,0,1] -> 5
    # Row 1: [0,1,0] -> 2
    # Row 2: [1,0,0] -> 4
    # Row 3: [0,1,0] -> 2
    # Row 4: [0,0,1] -> 1
    expected = np.array([5, 2, 4, 2, 1], dtype=result.dtype)
    np.testing.assert_array_equal(result, expected)


def test_select_bits_single_element(test_n_ary):
    multi = Multislice([0])
    result = multi.select_bits(test_n_ary.array, test_n_ary.bit_count)

    # Bit 0 (MSB) from each row
    expected = test_n_ary.array >> (test_n_ary.bit_count - 1)
    np.testing.assert_array_equal(result, expected.astype(result.dtype))


def test_select_bits_all_bits(test_n_ary):
    bc = test_n_ary.bit_count
    multi = Multislice(list(range(bc)))
    result = multi.select_bits(test_n_ary.array, bc)

    # Selecting all bits should return the original values
    np.testing.assert_array_equal(result, test_n_ary.array)


def test_select_bits_preserves_bit_pattern(test_bits, test_n_ary):
    multi = Multislice([1, 2, 3, 5, 6, 7])
    result = multi.select_bits(test_n_ary.array, test_n_ary.bit_count)

    # The selected bits should match the corresponding columns of test_bits
    expected_cols = test_bits[:, [1, 2, 3, 5, 6, 7]]
    result_bits = get_bits(value=result, count=6)
    np.testing.assert_array_equal(result_bits, expected_cols)
