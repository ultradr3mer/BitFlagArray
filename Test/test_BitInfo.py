import numpy as np
import pytest

from clarautils.BitInfo import BitInfo
from clarautils.commonEncoding import get_bits, get_bit_flags, get_number


def test_mode_from_str():
    assert BitInfo.Mode("flags") is BitInfo.Mode.FLAGS
    with pytest.raises(ValueError):
        BitInfo.Mode("nope")


def test_bits_scalar_derived_count():
    np.testing.assert_array_equal(BitInfo.from_value(0b1010, mode="bits"), [1, 0, 1, 0])


def test_bits_scalar_explicit_count():
    res = BitInfo.from_value(0b1010, 8, "bits")
    np.testing.assert_array_equal(res, [0, 0, 0, 0, 1, 0, 1, 0])


def test_bits_ndarray():
    res = BitInfo.from_value(np.array([0b101, 0b110], dtype=np.uint8), 3, "bits")
    np.testing.assert_array_equal(res, [[1, 0, 1], [1, 1, 0]])


def test_bits_list():
    res = BitInfo.from_value([0b1010, 0b110], 4, "bits")
    np.testing.assert_array_equal(res, [[1, 0, 1, 0], [0, 1, 1, 0]])


def test_flags_scalar():
    np.testing.assert_array_equal(BitInfo.from_value(0b1010, mode="flags"), [2, 8])


def test_flags_ndarray_flat():
    res = BitInfo.from_value(np.array([0b1000, 0b0001], dtype=np.uint8), 4, "flags")
    np.testing.assert_array_equal(res, [8, 1])


def test_indices_scalar():
    np.testing.assert_array_equal(BitInfo.from_value(0b1010, mode="indices"), [1, 3])


def test_indices_ndarray_flat():
    res = BitInfo.from_value(np.array([0b1000, 0b0001], dtype=np.uint8), 4, "indices")
    np.testing.assert_array_equal(res, [3, 0])


def test_b_count_scalar():
    assert BitInfo.from_value(0b1010, mode="count") == 4
    assert BitInfo.from_value(0, mode="count") == 0


def test_b_count_within_explicit_bit_count():
    assert BitInfo.from_value(0b101, 8, "count") == 3
    assert BitInfo.from_value(0b1010, 2, "count") == 2  # auf bit_count maskiert


def test_b_count_ndarray_per_entry():
    res = BitInfo.from_value(np.array([0b101, 0b110], dtype=np.uint8), mode="count")
    np.testing.assert_array_equal(res, [3, 3])


def test_b_count_ndarray_zero_entries():
    res = BitInfo.from_value(np.array([0, 0b101], dtype=np.uint8), mode="count")
    np.testing.assert_array_equal(res, [0, 3])
    res = BitInfo.from_value(np.zeros(3, dtype=np.uint8), mode="count")
    np.testing.assert_array_equal(res, [0, 0, 0])


def test_b_count_ndarray_2d_flat():
    res = BitInfo.from_value(np.array([[0b101, 0b110], [0b1, 0b1000]], dtype=np.uint32),
                             mode="count")
    np.testing.assert_array_equal(res, [3, 3, 1, 4])


def test_floats_rejected_by_default():
    # auch ganzzahlige Floats (uint64 - int64 -> float64) werden abgelehnt
    with pytest.raises(TypeError):
        BitInfo.from_value(np.array([2.0, 2.0, 2.0]), mode="count")


def test_allow_integer_floats():
    res = BitInfo.from_value(np.array([2.0, 2.0, 2.0]), mode="count",
                             allow_integer_floats=True)
    np.testing.assert_array_equal(res, [2, 2, 2])
    with pytest.raises(TypeError):
        BitInfo.from_value(np.array([2.5]), mode="count", allow_integer_floats=True)


def test_allow_integer_floats_scalar():
    assert BitInfo.from_value(2.0, mode="count", allow_integer_floats=True) == 2
    np.testing.assert_array_equal(
        BitInfo.from_value(np.float64(10), mode="flags", allow_integer_floats=True), [2, 8])
    np.testing.assert_array_equal(
        BitInfo.from_value(10.0, mode="bits", allow_integer_floats=True), [1, 0, 1, 0])
    with pytest.raises(TypeError):
        BitInfo.from_value(2.5, mode="count", allow_integer_floats=True)


def test_delegates_allow_integer_floats():
    res = get_bit_flags(np.array([10.0, 3.0]), allow_integer_floats=True)
    np.testing.assert_array_equal(res, [2, 8, 1, 2])


def test_common_n_bit_sc():
    sc = get_number([1, 0, 1, 0])
    np.testing.assert_array_equal(BitInfo.from_value(sc, mode="bits"), [1, 0, 1, 0])
    np.testing.assert_array_equal(BitInfo.from_value(sc, mode="flags"), [2, 8])
    np.testing.assert_array_equal(BitInfo.from_value(sc, mode="indices"), [1, 3])
    assert BitInfo.from_value(sc, mode="count") == 4


def test_from_string_bits():
    assert BitInfo.from_string("1010") == [1, 0, 1, 0]
    assert BitInfo.from_string("1010", "bits") == [1, 0, 1, 0]


def test_from_string_bits_multiple():
    res = BitInfo.from_string(["1010", "1100"], "bits")
    np.testing.assert_array_equal(res, [[1, 0, 1, 0], [1, 1, 0, 0]])


def test_from_string_other_modes():
    np.testing.assert_array_equal(BitInfo.from_string("1010", "flags"), [2, 8])
    np.testing.assert_array_equal(BitInfo.from_string("1010", "indices"), [1, 3])
    assert BitInfo.from_string("1010", "count") == 4


def test_from_string_multiple():
    res = BitInfo.from_string(["1010", "1100"], "flags")
    np.testing.assert_array_equal(res, [2, 8, 4, 8])


def test_from_string_long_only_bits():
    long_s = "1" * 129
    assert BitInfo.from_string(long_s, "bits")[:3] == [1, 1, 1]
    with pytest.raises(Exception):
        BitInfo.from_string(long_s, "flags")


def test_delegates_get_bits():
    res = get_bits(np.array([0b101, 0b110], dtype=np.uint8), count=3)
    np.testing.assert_array_equal(res, [[1, 0, 1], [1, 1, 0]])
    assert get_bits("1010") == [1, 0, 1, 0]


def test_delegates_get_bit_flags():
    np.testing.assert_array_equal(get_bit_flags(0b1010), [2, 8])
    res = get_bit_flags(np.array([0b1000, 0b0001], dtype=np.uint8), 4)
    np.testing.assert_array_equal(res, [8, 1])


def test_delegates_get_bit_flags_str():
    np.testing.assert_array_equal(get_bit_flags("1010"), [2, 8])
