import numpy as np
import pytest

from common import get_type_for_bit_count, get_type_for_scalar, iter_bits


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
