import numpy as np
import pytest

from BitFlagArray.BitFlagArray import Bitty


def test_basic(bit_data):
    bitty = Bitty.stack_bit(bit_data)

    assert bitty.get_bit_count() == 6
    assert len(bitty) == 6
    assert bitty.get_item_count() == 6
    np.testing.assert_array_equal(bitty.get_bitwise(), bit_data)

    test1 = bitty.b[1:4]
    assert test1.get_bit_count() == 3
    np.testing.assert_array_equal(test1.get_bitwise(), bit_data[:, 1:4])

    test2 = test1.i[1:4]
    np.testing.assert_array_equal(test2.get_bitwise(), bit_data[1:4, 1:4])

    test2b = test1[[1, 2, 3]]
    np.testing.assert_array_equal(test2b.get_bitwise(), bit_data[[1, 2, 3]][:, 1:4])

    selection = bitty.b[1:4][1:4]
    np.testing.assert_array_equal(selection.get_bitwise(), bit_data[1:4, 1:4])

    selection.write(np.full_like(selection, selection.get_max_item()))
    np.testing.assert_array_equal(selection.get_bitwise(), np.ones((3, 3), dtype=int))

    expected = bit_data.copy()
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(bitty.get_bitwise(), expected)


def test_slicing(bit_data):
    split = -2
    bitty = Bitty.stack_bit(bit_data)

    test1 = bitty.b[:split]
    assert test1.get_bit_count() == 4
    np.testing.assert_array_equal(test1.get_bitwise(), bit_data[:, :4])

    test2 = bitty.b[split:]
    assert test2.get_bit_count() == 2
    np.testing.assert_array_equal(test2.get_bitwise(), bit_data[:, 4:])

    new_bitty = Bitty.stack_bit_arys(test1, test2)
    assert new_bitty.get_bit_count() == 6
    np.testing.assert_array_equal(new_bitty.get_bitwise(), bit_data)


def test_advanced(bit_data):
    split = -2
    bitty = Bitty.stack_bit(bit_data)

    group_indices = [(bitty.b[split] == b) for b in range(2)]
    np.testing.assert_array_equal(group_indices[0], bit_data[:, 4] == 0)
    np.testing.assert_array_equal(group_indices[1], bit_data[:, 4] == 1)

    grop1 = bitty.i[group_indices[0]]
    np.testing.assert_array_equal(grop1.get_bitwise(), bit_data[bit_data[:, 4] == 0])

    new_bt1 = Bitty.stack_bit_arys(grop1[:split], grop1[split + 1:])
    np.testing.assert_array_equal(
        new_bt1.get_bitwise(),
        np.array([[0, 1, 0, 1, 1],
                  [1, 1, 0, 1, 1],
                  [0, 1, 0, 1, 1]]),
    )

    grop2 = bitty.i[group_indices[1]]
    np.testing.assert_array_equal(grop2.get_bitwise(), bit_data[bit_data[:, 4] == 1])

    new_bt2 = Bitty.stack_bit_arys(grop2[:split], grop2[split + 1:])
    np.testing.assert_array_equal(
        new_bt2.get_bitwise(),
        np.array([[1, 0, 1, 0, 0],
                  [1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1]]),
    )

    empty = Bitty.empty((6, 6))
    assert empty.get_bit_count() == 6
    assert len(empty) == 6
    np.testing.assert_array_equal(empty.get_array(), np.zeros(6, dtype=empty.get_array().dtype))
