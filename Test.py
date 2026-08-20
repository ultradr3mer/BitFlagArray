# content of test_sample.py
import numpy as np

from commonEncoding import get_number
from BitFlagArray.BitFlagArray import BitFlagArray, NBitAryTpl, NBitArray, SliceView, Bitty

test = np.array([[1, 0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0, 1],
                 [1, 1, 0, 1, 0, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1],
                 [0, 1, 0, 1, 0, 1]])

def test_basic():
    bitty = Bitty.stack_bit(test)
    print(bitty)
    test1 = bitty.b[1:4]
    print(test1)
    test2 = test1.i[1:4]
    print(test2)
    test2 = test1[[1,2,3]]
    print(test2)
    selection = bitty.b[1:4][1:4]
    print(selection)
    selection.write(np.full_like(selection, selection.get_max_item()))
    print(selection)
    print(bitty)
    # AUSGABE
    # <class '__main__.BitFlagArray'>, bit_length=6,
    # [[1 0 1 0 1 0]
    #  [0 1 0 1 0 1]
    #  [1 1 0 1 0 1]
    #  [1 1 1 1 1 1]
    #  [0 1 1 1 1 1]
    #  [0 1 0 1 0 1]]
    # <class '__main__.SliceView'>, bit_length=3,
    # [[0 1 0]
    #  [1 0 1]
    #  [1 0 1]
    #  [1 1 1]
    #  [1 1 1]
    #  [1 0 1]]
    # <class '__main__.SliceView'>, bit_length=3,
    # [[1 0 1]
    #  [1 0 1]
    #  [1 1 1]]
    # <class '__main__.SliceView'>, bit_length=3,
    # [[1 0 1]
    #  [1 0 1]
    #  [1 1 1]]
    # <class '__main__.SliceView'>, bit_length=3,
    # [[1 0 1]
    #  [1 0 1]
    #  [1 1 1]]
    # <class '__main__.SliceView'>, bit_length=3,
    # [[1 1 1]
    #  [1 1 1]
    #  [1 1 1]]
    # <class '__main__.BitFlagArray'>, bit_length=6,
    # [[1 0 1 0 1 0]
    #  [0 1 1 1 0 1]
    #  [1 1 1 1 0 1]
    #  [1 1 1 1 1 1]
    #  [0 1 1 1 1 1]
    #  [0 1 0 1 0 1]]

def test_slicing():
    split = -2
    bitty = Bitty.stack_bit(test)
    test1 = bitty.b[:split]
    test1.read()
    print(test1)
    test2 = bitty.b[split:]
    print(test2)
    new_bitty = Bitty.stack_bit_arys(test1, test2)
    print(new_bitty)
    # <class '__main__.SliceView'>, bit_length=4,
    # [[1 0 1 0]
    #  [0 1 0 1]
    #  [1 1 0 1]
    #  [1 1 1 1]
    #  [0 1 1 1]
    #  [0 1 0 1]]
    # <class '__main__.SliceView'>, bit_length=2,
    # [[1 0]
    #  [0 1]
    #  [0 1]
    #  [1 1]
    #  [1 1]
    #  [0 1]]
    # <class '__main__.BitFlagArray'>, bit_length=6,
    # [[1 0 1 0 1 0]
    #  [0 1 0 1 0 1]
    #  [1 1 0 1 0 1]
    #  [1 1 1 1 1 1]
    #  [0 1 1 1 1 1]
    #  [0 1 0 1 0 1]]


def test_advanced():
    split = -2
    bitty = Bitty.stack_bit(test)
    group_indices = [(bitty.b[split] == b) for b in range(2)]  # Or np.nonzero(bitty.b[split] == b)[0]
    print(group_indices)
    grop1 = bitty.i[group_indices[0]]
    print(grop1)
    new_bt1 = Bitty.stack_bit_arys(grop1[:split], grop1[split + 1:])
    print(new_bt1)
    grop2 = bitty.i[group_indices[1]]
    print(grop2)
    new_bt2 = Bitty.stack_bit_arys(grop2[:split], grop2[split + 1:])
    print(new_bt2)
    empty = Bitty.empty((6, 6))

    # AUSGABE
    # [array([False,  True,  True, False, False,  True]), array([ True, False, False,  True,  True, False])]
    # <class '__main__.SliceView'>, bit_length=6,
    # [[0 1 0 1 0 1]
    #  [1 1 0 1 0 1]
    #  [0 1 0 1 0 1]]
    # <class '__main__.BitFlagArray'>, bit_length=5,
    # [[0 1 0 1 1]
    #  [1 1 0 1 1]
    #  [0 1 0 1 1]]
    # <class '__main__.SliceView'>, bit_length=6,
    # [[1 0 1 0 1 0]
    #  [1 1 1 1 1 1]
    #  [0 1 1 1 1 1]]
    # <class '__main__.BitFlagArray'>, bit_length=5,
    # [[1 0 1 0 0]
    #  [1 1 1 1 1]
    #  [0 1 1 1 1]]