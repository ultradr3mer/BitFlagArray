# content of test_sample.py
import numpy as np

from OpenCode.commonEncoding import get_number
from OpenCode.BitFlagArray import BitFlagArray, NBitAryTpl, NBitArray, SliceView, Bitty

test = np.array((get_number([1, 0, 1, 0, 1, 0]),
                 get_number([0, 1, 0, 1, 0, 1]),
                 get_number([1, 1, 0, 1, 0, 1]),
                 get_number([1, 1, 1, 1, 1, 1]),
                 get_number([0, 1, 1, 1, 1, 1]),
                 get_number([0, 1, 0, 1, 0, 1])))

def test_answer():
    bitty = Bitty(test, max_bit=6)
    print(bitty)
    test1 = bitty[1:4]
    # assert
    print(test1)
    test2 = test1[1:4]
    print(test2)
    test2 = test1[[1, 2, 3]]
    print(test2)
    selection = bitty[1:4][1:4]
    print(selection)
    selection.write(np.full_like(selection, selection.get_max_item()))
    print(selection)
    print(bitty)

    # bitty = Bitty(test, max_bit=6, bit_slice_first=True)
    # arranged = bitty[2:5]
    # print(arranged)
    # selection = bitty[3:]
    # bitty[3:] = np.full_like(selection, selection.get_max_item())
    # print(selection)
    #
    # print(bitty)

    # test3 = bitty[1]
    # print(test3)

    # split = -2

    # bitty = Bitty(test, max_bit=6, bit_slice_first=True)
    # test1 = bitty[:split]
    # test1.read()
    # print(test1)
    # test2 = bitty[split:]
    # print(test2)
    # new_bitty = Bitty.stack_bit(test1, test2)
    # print(new_bitty)
    #
    # bitty = Bitty(test, max_bit=6, bit_slice_first=True)
    # group_indices = [np.nonzero(bitty[split] == b)[0] for b in range(2)]
    # print(group_indices)
    # grop1 = bitty[:][group_indices[0]]
    # print(grop1)
    # new_bt1 = Bitty.stack_bit(grop1[:split], grop1[split + 1:])
    # print(new_bt1)
    # grop2 = bitty[:][group_indices[1]]
    # print(grop2)
    # new_bt2 = Bitty.stack_bit(grop2[:split], grop2[split + 1:])
    # print(new_bt2)