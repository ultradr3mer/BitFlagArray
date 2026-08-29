import numpy as np

from clarautils.BitFlagArray import Bitty
from DebugPrint import print_debug
from TextDisplayMeasure import TextDisplayMeasure

test = np.array([[1, 0, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0, 1],
                 [1, 1, 0, 1, 0, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1],
                 [0, 1, 0, 1, 0, 1]])

long_data = np.array([[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
                      [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
                      [1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
                      [1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
                      [0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1],
                      [0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1]])

bitty = Bitty.stack_bit(test)
lbitty = Bitty.stack_bit(long_data)

views = [
    ("b[:]",           bitty.b[:]),
    ("b[1:4]",         bitty.b[1:4]),
    ("b[1:4][1:4]",    bitty.b[1:4][1:4]),
    ("i[1:4]",         bitty.i[1:4]),
    ("b[[0,2,4]]",     bitty.b[[0, 2, 4]]),
    ("i[[1,3,5]]",     bitty.i[[1, 3, 5]]),
    ("b[2]",           bitty.b[2]),
    ("i[2]",           bitty.i[2]),
    ("b[0:0]",         bitty.b[0:0]),
    ("long b[:]",             lbitty.b[:]),
    ("long b[8:]",            lbitty.b[8:]),
    ("long b[[0,5,10,15]]",   lbitty.b[[0, 5, 10, 15]]),
]

tdm = TextDisplayMeasure()

for label, v in views:
    print(f"═══ {label} ═══")
    for mode in ("1/0", "sub", "circ"):
        print(f"─── mode={mode} ───")
        dump = print_debug(v, mode)
        print(dump)
    print()