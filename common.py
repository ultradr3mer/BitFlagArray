from logging import raiseExceptions
from typing import TypeVar

import numpy as np

def get_type_for_scalar(value):
    def raise_exception() -> np.dtype:
        raise Exception("value to big")
    return np.ubyte if value <= np.iinfo(np.uint8).max else \
        np.uint16 if value <= np.iinfo(np.uint16).max else \
            np.uint32 if value <= np.iinfo(np.uint32).max else \
                np.uint64 if value <= np.iinfo(np.uint64).max else \
                    raise_exception()


def get_type_for_array(ary):
    return get_type_for_scalar(np.max(ary))

def get_type_for_bit_count(bit_count):
    def raise_exception() -> np.dtype:
        raise Exception("to many bits requested")
    return np.ubyte if bit_count <= np.iinfo(np.uint8).bits else \
        np.uint16 if  bit_count <= np.iinfo(np.uint16).bits else \
            np.uint32 if bit_count <= np.iinfo(np.uint32).bits else \
                np.uint64 if bit_count <= np.iinfo(np.uint64).bits else \
                    raise_exception()

def iter_bits(data):
    for byte in data:
        for i in range(8):
            yield (byte >> (7 - i)) & 1

len_steps = [np.pow(2, x) + 2*x for x in range(15)]
len_steps = np.array(len_steps, dtype=get_type_for_scalar(max(len_steps)))

def get_len_step(value: np.unsignedinteger, steps: np.array[np.unsignedinteger]) -> tuple[np.unsignedinteger,np.unsignedinteger]:
    last_step = 0
    for i, cur_step in enumerate(steps):
        if last_step < value <= cur_step:
            return i, cur_step
        last_step = cur_step
    return -1, None

def get_max_value(bit_count: int) -> np.unsignedinteger:
    return  np.pow(2, bit_count) - 1

def get_bit_count(value: np.unsignedinteger):
    return int(np.ceil(np.log2(value+1)))

# def get_bitmax(source: source, start: np.unsignedinteger = 0, to: np.unsignedinteger = -1, size: unsignedinteger = 32) -> int:
#     bytes = np.full(fill_value=source, shape=size, dtype=np.ubyte)
#     bytes[to : size] = 0
#     bytes[0 : start] = 0
#     return np.array([bytes[i] for i in range(size)])

