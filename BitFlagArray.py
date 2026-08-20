from abc import abstractmethod, abstractproperty, ABC
from dataclasses import dataclass
from typing import List, Iterable, Tuple

import numpy as np
import numpy.typing as npt

from common import get_type_for_bit_count, get_type_for_scalar
from commonEncoding import get_bitmask, get_number, get_bits


def slice_union(s1: slice, s2: slice) -> slice:
    start = min(s1.start, s2.start)
    stop = max(s1.stop, s2.stop)
    return slice(start, stop)


def slice_intersection(s1: slice, s2: slice) -> slice:
    start = max(s1.start, s2.start)
    stop = min(s1.stop, s2.stop)
    if start >= stop:
        return slice(0, 0)
    return slice(start, stop)

sliceTypes = slice | npt.NDArray[np.unsignedinteger]

def normalize_key(key, dim_max) -> sliceTypes:
    def normalize_idx(i):
        return dim_max + i if i < 0 else i

    if isinstance(key, slice):
        start, stop, step = key.indices(dim_max)
        if step != 1:
            raise NotImplementedError("Step != 1 wird nicht unterstützt")
        start = 0 if start is None else start
        stop = dim_max if stop is None else stop
        return slice(start, stop, 1)

    if isinstance(key, int):
        return np.array([normalize_idx(key)], dtype=get_type_for_scalar(dim_max))

    if isinstance(key, np.ndarray):
        if key.dtype == bool:
            if key.size != dim_max:
                raise NotImplementedError("Slice not supported.")
            return np.array(np.nonzero(key)[0], dtype=get_type_for_scalar(dim_max))
        else:
            np.where(key < 0, dim_max + key, key)

    return np.array([normalize_idx(i) for i in key], dtype=get_type_for_scalar(dim_max))


def get_slice_item_count(s: sliceTypes):
    if isinstance(s, slice):
        return s.stop - s.start
    elif isinstance(s, np.ndarray):
        return len(s)
    else:
        raise NotImplementedError("Slice not supported.")


def arrange_bits(values: NBitArray, take_indices):
    result = np.zeros_like(values)

    shifts = (np.full_like(take_indices, fill_value=values.get_bit_count() - 1)
              - np.flip(take_indices, axis=0))
    from_ary = values.get_array()

    for shift_to, shift_from in enumerate(shifts):
        result = result + (1 << shift_to) * ((from_ary >> shift_from) & 1)

    return result


def put_bits(values: np.ndarray, put_indices, original_bit_count: int):
    result = np.zeros_like(values)

    shifts = (np.full_like(put_indices, fill_value=original_bit_count - 1)
              - np.flip(put_indices, axis=0))

    for shift_from, shift_to in enumerate(shifts):
        result = result + (1 << shift_to) * ((values >> shift_from) & 1)

    return result


def merge_slices(global_slice: sliceTypes, local_slice: sliceTypes) -> sliceTypes:
    if isinstance(global_slice, slice):
        if isinstance(local_slice, slice):
            return slice(local_slice.start + global_slice.start, local_slice.stop + global_slice.start, 1)
        elif isinstance(local_slice, np.ndarray):
            return local_slice + global_slice.start
    elif isinstance(global_slice, np.ndarray):
        if isinstance(local_slice, slice):
            return global_slice[local_slice]
        elif isinstance(local_slice, np.ndarray):
            return global_slice[local_slice]

    raise NotImplementedError("Slice not supported.")


def limit_to_bit_count(value: np.ndarray, bit_count: int | slice, check_overflow=False) -> NBitAryTpl:
    if isinstance(bit_count, slice):
        bit_count = bit_count.stop - bit_count.start
    new_val = value & get_bitmask(bit_count)
    if check_overflow and not (new_val == value).all():
        raise Exception("Value to large for target bits!")
    return NBitAryTpl(new_val,bit_count)


def revert_bit_slice(bit_slice: sliceTypes, value: npt.NDArray[np.unsignedinteger], original_bit_count: int) -> \
    npt.NDArray[np.unsignedinteger]:
    if isinstance(bit_slice, slice):
        value = limit_to_bit_count(value, bit_slice, True)
        return value.array << original_bit_count - bit_slice.stop
    elif isinstance(bit_slice, List):
        value = limit_to_bit_count(value, len(bit_slice), True)
        return put_bits(value.array, bit_slice, original_bit_count)

    raise NotImplementedError("Slice not supported.")


def select_bits(data: NBitAryTpl, key: sliceTypes) -> NBitAryTpl :
    if isinstance(key, Iterable):
        if not all(0 <= x <= data.get_bit_count() for x in key):
            raise ValueError(f"All indices must be between 0 and {data.get_bit_count()}.")
        return NBitAryTpl(arrange_bits(data, key), len(key))

    elif isinstance(key, slice):
        start, stop, step = key.indices(data.get_bit_count())
        if step != 1:
            raise NotImplementedError("Step != 1 wird nicht unterstützt")

        shifted = data.get_array() >> (data.get_bit_count() - stop)

        return limit_to_bit_count(shifted, key)

    else:
        raise NotImplementedError("Index or slice supported.")



class NBitArray(ABC):
    """Interface for arrays of variable item bit count."""

    @abstractmethod
    def get_array(self) -> np.ndarray:
        """Returns the internal numpy array."""

    @abstractmethod
    def get_bit_count(self) -> int:
        """Returns count of the ."""

    def __array__(self, dtype=None, copy=False):
        ary = self.get_array()
        ary = ary.astype(dtype) if dtype else ary
        ary = ary.copy() if copy else ary
        return ary

    def __len__(self):
        """Ermöglicht len(accessor)"""
        return len(self.get_array())

    def __repr__(self):
        return f"{type(self)}, bit_length={self.get_bit_count()},\n{get_bits(self.get_array(), self.get_bit_count())}"

    def __eq__(self, other):
        return np.array(self) == np.array(other)

    def get_item_count(self) -> int:
        return len(self)

    def get_bitwise(self):
        return get_bits(self.get_array(), self.get_bit_count())

    @abstractproperty
    def b(self) -> SliceView:
        """View for bitwise selection."""

    @abstractproperty
    def i(self) -> SliceView:
        """View for bitwise selection."""

@dataclass
class NBitAryTpl(NBitArray):
    array: np.ndarray
    bit_count: int

    def get_array(self) -> np.ndarray:
        return self.array

    def get_bit_count(self) -> int:
        return self.bit_count

    @property
    def b(self) -> SliceView:
        return SliceView(self, is_bit_slice=True)

    @property
    def i(self) -> SliceView:
        return SliceView(self, is_bit_slice=False)


class BitFlagArray(NBitArray):
    def __init__(self, a: NBitArray | np.ndarray, max_bit: int = 0):
        if isinstance(a, NBitArray) or isinstance(a, SliceView):
            self.array = a.get_array()
            self.bit_count = min(a.get_bit_count(), max_bit) if max_bit > 0 else a.get_bit_count()
            # self.item_count = len(self.array)
            return
        if not isinstance(a, np.ndarray) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError("Eingabe muss ein numpy-Integer-Array sein.")
        if a.ndim != 1:
            raise ValueError("Array muss 1-dimensional sein.")
        if not np.issubdtype(a.dtype, np.unsignedinteger):
            if np.min(a) < 0:
                raise ValueError("Array darf keine negativen Werte enthalten.")
            else:
                a = a.astype(get_type_for_bit_count(np.iinfo(a.dtype).bits // 2))
        self.array = a
        type_max_bits = np.iinfo(a.dtype).bits
        self.bit_count = min(type_max_bits, max_bit) if max_bit > 0 else type_max_bits

    def get_bit_count(self) -> int:
        return self.bit_count

    def get_array(self) -> np.ndarray:
        return self.array

    @property
    def b(self):
        return SliceView(self, is_bit_slice=True)

    @property
    def i(self):
        return SliceView(self, is_bit_slice=False)

    @staticmethod
    def stack_bit_arys(*arrays: NBitArray):
        all_item_counts = np.array([a.get_item_count() for a in arrays])
        item_count = all_item_counts[0]
        if (item_count != all_item_counts[1:]).any():
            raise ValueError(f"Array muss die Size des ersten ({item_count}) haben.")
        bit_count = np.sum([a.get_bit_count() for a in arrays], dtype=int)

        array = np.zeros(shape=item_count, dtype=get_type_for_bit_count(bit_count))

        bitshift = 0
        for item in reversed(arrays):
            shifted = item.get_array() << bitshift
            array = array | shifted
            bitshift += item.get_bit_count()

        return BitFlagArray(array, bit_count)

    @classmethod
    def stack_n_bits(cls, *arrays: np.ndarray, bit_count: int | List[int] | None = None):
        if bit_count is None:
            ary = [NBitAryTpl(a, np.iinfo(a).bits) for a in arrays]
        elif isinstance(bit_count, List):
            ary = [NBitAryTpl(a, count) for a, count in zip(arrays, bit_count)]
        elif isinstance(bit_count, int):
            ary = [NBitAryTpl(a, bit_count) for a in arrays]
        else:
            raise TypeError("Unsupported type for bit_count")

        return cls.stack_bit_arys(*ary)

    @classmethod
    def stack_bit(cls, ary: np.ndarray, axis=1):
        tpl: Tuple[np.ndarray,int] = get_number(ary, axis)
        return BitFlagArray(NBitAryTpl(*tpl))


    @staticmethod
    def stack_items(*arrays: NBitArray):
        all_bit_counts = np.array([a.get_bit_count() for a in arrays])
        bit_count = all_bit_counts[0]
        if (bit_count != all_bit_counts[1:]).any():
            raise ValueError(f"Array muss die Bit-Count des ersten ({bit_count}) haben.")

        array = np.concatenate([a.get_array() for a in arrays])

        return BitFlagArray(array, bit_count)

    @staticmethod
    def empty(shape: Tuple[int, int]):
        ary = np.zeros(shape=shape[0], dtype=get_type_for_bit_count(shape[1]))
        return BitFlagArray(ary, shape[1])


class SliceView(NBitArray):
    def __init__(self,
                 data: NBitArray,
                 is_bit_slice: bool = True,
                 bit_slice: sliceTypes = slice(None),
                 item_slice: sliceTypes = slice(None)):
        self.item_slice = normalize_key(item_slice, len(data))
        self.bit_slice = normalize_key(bit_slice, data.get_bit_count())
        self.is_bit_slice = is_bit_slice
        self.data = data

    def get_bit_count(self) -> int:
        return get_slice_item_count(self.bit_slice)

    def get_next_view(self, key):
        slice_max = self.get_bit_count() if self.is_bit_slice else len(self.get_array())
        key = normalize_key(key, slice_max)
        return SliceView(self.data,
                         not self.is_bit_slice,
                         merge_slices(self.bit_slice, key) if self.is_bit_slice else self.bit_slice,
                         merge_slices(self.item_slice, key) if not self.is_bit_slice else self.item_slice)

    @property
    def b(self):
        return SliceView(self.data, True, self.bit_slice, self.item_slice)

    @property
    def i(self):
        return SliceView(self.data, False, self.bit_slice, self.item_slice)

    def __getitem__(self, key) -> SliceView:
        view = self.get_next_view(key)
        return view

    def __setitem__(self, key, value):
        view = self.get_next_view(key)
        view.write(value)

    def get_array(self) -> np.ndarray:
        return self.read().get_array()

    def read(self):
        items = self.data.get_array()[self.item_slice]
        return select_bits(NBitAryTpl(items, self.data.get_bit_count()), self.bit_slice)

    def write(self, value):
        reverted = revert_bit_slice(self.bit_slice, value, self.data.get_bit_count())
        slice_mask = revert_bit_slice(self.bit_slice, self.get_max_item(), self.data.get_bit_count())

        full_mask = get_bitmask(self.data.get_bit_count())
        inv_slice = full_mask - slice_mask  # get_slice_bitmask(self.bit_slice)
        new_items = self.data.get_array()[self.item_slice] & inv_slice | reverted
        self.data.get_array()[self.item_slice] = new_items

    def get_max_item(self):
        return get_bitmask(self.get_bit_count())


Bitty = BitFlagArray
