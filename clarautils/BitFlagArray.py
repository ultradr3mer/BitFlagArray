from abc import abstractmethod, abstractproperty, ABC
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Dict, Union
from typing import NamedTuple

import weakref

import numpy as np
import numpy.typing as npt

from commonTyping import get_type_for_bit_count


# ------------------------------------------------------------------ #
#  Read-once / cache configuration
# ------------------------------------------------------------------ #
MAX_READS = 10
CACHE_KB = 1024
_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_MAX_ENTRIES = 16


def cache_key(root_array: np.ndarray, item_slice, bit_slice) -> tuple:
    def sk(s):
        if isinstance(s, slice):
            return ("s", s.start, s.stop)
        if isinstance(s, list):
            return ("l", tuple((x.start, x.stop) for x in s))
        return ("a", s.tobytes())

    return (id(root_array), sk(item_slice), sk(bit_slice))


def cache_get(key):
    if CACHE_KB <= 0:
        return None
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ref, array = entry
    if ref() is None:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return array


def cache_put(key, root_array: np.ndarray, array: np.ndarray):
    if CACHE_KB <= 0:
        return
    _CACHE[key] = (weakref.ref(root_array), array)
    _CACHE.move_to_end(key)
    max_bytes = CACHE_KB * 1024
    while (len(_CACHE) > _CACHE_MAX_ENTRIES
           or sum(a.nbytes for _, a in _CACHE.values()) > max_bytes):
        _CACHE.popitem(last=False)


def cache_invalidate_root(root_id: int):
    keys = [k for k in _CACHE if k[0] == root_id]
    for k in keys:
        _CACHE.pop(k, None)


def clear_cache():
    _CACHE.clear()


# ------------------------------------------------------------------ #
#  Slice types & helpers
# ------------------------------------------------------------------ #
sliceTypes = Union[slice, npt.NDArray[np.unsignedinteger], List[slice]]


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


def slices_to_indices(s) -> np.ndarray:
    if isinstance(s, slice):
        return np.arange(s.start, s.stop, dtype=np.intp)
    if isinstance(s, list):
        return np.concatenate([np.arange(sl.start, sl.stop, dtype=np.intp) for sl in s])
    return np.asarray(s, dtype=np.intp)


def indices_to_slices(indices) -> sliceTypes:
    indices = np.asarray(indices)
    if indices.size == 0:
        return slice(0, 0, 1)
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    if len(breaks) == 0:
        return slice(int(indices[0]), int(indices[-1]) + 1, 1)
    return indices


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
        return slice(normalize_idx(key), normalize_idx(key) + 1, 1)

    if isinstance(key, list) and all(isinstance(x, slice) for x in key):
        return key

    if isinstance(key, np.ndarray):
        if key.dtype == bool:
            if key.size != dim_max:
                raise NotImplementedError("Slice not supported.")
            indices = np.nonzero(key)[0]
        else:
            indices = np.where(key < 0, dim_max + key, key)
        return indices_to_slices(indices)

    indices = np.array([normalize_idx(i) for i in key], dtype=np.intp)
    return indices_to_slices(indices)


def get_slice_item_count(s: sliceTypes):
    if isinstance(s, slice):
        return s.stop - s.start
    if isinstance(s, list):
        return sum(sl.stop - sl.start for sl in s)
    if isinstance(s, np.ndarray):
        return len(s)
    raise NotImplementedError("Slice not supported.")


def put_bits(values: np.ndarray, put_indices, original_bit_count: int):
    result = np.zeros_like(values)

    shifts = (np.full_like(put_indices, fill_value=original_bit_count - 1)
              - np.flip(put_indices, axis=0))

    for shift_from, shift_to in enumerate(shifts):
        result = result + (1 << int(shift_to)) * ((values >> shift_from) & 1)

    return result


def merge_slices(global_slice: sliceTypes, local_slice: sliceTypes) -> sliceTypes:
    if isinstance(global_slice, slice) and isinstance(local_slice, slice):
        return slice(local_slice.start + global_slice.start,
                     local_slice.stop + global_slice.start, 1)

    g_idx = slices_to_indices(global_slice)
    l_idx = slices_to_indices(local_slice)
    return indices_to_slices(g_idx[l_idx])


def limit_to_bit_count(value, bit_count: int | slice, check_overflow=False) -> NBitAryOnly:
    if isinstance(bit_count, slice):
        bit_count = bit_count.stop - bit_count.start
    new_val = value & get_bitmask(bit_count)
    if check_overflow and not np.array_equal(new_val, value):
        raise Exception("Value to large for target bits!")
    return NBitAryOnly(new_val, bit_count)


def revert_bit_slice(bit_slice: sliceTypes, value, original_bit_count: int):
    if isinstance(bit_slice, slice):
        value = limit_to_bit_count(value, bit_slice, True)
        return value.array << (original_bit_count - bit_slice.stop)

    if isinstance(bit_slice, list) and all(isinstance(x, slice) for x in bit_slice):
        return revert_bit_slice_multi(bit_slice, value, original_bit_count)

    if isinstance(bit_slice, np.ndarray):
        sl = indices_to_slices(bit_slice)
        if isinstance(sl, slice):
            return revert_bit_slice(sl, value, original_bit_count)
        return revert_bit_slice_multi(ndarray_to_slice_list(bit_slice), value, original_bit_count)

    if isinstance(bit_slice, Iterable):
        key_arr = np.asarray(list(bit_slice), dtype=np.intp)
        sl = indices_to_slices(key_arr)
        if isinstance(sl, slice):
            return revert_bit_slice(sl, value, original_bit_count)
        return revert_bit_slice_multi(ndarray_to_slice_list(key_arr), value, original_bit_count)

    raise NotImplementedError("Slice not supported.")


def revert_bit_slice_multi(slices: List[slice], value, original_bit_count: int):
    total = sum(s.stop - s.start for s in slices)
    value = value & get_bitmask(total)
    result = np.zeros_like(np.asarray(value))
    pos = 0
    for s in slices:
        run_len = s.stop - s.start
        extracted = (value >> (total - pos - run_len)) & get_bitmask(run_len)
        result = result | (extracted << (original_bit_count - s.stop))
        pos += run_len
    return result


def ndarray_to_slice_list(indices) -> List[slice]:
    indices = np.asarray(indices)
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    runs = np.split(indices, breaks)
    return [slice(int(r[0]), int(r[-1]) + 1, 1) for r in runs]


def select_bits(data: 'NBitAryOnly', key: sliceTypes) -> 'NBitAryOnly':
    if isinstance(key, slice):
        start, stop, step = key.indices(data.get_bit_count())
        if step != 1:
            raise NotImplementedError("Step != 1 wird nicht unterstützt")
        shifted = data.get_array() >> (data.get_bit_count() - stop)
        return limit_to_bit_count(shifted, key)

    if isinstance(key, list) and all(isinstance(x, slice) for x in key):
        return select_bits_multi(data, key)

    if isinstance(key, np.ndarray):
        if not np.all((key >= 0) & (key <= data.get_bit_count())):
            raise ValueError(f"All indices must be between 0 and {data.get_bit_count()}.")
        sl = indices_to_slices(key)
        if isinstance(sl, slice):
            return select_bits(data, sl)
        return select_bits_multi(data, ndarray_to_slice_list(key))

    if isinstance(key, Iterable):
        key_arr = np.asarray(list(key), dtype=np.intp)
        if not np.all((key_arr >= 0) & (key_arr <= data.get_bit_count())):
            raise ValueError(f"All indices must be between 0 and {data.get_bit_count()}.")
        sl = indices_to_slices(key_arr)
        if isinstance(sl, slice):
            return select_bits(data, sl)
        return select_bits_multi(data, ndarray_to_slice_list(key_arr))

    raise NotImplementedError("Index or slice supported.")


def select_bits_multi(data: 'NBitAryOnly', slices: List[slice]) -> 'NBitAryOnly':
    n = sum(s.stop - s.start for s in slices)
    if n == 0:
        return NBitAryOnly(np.zeros_like(data.get_array()), 0)
    src = data.get_array()
    bc = data.get_bit_count()
    result = np.zeros_like(src)
    pos = 0
    for s in slices:
        run_len = s.stop - s.start
        shifted = src >> (bc - s.stop)
        extracted = shifted & get_bitmask(run_len)
        result = result | (extracted << (n - pos - run_len))
        pos += run_len
    return NBitAryOnly(result, n)


def invert_key(key, max_length: int):
    key = normalize_key(key, max_length)
    mask = np.ones(max_length, dtype=bool)
    if isinstance(key, list):
        for s in key:
            mask[s] = False
    elif isinstance(key, slice):
        mask[key] = False
    else:
        mask[key] = False
    return normalize_key(mask, max_length)


def get_indices(key, length: int) -> List[int]:
    if isinstance(key, slice):
        start, stop, step = key.indices(length)
        return list(range(start, stop, step))
    if isinstance(key, list):
        result = []
        for s in key:
            result.extend(range(s.start, s.stop))
        return result
    return list(key)


def index_array(array: np.ndarray, key) -> np.ndarray:
    if isinstance(key, list) and all(isinstance(x, slice) for x in key):
        parts = [array[s] for s in key]
        return parts[0] if len(parts) == 1 else np.concatenate(parts)
    return array[key]


def set_index_array(array: np.ndarray, key, value):
    if isinstance(key, list) and all(isinstance(x, slice) for x in key):
        pos = 0
        for s in key:
            n = s.stop - s.start
            array[s] = value[pos:pos + n]
            pos += n
    else:
        array[key] = value


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

    def get_shape(self) -> Tuple[int, ...]:
        return (self.get_item_count(), self.get_bit_count())

    def get_bitwise(self):
        return get_bits(self.get_array(), self.get_bit_count())

    def get_bitwise_entropy(self) -> npt.NDArray[np.float32]:
        """Per-bit Shannon entropy over all items; forwards to commonEncoding."""
        return get_bitwise_entropy(self)

    def get_bitwise_mean(self, axis=1) -> npt.NDArray[np.float64]:
        """Mean over bits without unpacking; axis=1 per item, axis=0 per bit position."""
        return get_bitwise_mean(self, axis=axis)

    def get_defined_bits(self) -> List[DefinedBit]:
        """Bits that are constant over all items, as (idx, bit) pairs; MSB-first, without unpacking."""
        return get_defined_bits(self)

    @abstractproperty
    def b(self) -> SliceView:
        """View for bitwise selection."""

    @abstractproperty
    def i(self) -> SliceView:
        """View for bitwise selection."""

    @abstractmethod
    def rm_b(self, key) -> SliceView:
        """Perform bitwise remove."""

    @abstractmethod
    def rm_i(self, key) -> SliceView:
        """Perform itemwise remove."""

    def split_b(self, key) -> Tuple[SliceView, SliceView]:
        return self.b[key], self.rm_b(key)

    def split_i(self, key) -> Tuple[SliceView, SliceView]:
        return self.i[key], self.rm_i(key)


@dataclass
class NBitAryOnly(NBitArray):
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

    def rm_b(self, key) -> SliceView:
        return SliceView(self).rm_b(key)

    def rm_i(self, key) -> SliceView:
        return SliceView(self).rm_i(key)

    @classmethod
    def create_from(cls, param: CommonNBitAry):
        return NBitAryOnly(param.array, param.bit_count)


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

    def group_by_bit(self, key):
        return SliceView(self).group_by_bit(key)

    def rm_b(self, key) -> SliceView:
        return SliceView(self).rm_b(key)

    def rm_i(self, key) -> SliceView:
        return SliceView(self).rm_i(key)

    @staticmethod
    def stack_bit_arys(*arrays: NBitArray):
        all_item_counts = np.array([a.get_item_count() for a in arrays])
        item_count = all_item_counts[0]
        if (item_count != all_item_counts[1:]).any():
            raise ValueError(f"Array muss die Size des ersten ({item_count}) haben.")
        bit_count = np.sum([a.get_bit_count() for a in arrays], dtype=int)

        fitting_type = get_type_for_bit_count(bit_count)
        array = np.zeros(shape=item_count, dtype=fitting_type)

        bitshift = 0
        for item in reversed(arrays):
            shifted = np.array(item.get_array(), dtype=fitting_type) << bitshift
            array = array | shifted
            bitshift += item.get_bit_count()

        return BitFlagArray(array, bit_count)

    @classmethod
    def stack_n_bits(cls, *arrays: np.ndarray, bit_count: int | List[int] | None = None):
        if bit_count is None:
            ary = [NBitAryOnly(a, np.iinfo(a).bits) for a in arrays]
        elif isinstance(bit_count, List):
            ary = [NBitAryOnly(a, count) for a, count in zip(arrays, bit_count)]
        elif isinstance(bit_count, int):
            ary = [NBitAryOnly(a, bit_count) for a in arrays]
        else:
            raise TypeError("Unsupported type for bit_count")

        return cls.stack_bit_arys(*ary)

    @classmethod
    def stack_bit(cls, ary: np.ndarray, axis=1):
        ary = NBitAryOnly.create_from(get_number(ary, axis))
        return BitFlagArray(ary)

    # @classmethod
    # def stack_items(cls, *arrays: NBitArray):
    #     def construct_slice(col_bit: int):
    #         slice_cuts = np.zeros(col_bit.size + 1)
    #         np.cumsum(col_bit, axis=0, out=slice_cuts[1:])
    #         return [slice(min, max) for min, max in zip(slice_cuts, slice_cuts[1:])]
    #
    #     all_shapes = np.array([a.get_shape() for a in arrays])
    #     all_bits_counts = all_shapes[:, 1]
    #     all_item_counts = all_shapes[:, 0]
    #     bit_per_row = sum(all_bits_counts)
    #     item_count = np.unique(all_item_counts)
    #     if item_count.size > 1:
    #         raise ValueError(f"Array muss die Item-Count des ersten ({bit_count}) haben.")
    #
    #     bitty = cls.empty((item_count[0], bit_per_row))
    #     place_item_view = SliceView(bitty)
    #     bit_indices = bitty.get_defined_bits()
    #     for i in range(len(arrays)):
    #         write_slice = slice(0, single_ary.get_bit_count())
    #         place_item_view[write_slice].write(single_ary)
    #         place_item_view.rm_b(write_slice)
    #
    #     return BitFlagArray(array, bit_count)

    @staticmethod
    def empty(shape: Tuple[int, int]):
        ary = np.zeros(shape=shape[0], dtype=get_type_for_bit_count(shape[1]))
        return BitFlagArray(ary, shape[1])


Bitty = BitFlagArray


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
        self._read_count = 0
        self._write_mask = None

    def get_bit_count(self) -> int:
        return get_slice_item_count(self.bit_slice)

    def get_root_data(self) -> NBitArray:
        data = self.data
        while isinstance(data, SliceView):
            data = data.data
        return data

    def get_bit_indices(self):
        return get_indices(self.bit_slice, self.get_root_data().get_bit_count())

    def get_item_indices(self):
        return get_indices(self.item_slice, len(self.get_root_data()))

    def get_item_count(self) -> int:
        return get_slice_item_count(self.item_slice)

    def get_next_view(self, key):
        slice_max = (self.get_bit_count() if self.is_bit_slice
                     else self.get_item_count())
        key = normalize_key(key, slice_max)
        return SliceView(self.data,
                         not self.is_bit_slice,
                         merge_slices(self.bit_slice, key) if self.is_bit_slice else self.bit_slice,
                         merge_slices(self.item_slice, key) if not self.is_bit_slice else self.item_slice)

    @property
    def b(self):
        return self.copy(is_bit_slice=True)

    @property
    def i(self):
        return self.copy(is_bit_slice=False)

    def rm_b(self, key):
        key = invert_key(key, self.get_bit_count())
        return self.copy(bit_slice=merge_slices(self.bit_slice, key))

    def rm_i(self, key):
        key = invert_key(key, self.get_item_count())
        return self.copy(item_slice=merge_slices(self.item_slice, key))

    def group_by_bit(self, key) -> Dict[int, "SliceView"]:
        slice_max = self.get_bit_count()
        key = normalize_key(key, slice_max)
        col_view = self.b[key]
        col = col_view.get_array()
        all_key_vals = np.unique(col)

        def select_data(key_val):
            mask = col == key_val
            return self.i[mask].rm_b(key)

        return {
            key_val: select_data(key_val)
            for key_val in all_key_vals
        }

    def __getitem__(self, key) -> SliceView:
        view = self.get_next_view(key)
        return view

    def __setitem__(self, key, value):
        view = self.get_next_view(key)
        view.write(value)

    def get_array(self) -> np.ndarray:
        return self.read().get_array()

    def read(self):
        self._read_count += 1
        if self._read_count > MAX_READS:
            raise RuntimeError(
                f"View read {self._read_count} times (limit {MAX_READS}). "
                f"Call .copy() to materialize and persist."
            )
        root_arr = self.data.get_array()
        ckey = cache_key(root_arr, self.item_slice, self.bit_slice)
        cached = cache_get(ckey)
        if cached is not None:
            return NBitAryOnly(cached, self.get_bit_count())

        items = index_array(root_arr, self.item_slice)
        result = select_bits(NBitAryOnly(items, self.data.get_bit_count()), self.bit_slice)
        cache_put(ckey, root_arr, result.get_array())
        return result

    def write(self, value):
        root_bc = self.data.get_bit_count()
        reverted = revert_bit_slice(self.bit_slice, value, root_bc)

        if self._write_mask is None:
            slice_mask = revert_bit_slice(self.bit_slice, self.get_max_item(), root_bc)
            full_mask = get_bitmask(root_bc)
            self._write_mask = (full_mask - slice_mask, full_mask)

        inv_slice, full_mask = self._write_mask
        root_arr = self.data.get_array()
        current = index_array(root_arr, self.item_slice)
        new_items = current & inv_slice | reverted
        set_index_array(root_arr, self.item_slice, new_items)
        cache_invalidate_root(id(root_arr))

    def get_max_item(self):
        return get_bitmask(self.get_bit_count())

    def materialize(self) -> "BitFlagArray":
        """Read the view once and return a root BitFlagArray (no view overhead)."""
        return BitFlagArray(self.read().get_array(), max_bit=self.get_bit_count())

    def copy(
            self,
            *,
            data: NBitArray | None = None,
            is_bit_slice: bool | None = None,
            bit_slice: sliceTypes | None = None,
            item_slice: sliceTypes | None = None,
    ) -> "SliceView":
        return SliceView(
            data=self.data if data is None else data,
            is_bit_slice=(
                self.is_bit_slice
                if is_bit_slice is None
                else is_bit_slice
            ),
            bit_slice=self.bit_slice if bit_slice is None else bit_slice,
            item_slice=self.item_slice if item_slice is None else item_slice,
        )


# ------------------------------------------------------------------ #
#  Indexer for espeacialiy Ranked Bit Combinations
# ------------------------------------------------------------------ #
class IndexKey(NamedTuple):
    index_slice: slice
    key_value: int

class BitFlagIndex:
    def __init__(self, index_slice: slice, key_value: int):
        self.index = {}

    @staticmethod
    def build_index(data: SliceView, bit_used_by_cols: np.ndarray) -> None:
        next_bit = slice(0, bit_used_by_cols[0])
        index = {}
        for key, group in data.group_by_bit(next_bit).items():
            ms = Multislice(group.i)
            s = ms.get_slices()
            if len(s) > s:
                raise Exception("More than one slice is not supported")
            index[IndexKey(s, key)] = "Dummy"
        pass

BittyIndex = BitFlagIndex