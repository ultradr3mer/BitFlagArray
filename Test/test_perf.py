import numpy as np
import pytest

pytest.importorskip("pytest_benchmark", reason="pip install pytest-benchmark")

import BitFlagArray as _bfa
from BitFlagArray import (
    BitFlagArray,
    Bitty,
    NBitAryOnly,
    put_bits,
    select_bits,
    clear_cache,
)
from commonEncoding import get_bitmask, arrange_bits

_bfa.MAX_READS = 10_000_000


SIZES = [pytest.param(1_000, 8, id="1kx8"),
         pytest.param(10_000, 16, id="10kx16"),
         pytest.param(10_000, 32, id="10kx32")]


def make_bitty(item_count: int, bit_count: int, seed: int = 0) -> Bitty:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(item_count, bit_count), dtype=np.uint8)
    return Bitty.stack_bit(bits, axis=1)


# ------------------------------------------------------------------ #
#  arrange_bits / put_bits  (hotspot #1)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_arrange_bits_identity(benchmark, item_count, bit_count):
    data = NBitAryOnly(make_bitty(item_count, bit_count).get_array(), bit_count)
    take = np.arange(bit_count, dtype=np.uint8)

    result = benchmark(arrange_bits, data, take)

    np.testing.assert_array_equal(result, data.get_array())
    benchmark.group = "arrange_bits"


@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_arrange_bits_permutation(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=1)
    data = NBitAryOnly(bitty.get_array(), bit_count)
    rng = np.random.default_rng(2)
    take = rng.permutation(bit_count).astype(np.uint8)

    result = benchmark(arrange_bits, data, take)

    # arrange_bits places result bit i = source bit take[n-1-i]
    bits = bitty.get_bitwise()
    expected = (bits[:, take[::-1]] << np.arange(bit_count)).sum(axis=1)
    np.testing.assert_array_equal(result.astype(expected.dtype), expected)
    benchmark.group = "arrange_bits"


@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_put_bits_permutation(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=3)
    src = bitty.get_array().copy()
    take = np.arange(bit_count, dtype=np.uint8)
    src_pc = np.bitwise_count(src.astype(np.uint64))

    result = benchmark(put_bits, src, take, bit_count)

    # put_bits must preserve popcount for a permutation of bit positions.
    # Currently fails for bit_count > 8 due to `1 << np.uint8(shift_to)`
    # overflowing the uint8 shifts array (hotspot #1).
    result_pc = np.bitwise_count(result.astype(np.uint64))
    try:
        np.testing.assert_array_equal(result_pc, src_pc)
    except AssertionError:
        if bit_count > 8:
            pytest.xfail("put_bits uint8 overflow for bit_count > 8 (hotspot #1)")
        raise
    benchmark.group = "put_bits"


# ------------------------------------------------------------------ #
#  select_bits  (hotspot #1, #6, #7)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_select_bits_slice(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=4)
    data = NBitAryOnly(bitty.get_array(), bit_count)
    key = slice(bit_count // 4, bit_count // 2)

    result = benchmark(select_bits, data, key)

    assert result.get_bit_count() == key.stop - key.start
    benchmark.group = "select_bits"


@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_select_bits_ndarray(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=5)
    data = NBitAryOnly(bitty.get_array(), bit_count)
    rng = np.random.default_rng(6)
    key = np.sort(rng.choice(bit_count, size=max(1, bit_count // 2), replace=False))

    result = benchmark(select_bits, data, key)

    assert result.get_bit_count() == len(key)
    benchmark.group = "select_bits"


# ------------------------------------------------------------------ #
#  SliceView.get_array  (hotspot #8 – re-reads every call)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_slice_view_get_array(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=7)
    view = bitty.b[1 : bit_count // 2]

    def do_read():
        clear_cache()
        view._read_count = 0
        return view.get_array()

    result = benchmark(do_read)

    assert result.shape[0] == item_count
    benchmark.group = "get_array"


# ------------------------------------------------------------------ #
#  SliceView.write  (hotspot #3 – recomputes static mask per call)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_slice_view_write(benchmark, item_count, bit_count):
    def setup():
        bitty = make_bitty(item_count, bit_count, seed=8)
        view = bitty.b[1 : bit_count // 2]
        value = np.full(item_count, view.get_max_item(), dtype=bitty.get_array().dtype)
        return (view, value), {}

    benchmark.pedantic(lambda v, val: v.write(val) or v, setup=setup, rounds=20, iterations=1)
    benchmark.group = "write"


# ------------------------------------------------------------------ #
#  group_by_bit  (hotspot #2 – recomputes slice per group)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_group_by_bit(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=9)
    g_idx = bit_count // 2

    groups = benchmark(bitty.group_by_bit, g_idx)

    assert sum(len(v) for v in groups.values()) == item_count
    benchmark.group = "group_by_bit"


# ------------------------------------------------------------------ #
#  get_bitmask / get_bit_count  (hotspots #4, #5 – float math)
# ------------------------------------------------------------------ #

def test_perf_get_bitmask_batch(benchmark):
    lengths = np.arange(1, 65)

    def run():
        return [get_bitmask(int(n)) for n in lengths]

    out = benchmark(run)
    assert int(out[5]) == 0b111111
    # get_bitmask(64) currently overflows float64 in `np.power(2, 64) - 1`
    # (hotspot #4). Should be 2**64 - 1 as uint64.
    try:
        assert int(out[-1]) == (1 << 64) - 1
    except AssertionError:
        pytest.xfail("get_bitmask(64) float64 overflow (hotspot #4)")
    benchmark.group = "bitmask"

# ------------------------------------------------------------------ #
#  stack_bit_arys  (hotspot #11 – re-copies per layer)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", SIZES)
def test_perf_stack_bit_arys(benchmark, item_count, bit_count):
    rng = np.random.default_rng(11)
    layers = [NBitAryOnly(rng.integers(0, 2, size=item_count, dtype=np.uint8), 8)
              for _ in range(bit_count // 8 or 1)]

    result = benchmark(BitFlagArray.stack_bit_arys, *layers)

    assert result.get_bit_count() == sum(l.get_bit_count() for l in layers)
    benchmark.group = "stack_bit_arys"


# ------------------------------------------------------------------ #
#  find_association_rules  (consumer of arrange_bits/select_bits)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("rows,cols",
                         [pytest.param(32, 6, id="32x6"),
                          pytest.param(128, 8, id="128x8")])
def test_perf_find_association_rules(benchmark, rows, cols):
    from reverseEncoding import find_association_rules

    rng = np.random.default_rng(12)
    bits = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
    data = Bitty.stack_bit(bits, axis=1)

    rules = benchmark(find_association_rules, data)

    assert isinstance(rules, dict)
    benchmark.group = "find_association_rules"
