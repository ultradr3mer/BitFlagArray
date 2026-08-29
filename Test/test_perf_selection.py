import numpy as np
import pytest

pytest.importorskip("pytest_benchmark", reason="pip install pytest-benchmark")

import clarautils.BitFlagArray as _bfa
from clarautils.BitFlagArray import Bitty, clear_cache

_bfa.MAX_READS = 10_000_000


def make_bitty(item_count: int, bit_count: int, seed: int = 0) -> Bitty:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(item_count, bit_count), dtype=np.uint8)
    return Bitty.stack_bit(bits, axis=1)


# ------------------------------------------------------------------ #
#  Iterative chained selection (the core benchmark)
#  Simulates: b[1:4].i[2:5].b[0:3].i[1:4]...
#  Measures cumulative view-construction + materialization cost.
# ------------------------------------------------------------------ #

CHAIN_DEPTHS = [1, 3, 5, 10, 20]
CHAIN_SIZES = [pytest.param(1_000, 8, id="1kx8"),
               pytest.param(10_000, 16, id="10kx16"),
               pytest.param(10_000, 32, id="10kx32")]


def build_chain(bitty, depth, bit_count, item_count):
    view = bitty
    for d in range(depth):
        if d % 2 == 0:
            lo = (d * 1) % max(1, bit_count - 2)
            hi = min(lo + max(1, bit_count // 3), bit_count)
            if hi <= lo:
                hi = lo + 1
            view = view.b[lo:hi]
        else:
            lo = (d * 1) % max(1, item_count - 2)
            hi = min(lo + max(1, item_count // 3), item_count)
            if hi <= lo:
                hi = lo + 1
            view = view.i[lo:hi]
    return view


@pytest.mark.parametrize("item_count,bit_count", CHAIN_SIZES)
@pytest.mark.parametrize("depth", CHAIN_DEPTHS)
def test_perf_chained_selection(benchmark, item_count, bit_count, depth):
    bitty = make_bitty(item_count, bit_count, seed=42)

    def do_chain():
        clear_cache()
        view = build_chain(bitty, depth, bit_count, item_count)
        return view.get_array()

    result = benchmark(do_chain)
    assert result is not None
    benchmark.group = f"chain_d{depth}"


# ------------------------------------------------------------------ #
#  Chained selection without final materialization (view construction only)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", CHAIN_SIZES)
@pytest.mark.parametrize("depth", CHAIN_DEPTHS)
def test_perf_chained_view_construction(benchmark, item_count, bit_count, depth):
    bitty = make_bitty(item_count, bit_count, seed=42)

    def do_construct():
        return build_chain(bitty, depth, bit_count, item_count)

    view = benchmark(do_construct)
    assert view is not None
    benchmark.group = f"view_construct_d{depth}"


# ------------------------------------------------------------------ #
#  Sparse vs dense index lists (measures slice-merge effectiveness)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", CHAIN_SIZES)
def test_perf_dense_index_list(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=43)
    key = list(range(0, bit_count // 2))

    def do_select():
        clear_cache()
        return bitty.b[key].get_array()

    result = benchmark(do_select)
    assert result is not None
    benchmark.group = "dense_index"


@pytest.mark.parametrize("item_count,bit_count", CHAIN_SIZES)
def test_perf_sparse_index_list(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=44)
    key = list(range(0, bit_count, 2))

    def do_select():
        clear_cache()
        return bitty.b[key].get_array()

    result = benchmark(do_select)
    assert result is not None
    benchmark.group = "sparse_index"


# ------------------------------------------------------------------ #
#  Multi-range index list [0,1,2, 5,6, 9]
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count",
                         [pytest.param(1_000, 16, id="1kx16"),
                          pytest.param(10_000, 32, id="10kx32")])
def test_perf_multi_range_index(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=45)
    key = list(range(0, bit_count // 3)) + list(range(bit_count // 2, bit_count * 2 // 3))

    def do_select():
        clear_cache()
        return bitty.b[key].get_array()

    result = benchmark(do_select)
    assert result is not None
    benchmark.group = "multi_range"


# ------------------------------------------------------------------ #
#  Repeated reads (measures cache effectiveness)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("item_count,bit_count", CHAIN_SIZES)
def test_perf_repeated_read_cached(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=46)
    view = bitty.b[1 : bit_count // 2]
    view.get_array()

    def do_cached_read():
        view._read_count = 0
        return view.get_array()

    result = benchmark(do_cached_read)
    assert result is not None
    benchmark.group = "cached_read"


@pytest.mark.parametrize("item_count,bit_count", CHAIN_SIZES)
def test_perf_materialize_then_read(benchmark, item_count, bit_count):
    bitty = make_bitty(item_count, bit_count, seed=47)
    view = bitty.b[1 : bit_count // 2]

    def do_materialize():
        clear_cache()
        return view.materialize()

    result = benchmark(do_materialize)
    assert result is not None
    benchmark.group = "materialize"
