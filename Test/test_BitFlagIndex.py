from itertools import combinations

import numpy as np
import pytest

from clarautils.BitFlagArray import (
    Bitty,
    BitFlagIndex,
    BittyIndex,
    FluentBuilder,
    NBitAryOnly,
    IndexingNode,
    LeafNode,
    _CACHE,
)


@pytest.fixture
def bitty(bit_data):
    return Bitty.stack_bit(bit_data)


def test_builder_methods_return_self(bitty):
    builder = BitFlagIndex(bitty, dispose_bitty=False).index_by(2)
    assert isinstance(builder, FluentBuilder)
    assert builder.then_by(1) is builder
    assert builder.with_leafs(0) is builder
    assert builder.index_by_key() is builder
    assert builder.index_by_slice() is builder
    assert builder.index_by_fullindex() is builder
    assert builder.build() is not None


def test_default_options_index_key_only(bitty):
    idx = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).build()
    root = idx.root_node
    assert isinstance(root, IndexingNode)
    assert set(root.key_index) == {0, 1}
    assert root.slice_index == {}
    assert root.int_index == {}


def test_flat_index_matches_group_by_bit(bitty):
    groups = bitty.group_by_bit(2)
    idx = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).build()
    leaf0 = idx.get(0)
    leaf1 = idx.get(1)
    assert isinstance(leaf0, LeafNode)
    assert leaf0.item_indices == [1, 2, 5]
    assert leaf1.item_indices == [0, 3, 4]
    assert leaf0.key_path == (0,)
    assert leaf1.key_path == (1,)
    assert leaf0.data is not bitty.get_array()
    np.testing.assert_array_equal(leaf0.data, groups[0].get_array())
    np.testing.assert_array_equal(leaf1.data, groups[1].get_array())
    assert idx.root_node.key_index[0] is leaf0
    assert idx.root_node.key_index[1] is leaf1


def test_multi_level_then_by(bitty):
    groups = bitty.group_by_bit(2)
    sub0 = groups[0].group_by_bit(1)
    sub1 = groups[1].group_by_bit(1)
    idx = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).then_by(1).build()
    assert set(idx.root_node.key_index) == {0, 1}
    assert isinstance(idx.get((0,)), IndexingNode)
    leaf01 = idx.get((0, 1))
    leaf10 = idx.get((1, 0))
    leaf11 = idx.get((1, 1))
    assert leaf01.item_indices == [1, 2, 5]
    assert leaf10.item_indices == [0]
    assert leaf11.item_indices == [3, 4]
    assert leaf01.key_path == (0, 1)
    np.testing.assert_array_equal(leaf01.data, sub0[1].get_array())
    np.testing.assert_array_equal(leaf10.data, sub1[0].get_array())
    np.testing.assert_array_equal(leaf11.data, sub1[1].get_array())


def test_with_leafs_sets_leaf_level(bitty):
    idx = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).with_leafs(1).build()
    assert idx.structure_root_key == 2
    assert idx.structure_sub_keys == []
    assert idx.structure_leafs == 1
    assert idx.get((0, 1)).item_indices == [1, 2, 5]
    assert idx.get((1, 0)).item_indices == [0]
    assert idx.get((1, 1)).item_indices == [3, 4]


def test_index_by_slice_and_fullindex(bitty):
    idx = (BitFlagIndex(bitty, dispose_bitty=False).index_by(2)
           .index_by_slice().index_by_fullindex().build())
    root = idx.root_node
    assert root.key_index == {}
    leaf0 = root.int_index[1]
    leaf1 = root.int_index[0]
    assert set(root.slice_index) == {
        slice(1, 3, 1), slice(5, 6, 1),
        slice(0, 1, 1), slice(3, 5, 1),
    }
    assert root.slice_index[slice(1, 3, 1)] is leaf0
    assert root.slice_index[slice(5, 6, 1)] is leaf0
    assert root.slice_index[slice(0, 1, 1)] is leaf1
    assert root.slice_index[slice(3, 5, 1)] is leaf1
    assert root.int_index == {
        1: leaf0, 2: leaf0, 5: leaf0,
        0: leaf1, 3: leaf1, 4: leaf1,
    }


def test_explicitly_disabled_options_build_empty_maps(bitty):
    idx = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).index_by_key(do=False).build()
    assert idx.root_node.key_index == {}
    assert idx.root_node.slice_index == {}
    assert idx.root_node.int_index == {}


def test_dispose_bitty_drops_bitty_and_cache(bitty):
    expected0 = bitty.group_by_bit(2)[0].get_array().copy()
    root_arr = bitty.get_array()
    arr_id = id(root_arr)
    idx = BitFlagIndex(bitty, dispose_bitty=True).index_by(2).build()
    assert idx.bty is None
    np.testing.assert_array_equal(idx.get(0).data, expected0)
    assert all(key[0] != arr_id for key in _CACHE)


def test_no_dispose_keeps_bitty(bitty):
    idx = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).build()
    assert idx.bty is bitty


def test_build_twice_raises(bitty):
    idx = BitFlagIndex(bitty, dispose_bitty=False)
    idx.index_by(2).build()
    with pytest.raises(RuntimeError):
        idx.index_by(3).build()


def test_get_before_build_raises(bitty):
    idx = BitFlagIndex(bitty, dispose_bitty=False)
    with pytest.raises(RuntimeError):
        idx.get(0)


def test_list_int_key_is_single_level(bitty):
    groups = bitty.group_by_bit([2, 3])
    idx = BitFlagIndex(bitty, dispose_bitty=False).index_by([2, 3]).build()
    assert set(idx.root_node.key_index) == set(groups) == {1, 2, 3}
    for key_val, ref in groups.items():
        leaf = idx.get(key_val)
        assert leaf.item_indices == ref.get_item_indices()
        np.testing.assert_array_equal(leaf.data, ref.get_array())


def test_then_by_varargs_one_level_per_key(bitty):
    chained = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).then_by(1).then_by(0).build()
    varargs = BitFlagIndex(bitty, dispose_bitty=False).index_by(2).then_by(1, 0).build()
    for key_path in ((1, 0, 1), (1, 1, 0), (0, 1, 0)):
        assert chained.get(key_path).key_path == varargs.get(key_path).key_path
        assert chained.get(key_path).item_indices == varargs.get(key_path).item_indices


def test_ranking_roundtrip_without_bitty():
    combs = list(combinations(range(4), 2))
    tbl = np.array(combs) - np.arange(2)
    cols = [NBitAryOnly(tbl[:, i].copy(), 2) for i in range(2)]
    bty = Bitty.stack_bit_arys(*cols)

    idx = (BitFlagIndex(bty, dispose_bitty=True)
           .index_by(slice(0, 2))
           .with_leafs(slice(0, 2))
           .index_by_key()
           .index_by_fullindex()
           .build())

    assert idx.bty is None
    for row, (a, b) in enumerate(tbl):
        leaf = idx.get((int(a), int(b)))
        assert isinstance(leaf, LeafNode)
        assert leaf.item_indices == [row]
        assert leaf.key_path == (int(a), int(b))
        node = idx.root_node
        while isinstance(node, IndexingNode):
            node = node.int_index[row]
        assert node is leaf
        positions = tuple(np.array(leaf.key_path) + np.arange(2))
        assert positions == combs[row]


def test_bitty_index_alias():
    assert BittyIndex is BitFlagIndex
