import numpy as np
import pytest

from clarautils.BitFlagArray import Bitty
from association import build
from reverseEncoding import IndexedBit, Association


def ib(index: int, value: int) -> IndexedBit:
    return IndexedBit(index, value)


def assoc_set(rules: dict[int, list[Association]]) -> set[tuple[frozenset, IndexedBit]]:
    return {
        (frozenset(r.antecedent), r.consequent)
        for r in (rule for rules_list in rules.values() for rule in rules_list)
    }


def bitty(data: list[list[int]]) -> Bitty:
    return Bitty.stack_bit(np.array(data))


# ------------------------------------------------------------------ #
#  Level 0
# ------------------------------------------------------------------ #

def test_build_level0_fixed_bits():
    data = bitty([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    rules = build(data)
    rs = assoc_set(rules)

    assert (frozenset(), ib(0, 1)) in rs
    assert (frozenset(), ib(2, 1)) in rs


def test_build_level0_suppresses_conditional_supersets():
    data = bitty([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    rules = build(data)

    for rule_list in rules.values():
        for r in rule_list:
            if r.consequent == ib(2, 1):
                assert r.antecedent == frozenset()


def test_build_all_constant():
    data = bitty([
        [1, 0],
        [1, 0],
        [1, 0],
    ])
    rules = build(data)
    rs = assoc_set(rules)

    assert (frozenset(), ib(0, 1)) in rs
    assert (frozenset(), ib(1, 0)) in rs
    assert len(rs) == 2


# ------------------------------------------------------------------ #
#  Conditional rules
# ------------------------------------------------------------------ #

def test_build_or_relation():
    data = bitty([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    rules = build(data)
    rs = assoc_set(rules)

    assert (frozenset({ib(0, 1)}), ib(2, 1)) in rs
    assert (frozenset({ib(1, 1)}), ib(2, 1)) in rs
    assert (frozenset({ib(2, 0)}), ib(0, 0)) in rs
    assert (frozenset({ib(2, 0)}), ib(1, 0)) in rs


def test_build_no_redundant_supersets():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
    ])
    rules = build(data)

    for rule_list in rules.values():
        for r in rule_list:
            if r.consequent == ib(2, 1):
                assert r.antecedent == frozenset({ib(0, 1)})


# ------------------------------------------------------------------ #
#  Noise bits
# ------------------------------------------------------------------ #

def test_build_noise_bit_produces_no_rule():
    # bit 1 is independent noise, bits 0 and 2 are identical (c = a)
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
    ])
    rules = build(data)
    rs = assoc_set(rules)

    # bit 1 should never appear as consequent
    for _, consequent in rs:
        assert consequent.index != 1


def test_build_no_rules_when_all_random():
    data = bitty([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    rules = build(data)
    rs = assoc_set(rules)
    # No deterministic rules (no constant columns either)
    for antecedent, _ in rs:
        assert antecedent  # at least one bit in antecedent


# ------------------------------------------------------------------ #
#  max_level
# ------------------------------------------------------------------ #

def test_build_max_level_limits_antecedent_size():
    # c = a AND b (needs level-2 antecedent {a=1, b=1} => c=1)
    data = bitty([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ])

    # max_level=1: {a=1,b=1} not found
    rules1 = build(data, max_level=1)
    rs1 = assoc_set(rules1)
    assert (frozenset({ib(0, 1), ib(1, 1)}), ib(2, 1)) not in rs1

    # max_level=2: {a=1,b=1} => c=1 found
    rules2 = build(data, max_level=2)
    rs2 = assoc_set(rules2)
    assert (frozenset({ib(0, 1), ib(1, 1)}), ib(2, 1)) in rs2


# ------------------------------------------------------------------ #
#  Cross-check
# ------------------------------------------------------------------ #

def test_build_matches_find_association_rules():
    from reverseEncoding import find_association_rules

    data = bitty([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    mlxtend_rules = assoc_set(build(data))
    own_rules = assoc_set(find_association_rules(data))

    assert mlxtend_rules == own_rules


def test_build_matches_find_association_rules_with_fixed():
    from reverseEncoding import find_association_rules

    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
    ])
    mlxtend_rules = assoc_set(build(data))
    own_rules = assoc_set(find_association_rules(data))

    assert mlxtend_rules == own_rules