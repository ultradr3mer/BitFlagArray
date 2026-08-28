import numpy as np
import pytest

from BitFlagArray import Bitty
from reverseEncoding import (
    IndexedBit,
    Association,
    AccIdxBit,
    add_rule,
    applicable_rules,
    find_consequents,
    find_association_rules,
    CombinationGen,
)


def ib(index: int, value: int) -> IndexedBit:
    return IndexedBit(index, value)


def assoc(antecedent, consequent_idx, consequent_val) -> Association:
    return Association(frozenset(antecedent), ib(consequent_idx, consequent_val))


def bitty(data: list[list[int]]) -> Bitty:
    return Bitty.stack_bit(np.array(data))


# ------------------------------------------------------------------ #
#  Association / IndexedBit
# ------------------------------------------------------------------ #

def test_indexed_bit_is_hashable_and_equal():
    assert ib(0, 1) == ib(0, 1)
    assert {ib(0, 1), ib(0, 1)} == {ib(0, 1)}


def test_association_is_hashable():
    a = assoc({ib(0, 1)}, 2, 1)
    b = assoc({ib(0, 1)}, 2, 1)
    assert len({a, b}) == 1


def test_from_multiple_returns_frozenset():
    fs = IndexedBit.from_multiple([0, 1], [1, 0])
    assert isinstance(fs, frozenset)
    assert fs == frozenset({ib(0, 1), ib(1, 0)})


def test_from_multiple_accepts_int_value():
    fs = IndexedBit.from_multiple([0, 1], 0b01)
    assert fs == frozenset({ib(0, 0), ib(1, 1)})


# ------------------------------------------------------------------ #
#  add_rule
# ------------------------------------------------------------------ #

def test_add_rule_new_returns_true():
    rules: dict[int, list[Association]] = {}
    r = assoc({ib(0, 1)}, 2, 1)
    assert add_rule(r, rules) is True
    assert rules[2] == [r]


def test_add_rule_redundant_superset_returns_false():
    rules: dict[int, list[Association]] = {}
    add_rule(assoc({ib(0, 1)}, 2, 1), rules)
    assert add_rule(assoc({ib(0, 1), ib(1, 0)}, 2, 1), rules) is False
    assert len(rules[2]) == 1


def test_add_rule_different_consequent_kept():
    rules: dict[int, list[Association]] = {}
    add_rule(assoc({ib(0, 1)}, 2, 1), rules)
    assert add_rule(assoc({ib(0, 1)}, 2, 0), rules) is True
    assert len(rules[2]) == 2


def test_add_rule_different_target_kept():
    rules: dict[int, list[Association]] = {}
    add_rule(assoc({ib(0, 1)}, 2, 1), rules)
    assert add_rule(assoc({ib(0, 1)}, 3, 1), rules) is True
    assert set(rules.keys()) == {2, 3}


# ------------------------------------------------------------------ #
#  applicable_rules
# ------------------------------------------------------------------ #

def test_applicable_rules_subset_and_target_unknown():
    rules: dict[int, list[Association]] = {}
    add_rule(assoc({ib(0, 1)}, 2, 1), rules)
    add_rule(assoc({ib(1, 0)}, 3, 1), rules)

    known = {ib(0, 1)}
    applicable = applicable_rules(rules, known)
    assert assoc({ib(0, 1)}, 2, 1) in applicable
    assert assoc({ib(1, 0)}, 3, 1) not in applicable


def test_applicable_rules_skips_known_target():
    rules: dict[int, list[Association]] = {}
    add_rule(assoc({ib(0, 1)}, 2, 1), rules)

    known = {ib(0, 1), ib(2, 1)}
    assert applicable_rules(rules, known) == []


def test_applicable_rules_strict_superset_antecedent_not_applicable():
    rules: dict[int, list[Association]] = {}
    add_rule(assoc({ib(0, 1), ib(1, 0)}, 2, 1), rules)

    known = {ib(0, 1)}
    assert applicable_rules(rules, known) == []

    known = {ib(0, 1), ib(1, 0)}
    assert applicable_rules(rules, known) == [assoc({ib(0, 1), ib(1, 0)}, 2, 1)]


# ------------------------------------------------------------------ #
#  AccIdxBit
# ------------------------------------------------------------------ #

def test_acc_idx_bit_sorts_by_index():
    a = AccIdxBit([ib(2, 1), ib(0, 1), ib(1, 0)])
    assert a[IndexedBit.Fields.index] == [0, 1, 2]
    assert a[IndexedBit.Fields.value] == [1, 0, 1]


def test_acc_idx_bit_accepts_frozenset():
    a = AccIdxBit(frozenset({ib(2, 1), ib(0, 1)}))
    assert a[IndexedBit.Fields.index] == [0, 2]


def test_acc_idx_bit_select_matches_rows():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
    ])
    a = AccIdxBit(frozenset({ib(0, 1), ib(2, 1)}))
    matching = a.select(data)
    np.testing.assert_array_equal(
        matching.get_bitwise(),
        np.array([[1, 0, 1], [1, 1, 1]]),
    )


def test_acc_idx_bit_select_no_match():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
    ])
    a = AccIdxBit(frozenset({ib(0, 0)}))
    matching = a.select(data)
    assert len(matching) == 0


def test_acc_idx_bit_select_single_bit():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
    ])
    a = AccIdxBit(frozenset({ib(0, 1)}))
    matching = a.select(data)
    np.testing.assert_array_equal(
        matching.get_bitwise(),
        np.array([[1, 0, 1], [1, 1, 1]]),
    )


# ------------------------------------------------------------------ #
#  CombinationGen
# ------------------------------------------------------------------ #

def test_combination_gen_level_zero():
    data = bitty([[1, 0], [0, 1]])
    gen = CombinationGen(2)
    assert list(gen.gen(0, data)) == [frozenset()]


def test_combination_gen_yields_existing_value_combos():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
    ])
    gen = CombinationGen(3)
    antecedents = list(gen.gen(1, data))
    assert frozenset({ib(0, 1)}) in antecedents
    assert frozenset({ib(1, 0)}) in antecedents
    assert frozenset({ib(1, 1)}) in antecedents
    assert frozenset({ib(2, 1)}) in antecedents
    assert frozenset({ib(0, 0)}) not in antecedents


def test_combination_gen_multi_bit_combo():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
    ])
    gen = CombinationGen(3)
    antecedents = list(gen.gen(2, data))
    assert frozenset({ib(0, 1), ib(1, 0)}) in antecedents
    assert frozenset({ib(0, 1), ib(1, 1)}) in antecedents
    assert frozenset({ib(0, 0), ib(1, 0)}) in antecedents


def test_combination_gen_no_duplicate_indices():
    gen = CombinationGen(4)
    for antecedent in gen.gen(2, bitty([[0, 1, 0, 1]])):
        indices = [bit.index for bit in antecedent]
        assert len(indices) == len(set(indices))


# ------------------------------------------------------------------ #
#  find_consequents
# ------------------------------------------------------------------ #

def test_find_consequents_finds_constant_column():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
    ])
    determined: set[int] = set()
    new = find_consequents(data, frozenset({ib(0, 1)}), determined)
    assert ib(2, 1) in new
    assert ib(1, 0) not in new


def test_find_consequents_respects_determined():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
    ])
    determined = {2}
    new = find_consequents(data, frozenset({ib(0, 1)}), determined)
    assert ib(2, 1) not in new


def test_find_consequents_does_not_exclude_ruled_indices():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
    ])
    determined: set[int] = set()
    # find_consequents only excludes `determined` (level-0); redundancy
    # between conditional rules is add_rule's job.
    new = find_consequents(data, frozenset({ib(0, 1)}), determined)
    assert ib(2, 1) in new


def test_find_consequents_empty_antecedent_finds_fixed_bits():
    data = bitty([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])
    determined: set[int] = set()
    new = find_consequents(data, frozenset(), determined)
    assert ib(0, 1) in new  # a ist immer 1
    assert ib(1, 0) not in new
    assert ib(2, 0) not in new


def test_find_consequents_no_match_yields_nothing():
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
    ])
    determined: set[int] = set()
    new = find_consequents(data, frozenset({ib(0, 0)}), determined)
    assert new == []


# ------------------------------------------------------------------ #
#  determined_indices
# ------------------------------------------------------------------ #

def test_determined_indices_empty():
    from reverseEncoding import determined_indices
    assert determined_indices({}) == set()


def test_determined_indices_only_empty_antecedents():
    from reverseEncoding import determined_indices
    rules: dict[int, list[Association]] = {}
    add_rule(assoc(set(), 0, 1), rules)
    add_rule(assoc(set(), 2, 1), rules)
    assert determined_indices(rules) == {0, 2}


def test_determined_indices_excludes_conditional():
    from reverseEncoding import determined_indices
    rules: dict[int, list[Association]] = {}
    add_rule(assoc(set(), 0, 1), rules)
    add_rule(assoc({ib(0, 1)}, 2, 1), rules)
    assert determined_indices(rules) == {0}


# ------------------------------------------------------------------ #
#  find_association_rules (end-to-end)
# ------------------------------------------------------------------ #

def test_find_association_rules_or_relation():
    # c = a OR b
    data = bitty([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    rules = find_association_rules(data)

    rule_set = {
        (frozenset(r.antecedent), r.consequent)
        for r in (rule for rules_list in rules.values() for rule in rules_list)
    }

    assert (frozenset({ib(0, 1)}), ib(2, 1)) in rule_set  # a=1 => c=1
    assert (frozenset({ib(1, 1)}), ib(2, 1)) in rule_set  # b=1 => c=1
    assert (frozenset({ib(2, 0)}), ib(0, 0)) in rule_set  # c=0 => a=0
    assert (frozenset({ib(2, 0)}), ib(1, 0)) in rule_set  # c=0 => b=0


def test_find_association_rules_level0_as_empty_antecedent():
    # a=1 immer, also c = a OR b = 1 immer => a und c fixiert
    data = bitty([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    rules = find_association_rules(data)

    rule_set = {
        (frozenset(r.antecedent), r.consequent)
        for r in (rule for rules_list in rules.values() for rule in rules_list)
    }

    # Level-0: a=1 und c=1 mit leerer Antezedenz
    assert (frozenset(), ib(0, 1)) in rule_set
    assert (frozenset(), ib(2, 1)) in rule_set
    # Keine weiteren bedingten Regeln – alle bestimmten Bits sind Level-0
    assert len(rule_set) == 2


def test_find_association_rules_level0_does_not_produce_supersets():
    data = bitty([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    rules = find_association_rules(data)

    # Da c=1 bedingungslos (Level 0) bestimmt ist, darf keine Regel
    # wie {a=1} => c=1 hinzukommen (waere Redundant).
    for rule_list in rules.values():
        for r in rule_list:
            if r.consequent == ib(2, 1):
                assert r.antecedent == frozenset()


def test_find_association_rules_no_redundant_supersets():
    # c = a (b variiert und ist irrelevant)
    data = bitty([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
    ])
    rules = find_association_rules(data)

    # {a=1} => c=1 existiert bereits; {a=1, b=0} => c=1 darf nicht dazu kommen
    for rule_list in rules.values():
        for r in rule_list:
            if r.consequent == ib(2, 1):
                assert r.antecedent == frozenset({ib(0, 1)})
