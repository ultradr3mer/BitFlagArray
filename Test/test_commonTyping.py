from typing import get_args, get_type_hints

import numpy as np
import pytest

from clarautils.commonTyping import (
    DTableFields,
    INTEGER_TYPES,
    TypeTable,
    get_as_signed,
    get_as_unsigned,
    get_as_fitting,
    get_type_for_array,
    get_type_for_bit_count,
    get_type_for_scalar,
)
from clarautils.QueryableTable import CCol, Undefined


# --------------------------------------------------------------------
# Fields are declared once — on DTableFields, which IS the table type
# --------------------------------------------------------------------

def test_fields_declared_once_on_dtablefields():
    item_type = INTEGER_TYPES.item_type
    assert DTableFields in TypeTable.__mro__                                     # table inherits the declaration
    assert INTEGER_TYPES.table_type is DTableFields                             # declaration is the table type
    assert not TypeTable.__annotations__                                        # table redeclares nothing
    assert list(get_type_hints(DTableFields)) == list(item_type._fields)        # item variant derives from it
    assert list(item_type._fields) == [f.name for f in INTEGER_TYPES.fields]   # columns derive from it


def test_dtablefields_are_range_item_unions():
    hints = get_type_hints(DTableFields)
    for hint in hints.values():
        args = get_args(hint)
        assert len(args) == 2
        rng, item = args if args[0] is CCol else args[::-1]
        assert rng is CCol
        assert issubclass(item, np.generic)


# --------------------------------------------------------------------
# Union hints resolve to the same scalar dtypes as the old plain hints
# --------------------------------------------------------------------

def test_union_hints_resolve_to_item_dtypes():
    expected = np.dtype([
        ("signed", np.bool_),
        ("abs_min", np.uint64),
        ("max", np.uint64),
        ("bits", np.uint8),
    ])
    assert INTEGER_TYPES.dtype == expected


# --------------------------------------------------------------------
# TypeTable — attrs hold the right columns, in field order
# --------------------------------------------------------------------

def test_typetable_attrs_hold_columns_in_field_order():
    for i, name in enumerate(INTEGER_TYPES.item_type._fields):
        assert getattr(INTEGER_TYPES, name) is INTEGER_TYPES.cols[i]


def test_typetable_query_smoke():
    signed_rows = [t for t in INTEGER_TYPES if t.signed == True]
    assert len(signed_rows) == 4
    assert int(INTEGER_TYPES[0].bits) == 8


# --------------------------------------------------------------------
# Item / Range variants on selection — both specific, generated inside
# --------------------------------------------------------------------

def test_item_variant_is_specific():
    hints = get_type_hints(INTEGER_TYPES.item_type)
    assert set(hints) == set(INTEGER_TYPES.item_type._fields)
    assert all(issubclass(h, np.generic) for h in hints.values())   # scalar types only


def test_range_variant_is_specific():
    hints = get_type_hints(INTEGER_TYPES.range_type)
    assert all(h is CCol for h in hints.values())                  # column types only


def test_table_index_returns_item_variant():
    item = INTEGER_TYPES[0]
    assert isinstance(item, INTEGER_TYPES.item_type)
    assert item.signed == np.False_
    item4 = INTEGER_TYPES[4]
    assert item4.signed == np.True_


def test_table_slice_returns_range_variant():
    rng = INTEGER_TYPES[0:4]
    assert rng.__class__.__name__ == "DTableRange"
    assert all(isinstance(rng.__getattribute__(n), CCol) for n in INTEGER_TYPES.item_type._fields)
    assert not any(bool(v) for v in rng.signed.column)

    rng2 = INTEGER_TYPES[4:8]
    assert all(bool(v) for v in rng2.signed.column)
    assert [int(v) for v in rng2.bits.column] == [8, 16, 32, 64]


# --------------------------------------------------------------------
# get_type_for_* — legacy unsigned contract
# --------------------------------------------------------------------

def test_get_type_for_scalar_legacy():
    assert get_type_for_scalar(0) == np.uint8
    assert get_type_for_scalar(255) == np.uint8
    assert get_type_for_scalar(256) == np.uint16
    assert get_type_for_scalar(2 ** 32 - 1) == np.uint32
    assert get_type_for_scalar(2 ** 32) == np.uint64


def test_get_type_for_bit_count_legacy():
    assert get_type_for_bit_count(8) == np.uint8
    assert get_type_for_bit_count(16) == np.uint16
    assert get_type_for_bit_count(32) == np.uint32
    assert get_type_for_bit_count(64) == np.uint64


def test_get_type_for_array_legacy():
    assert get_type_for_array([1, 2, 3]) == np.uint8
    assert get_type_for_array([300]) == np.uint16
    assert get_type_for_array(np.array([1, 2, 300])) == np.uint16


def test_get_type_errors_legacy_messages():
    with pytest.raises(Exception, match="to many bits requested"):
        get_type_for_bit_count(65)
    with pytest.raises(Exception, match="value to big"):
        get_type_for_scalar(2 ** 64)


# --------------------------------------------------------------------
# get_type_for_* — signed family, abs_min bounds the negative side
# --------------------------------------------------------------------

def test_get_type_for_scalar_signed():
    assert get_type_for_scalar(127, signed=True) == np.int8
    assert get_type_for_scalar(128, signed=True) == np.int16
    assert get_type_for_scalar(-128, signed=True) == np.int8    # abs_min lets -128 fit int8
    assert get_type_for_scalar(-129, signed=True) == np.int16


def test_get_type_for_bit_count_signed():
    assert get_type_for_bit_count(8, signed=True) == np.int8
    assert get_type_for_bit_count(9, signed=True) == np.int16


def test_get_type_for_array_signed():
    assert get_type_for_array([-128, 5], signed=True) == np.int8
    assert get_type_for_array([-129, 5], signed=True) == np.int16
    assert get_type_for_array(np.array([-1000, 1000]), signed=True) == np.int16


# --------------------------------------------------------------------
# get_type_for_* — Undefined = either family
# --------------------------------------------------------------------

def test_get_type_for_scalar_undefined():
    assert get_type_for_scalar(255, Undefined) == np.uint8
    assert get_type_for_scalar(128, Undefined) == np.uint8
    assert get_type_for_scalar(-128, Undefined) == np.int8
    assert get_type_for_scalar(256, Undefined) == np.uint16


def test_get_type_for_bit_count_undefined():
    assert get_type_for_bit_count(8, Undefined) == np.uint8
    assert get_type_for_bit_count(9, Undefined) == np.uint16


# --------------------------------------------------------------------
# get_as_signed/get_as_unsigned — same-size cast (fit=False)
# --------------------------------------------------------------------

def test_get_as_signed_unsigned_dtypes():
    assert get_as_signed(np.dtype("u1")) == np.dtype("i1")
    assert get_as_signed(np.dtype("u4")) == np.dtype("i4")
    assert get_as_signed(np.dtype("i2")) == np.dtype("i2")
    with pytest.raises(TypeError):
        get_as_signed(np.dtype("f4"))


def test_get_as_unsigned_dtypes():
    assert get_as_unsigned(np.dtype("i1")) == np.dtype("u1")
    assert get_as_unsigned(np.dtype("i8")) == np.dtype("u8")
    assert get_as_unsigned(np.dtype("u2")) == np.dtype("u2")
    with pytest.raises(TypeError):
        get_as_unsigned(np.dtype("f4"))


def test_get_as_signed_arrays():
    out = get_as_signed(np.array([5, 200], dtype=np.uint16))
    assert out.dtype == np.dtype("i2")
    np.testing.assert_array_equal(out, [5, 200])
    signed = np.array([-5, 5], dtype=np.int32)
    assert get_as_signed(signed) is signed
    with pytest.raises(TypeError):
        get_as_signed(np.array([1.0]))


def test_get_as_unsigned_arrays():
    out = get_as_unsigned(np.array([-5, 5], dtype=np.int16))
    assert out.dtype == np.dtype("u2")
    unsigned = np.array([5], dtype=np.uint8)
    assert get_as_unsigned(unsigned) is unsigned
    with pytest.raises(TypeError):
        get_as_unsigned(np.array([1.0]))


# --------------------------------------------------------------------
# get_as_signed/get_as_unsigned — fit=True casts via _get_type_for_bounds
# --------------------------------------------------------------------

def test_get_as_signed_fit():
    assert get_as_signed(np.array([0, 100], dtype=np.uint16), fit=True).dtype == np.dtype("i1")
    assert get_as_signed(np.array([0, 200], dtype=np.uint8), fit=True).dtype == np.dtype("i2")
    assert get_as_signed(np.array([-128, 100], dtype=np.int32), fit=True).dtype == np.dtype("i1")
    assert get_as_signed(np.array([-129, 5], dtype=np.int64), fit=True).dtype == np.dtype("i2")
    np.testing.assert_array_equal(get_as_signed(np.array([0, 200], dtype=np.uint8), fit=True), [0, 200])


def test_get_as_signed_fit_scalar():
    out = get_as_signed(np.uint16(5), fit=True)
    assert out.dtype == np.dtype("i1")
    assert out.shape == ()
    assert get_as_signed(5, fit=True).dtype == np.dtype("i1")


def test_get_as_signed_fit_no_copy_when_fitting():
    a = np.array([1, 2], dtype=np.int8)
    assert get_as_signed(a, fit=True) is a


def test_get_as_unsigned_fit():
    assert get_as_unsigned(np.array([0, 300], dtype=np.int32), fit=True).dtype == np.dtype("u2")
    assert get_as_unsigned(np.array([0, 5], dtype=np.int8), fit=True).dtype == np.dtype("u1")
    assert get_as_unsigned(np.array([0, 5], dtype=np.uint16), fit=True).dtype == np.dtype("u1")
    np.testing.assert_array_equal(get_as_unsigned(np.array([0, 300], dtype=np.int32), fit=True), [0, 300])


def test_get_as_unsigned_fit_scalar():
    out = get_as_unsigned(np.int32(300), fit=True)
    assert out.dtype == np.dtype("u2")
    assert out.shape == ()


def test_get_as_unsigned_fit_negative_raises():
    with pytest.raises(Exception, match="value to big"):
        get_as_unsigned(np.array([-1, 5], dtype=np.int8), fit=True)
    with pytest.raises(Exception, match="value to big"):
        get_as_unsigned(-1, fit=True)


def test_get_as_unsigned_allow_integer_floats():
    out = get_as_unsigned(np.array([2.0, 10.0]), fit=True, acc_floats=True)
    assert out.dtype == np.dtype("u1")
    np.testing.assert_array_equal(out, [2, 10])
    assert get_as_unsigned(2.0, acc_floats=True) == 2
    with pytest.raises(TypeError):
        get_as_unsigned(np.array([2.0]))
    with pytest.raises(TypeError):
        get_as_unsigned(np.array([2.5]), acc_floats=True)
    with pytest.raises(TypeError):
        get_as_unsigned(2.5, acc_floats=True)


# --------------------------------------------------------------------
# scalars in, scalars out (int | np.integer)
# --------------------------------------------------------------------

def test_get_as_signed_scalar():
    out = get_as_signed(np.uint16(5))
    assert isinstance(out, np.integer)
    assert out.dtype == np.dtype("i2")
    assert out == 5
    out = get_as_signed(5)
    assert isinstance(out, np.integer)
    assert out.dtype == np.dtype("i8")
    assert get_as_signed(np.int32(-5), fit=True) == np.int8(-5)
    with pytest.raises(TypeError):
        get_as_signed(True)


def test_get_as_unsigned_scalar():
    out = get_as_unsigned(np.int16(-1))
    assert isinstance(out, np.integer)
    assert out.dtype == np.dtype("u2")
    out = get_as_unsigned(5)
    assert isinstance(out, np.integer)
    assert out.dtype == np.dtype("u8")
    with pytest.raises(TypeError):
        get_as_unsigned(True)


# --------------------------------------------------------------------
# get_as_fitting — smallest fitting dtype of the signed family
# --------------------------------------------------------------------

def test_get_as_fitting():
    assert get_as_fitting(np.array([0, 300], dtype=np.int32)).dtype == np.dtype("u2")
    assert get_as_fitting(np.array([-128, 100], dtype=np.int32), signed=True).dtype == np.dtype("i1")
    assert get_as_fitting(np.array([-129, 5], dtype=np.int64), signed=True).dtype == np.dtype("i2")
    a = np.array([1, 2], dtype=np.uint8)
    assert get_as_fitting(a) is a
    np.testing.assert_array_equal(get_as_fitting(np.array([0, 200], dtype=np.uint8), signed=True), [0, 200])


def test_get_as_fitting_undefined():
    assert get_as_fitting(np.array([-128, 100], dtype=np.int32), Undefined).dtype == np.dtype("i1")
    assert get_as_fitting(np.array([0, 255], dtype=np.int32), Undefined).dtype == np.dtype("u1")


def test_get_as_fitting_scalar():
    out = get_as_fitting(300, signed=True)
    assert isinstance(out, np.integer)
    assert out.dtype == np.dtype("i2")
    out = get_as_fitting(np.uint16(5))
    assert isinstance(out, np.integer)
    assert out.dtype == np.dtype("u1")


def test_get_as_fitting_negative_unsigned_raises():
    with pytest.raises(Exception, match="value to big"):
        get_as_fitting(np.array([-1, 5], dtype=np.int8))
    with pytest.raises(TypeError):
        get_as_fitting(np.array([1.0]))