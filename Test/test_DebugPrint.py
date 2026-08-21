import numpy as np
import pytest

from BitFlagArray import Bitty
from DebugPrint import print_debug, _bit_letter, _superscript_letter, _subscript_number, _FILLED_CIRCLED, _EMPTY_CIRCLED
from TextDisplayMeasure import TextDisplayMeasure

from showcase import test, long_data

@pytest.fixture
def test_bitty():
  return Bitty.stack_bit(test)

@pytest.fixture
def long_bitty():
  return Bitty.stack_bit(long_data)

@pytest.fixture
def bitty(bit_data):
    return Bitty.stack_bit(bit_data)


def lines(view, mode="1/0"):
    return print_debug(view, mode).splitlines()


def _first_col(line):
    return line[0] if line else ''


def _find_char(line, char, n=1):
    count = 0
    for i, c in enumerate(line):
        if c == char:
            count += 1
            if count == n:
                return i
    return -1


def _body_line(view, mode="1/0", row=0):
    return lines(view, mode)[2 + row]


def test1_slice_fence_at_bracket(bitty):
    for view, mode in [(bitty.b[:], "1/0"), (bitty.b[1:4], "sub"), (bitty.i[1:4], "circ")]:
        out = lines(view, mode)
        sep = out[1]
        body = out[2]
        fence_col = _find_char(sep, '⫣')
        bracket_col = _find_char(body, ']')
        assert fence_col == bracket_col, f"⫣ at {fence_col}, ] at {bracket_col}: {sep!r} vs {body!r}"


# def test_text_display_measurment():
#     display = TextDisplayMeasure()
#
#     table_row = " [a1 b0 c1 d0 e1 f0 g1 h0 i1 j0 k1 l0 m1 n0 o1 p0]"
#     row_px_with = display.get_table_info(table_row)
#     slice_str, text_px_width = display.gen_slice(row_px_with)
#     np.testing.assert_allclose(actual=text_px_width,
#                                desired=row_px_with,
#                                rtol=display.SPACE_CHAR_WIDTH,
#                                atol=display.SPACE_CHAR_WIDTH,
#                                err_msg=f"got {slice_str} for {table_row}: {row_px_with!r} vs {text_px_width!r} -> {abs(row_px_with - text_px_width)}px appart")
#
#
# def test1_slice_fence_at_bracket_visualy(test_bitty):
#     bitty = test_bitty
#     for view, mode in [(bitty.b[:], "1/0"), (bitty.b[1:4], "sub"), (bitty.i[1:4], "circ")]:
#         out = lines(view, mode)
#         display = TextDisplayMeasure()
#         display.draw_debug_image(out)
#
#         sep = out[1]
#         body = out[2]
#
#         fence_col = _find_char(sep, '⫣')
#         bracket_col = _find_char(body, ']')
#
#         fence_str = sep[:fence_col+1]
#         bracket_str = body[:bracket_col+1]
#
#         bracket_px_width = display.get_display_size(bracket_str)
#         fence_px_width = display.get_display_size(fence_str)
#         np.testing.assert_allclose(actual=fence_px_width,
#                                    desired=bracket_px_width,
#                                    rtol=display.SPACE_CHAR_WIDTH,
#                                    atol=display.SPACE_CHAR_WIDTH,
#                                    err_msg=f"⫣ at {fence_col}, ] at {bracket_col}: {fence_px_width!r} vs {bracket_px_width!r} -> {abs(fence_px_width - bracket_px_width)} appart")


def test2_closing_corner_at_arrow(bitty):
    for view, mode in [(bitty.b[:], "1/0"), (bitty.b[1:4], "sub"), (bitty.i[2], "circ")]:
        out = lines(view, mode)
        sep = out[1]
        bottom = out[-1]
        body = out[2]
        last_corner = _find_char(sep, '⌞', n=sep.count('⌞'))
        bottom_corner = _find_char(bottom, '⌜')
        arrow_col = _find_char(body, '↤')
        assert last_corner == arrow_col, f"⌞ at {last_corner}, ↤ at {arrow_col}: {sep!r} vs {body!r}"
        assert bottom_corner == arrow_col, f"⌜ at {bottom_corner}, ↤ at {arrow_col}: {bottom!r} vs {body!r}"


def test2_index_sep_ends_with_clover(bitty):
    from DebugPrint import _str_w
    for view, mode in [(bitty.b[[0, 2, 4]], "1/0"), (bitty.b[2], "sub")]:
        out = lines(view, mode)
        sep = out[1]
        assert sep.endswith('⌟⌞')
        body = out[2]
        arrow_col = _find_char(body, '↤')
        bottom = out[-1]
        bottom_corner = _find_char(bottom, '⌜')
        assert bottom_corner == arrow_col, f"⌜ at {bottom_corner}, ↤ at {arrow_col}"


def test3_fence_before_bracket(bitty):
    for view, mode in [(bitty.b[:], "1/0"), (bitty.b[1:4], "sub")]:
        out = lines(view, mode)
        sep = out[1]
        body = out[2]
        corner_col = _find_char(sep, '⌟')
        bracket_open = _find_char(body, '[')
        assert corner_col < bracket_open, f"⌟ at {corner_col} should be before [ at {bracket_open}"
        wall_col = _find_char(sep, '⊩')
        assert wall_col == bracket_open, f"⊩ at {wall_col}, [ at {bracket_open}: {sep!r} vs {body!r}"


def test4_first_col_only_corners(bitty):
    for view, mode in [(bitty.b[:], "1/0"), (bitty.b[1:4], "sub"), (bitty.b[[0, 2, 4]], "1/0"),
                       (bitty.i[1:4], "circ"), (bitty.b[2], "1/0"), (bitty.b[0:0], "1/0")]:
        out = lines(view, mode)
        for ln in out:
            c = _first_col(ln)
            assert c in ('⌟', '⌝', ' '), f"First col {c!r} not allowed: {ln!r}"


def test5_slice_header_aligned(bitty):
    for view, mode in [(bitty.b[:], "1/0"), (bitty.b[1:4], "sub")]:
        out = lines(view, mode)
        header = out[0]
        sep = out[1]
        body = out[2]
        header_bracket = _find_char(header, '[')
        body_bracket = _find_char(body, '[')
        assert header_bracket == body_bracket, f"header [ at {header_bracket}, body [ at {body_bracket}"
        wall_col = _find_char(sep, '⊩')
        assert wall_col == body_bracket, f"⊩ at {wall_col}, [ at {body_bracket}"


def test6_indices_first_clover_at_bracket(bitty):
    for view, mode in [(bitty.b[[0, 2, 4]], "1/0"), (bitty.b[2], "sub"), (bitty.b[[0, 2, 4]], "circ")]:
        out = lines(view, mode)
        sep = out[1]
        body = out[2]
        clover_col = _find_char(sep, '⌞')
        bracket_col = _find_char(body, '[')
        assert clover_col == bracket_col, f"⌞ at {clover_col}, [ at {bracket_col}: {sep!r} vs {body!r}"


def test7_letters_above_arrows(bitty):
    for view, mode in [(bitty.b[[0, 2, 4]], "1/0"), (bitty.b[2], "1/0")]:
        out = lines(view, mode)
        header = out[0]
        sep = out[1]
        markers_in_sep = [(i, c) for i, c in enumerate(sep) if c in ('↧', '⥝')]
        letters_in_header = [(i, c) for i, c in enumerate(header) if c.isalpha() and c != ' ']
        assert len(markers_in_sep) == len(letters_in_header)
        for (m_pos, m_char), (l_pos, l_char) in zip(markers_in_sep, letters_in_header):
            assert m_pos == l_pos, f"letter {l_char} at {l_pos}, marker {m_char} at {m_pos}"


def test8_multi_letter_after_z(long_data):
    lb = Bitty.stack_bit(long_data)
    out = lines(lb.b[:], "1/0")
    header = out[0]
    assert 'p' in header
    body = lines(lb.b[:], "sub")
    assert _superscript_letter(15) in body[2]


def test_no_horizontal_lines(bitty):
    out = print_debug(bitty.b[:])
    assert '─' not in out


def test_body_row_count(bitty):
    for expr in ["b[:]", "b[1:4]", "i[1:4]", "b[[0,2,4]]", "i[[1,3,5]]", "b[2]", "i[2]", "b[0:0]"]:
        view = eval(f"bitty.{expr}")
        body = lines(view)[2:-1]
        assert len(body) == view.get_item_count(), f"{expr}: {len(body)} != {view.get_item_count()}"


def test_mode_1_0_correct_values(bitty):
    body = print_debug(bitty.b[:], "1/0").splitlines()[2:-1]
    bits = bitty.get_bitwise()
    for row_i, ln in enumerate(body):
        for col_j, val in enumerate(bits[row_i]):
            letter = chr(ord('a') + col_j)
            assert f"{letter}{int(val)}" in ln


def test_mode_sub_shows_superscript_subscript(bitty):
    body = print_debug(bitty.b[:], "sub").splitlines()[2:-1]
    assert _superscript_letter(0) + _subscript_number(0) in body[0]
    assert _superscript_letter(5) + _subscript_number(5) in body[5]


def test_mode_circ_shows_circled(bitty):
    body = print_debug(bitty.b[:], "circ").splitlines()[2:-1]
    bits = bitty.get_bitwise()
    for row_i, ln in enumerate(body):
        for col_j, val in enumerate(bits[row_i]):
            circ = _FILLED_CIRCLED[col_j] if int(val) == 1 else _EMPTY_CIRCLED[col_j]
            assert _superscript_letter(col_j) + circ in ln


def test_empty_bit_slice(bitty):
    body = print_debug(bitty.b[0:0]).splitlines()[2:-1]
    assert len(body) == 6
    for ln in body:
        assert "[]" in ln
        assert "↤" in ln


def test_invalid_mode_raises(bitty):
    with pytest.raises(ValueError):
        print_debug(bitty.b[:], "invalid")


def test_print_showcase(bitty, long_data):
    print()
    for expr in ["b[:]", "b[1:4]", "b[1:4][1:4]", "i[1:4]", "b[[0,2,4]]", "i[[1,3,5]]", "b[2]", "i[2]", "b[0:0]"]:
        view = eval(f"bitty.{expr}")
        print(f"═══ {expr} ═══")
        for mode in ("1/0", "sub", "circ"):
            print(f"─── mode={mode} ───")
            print(print_debug(view, mode))
        print()

    lb = Bitty.stack_bit(long_data)
    for expr in ["b[:]", "b[8:]", "b[[0,5,10,15]]"]:
        view = eval(f"lb.{expr}")
        print(f"═══ long {expr} ═══")
        for mode in ("1/0", "sub", "circ"):
            print(f"─── mode={mode} ───")
            print(print_debug(view, mode))
        print()
