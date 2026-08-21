from typing import List

from wcwidth import wcwidth as _wcwidth, wcswidth as _wcswidth

from BitFlagArray import SliceView, sliceTypes

_SUPERSCRIPT = {
    'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ',
    'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ',
    'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ',
    'p': 'ᵖ', 'q': 'q', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ',
    'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
}

_SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"

_FILLED_CIRCLED = "❶❷❸❹❺❻❼❽❾❿⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴"
_EMPTY_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


_WIDE_CHARS = set()
def set_wide_chars(chars: str):
    global _WIDE_CHARS
    _WIDE_CHARS = set(chars)

def _char_w(c: str) -> int:
    if c in _WIDE_CHARS:
        return 2
    w = _wcwidth(c)
    return w if w >= 1 else 1

def _str_w(s: str) -> int:
    return sum(_char_w(c) for c in s)

def _pad_w(s: str, width: int, fill: str = ' ') -> str:
    pad = width - _str_w(s)
    if pad <= 0:
        return s
    return s + fill * pad

def _center_w(s: str, width: int, fill: str = ' ') -> str:
    total = width - _str_w(s)
    if total <= 0:
        return s
    left = total // 2
    right = total - left
    return fill * left + s + fill * right

def _pos_w(s: str, target_w: int) -> int:
    pos = 0
    w = 0
    for i, c in enumerate(s):
        if w >= target_w:
            return pos
        w += _char_w(c)
        pos = i
    return pos

def _find_w(s: str, char: str, n=1) -> int:
    count = 0
    w = 0
    for i, c in enumerate(s):
        if c == char:
            count += 1
            if count == n:
                return w
        w += _char_w(c)
    return -1

def _char_at_w(s: str, target_w: int) -> str:
    w = 0
    for i, c in enumerate(s):
        if w == target_w:
            return c
        w += _char_w(c)
    return ''

def _max_w_key(positions: dict[int, str]) -> int:
    return max(positions.keys()) if positions else 0


def _build_sep_at_w(width: int, positions: dict[int, str]) -> str:
    pos = 0
    w = 0
    result = []
    for target_w, char in sorted(positions.items()):
        while w < target_w:
            result.append(' ')
            w += 1
            pos += 1
        result.append(char)
        w += _char_w(char)
        pos += 1
    while w < width:
        result.append(' ')
        w += 1
        pos += 1
    return ''.join(result)


def _bit_letter(idx: int) -> str:
    if idx < 26:
        return chr(ord('a') + idx)
    first = (idx - 26) // 26
    second = (idx - 26) % 26
    return chr(ord('a') + first) + chr(ord('a') + second)


def _superscript_letter(idx: int) -> str:
    if idx < 26:
        letter = chr(ord('a') + idx)
        return _SUPERSCRIPT.get(letter, letter)
    first = (idx - 26) // 26
    second = (idx - 26) % 26
    l1 = chr(ord('a') + first)
    l2 = chr(ord('a') + second)
    return _SUPERSCRIPT.get(l1, l1) + _SUPERSCRIPT.get(l2, l2)


def _subscript_number(n: int) -> str:
    return ''.join(_SUBSCRIPT_DIGITS[int(d)] for d in str(n))


def _circled_number(bit_idx: int, set: bool) -> str:
    if 0 <= bit_idx < 20:
        return _FILLED_CIRCLED[bit_idx] if set else _EMPTY_CIRCLED[bit_idx]
    return '●' if set else '○'


def print_debug(view: SliceView, mode: str = "1/0") -> str:
    def get_indices(key, length: int) -> List[int]:
        if isinstance(key, slice):
            start, stop, step = key.indices(length)
            return list(range(start, stop, step))
        return list(key)

    item_indices = get_indices(view.item_slice, len(view.data))
    bit_indices = get_indices(view.bit_slice, view.data.get_bit_count())
    bits = view.get_bitwise()
    is_bit_slice = isinstance(view.bit_slice, slice)

    def cell_str(row: int, col: int) -> str:
        val = int(bits[row, col])
        bit_idx = bit_indices[col]
        item_idx = item_indices[row]
        if mode == "1/0":
            return f"{_bit_letter(bit_idx)}{val}"
        elif mode == "sub":
            return _superscript_letter(bit_idx) + _subscript_number(item_idx)
        elif mode == "circ":
            return _superscript_letter(bit_idx) + _circled_number(bit_idx, val == 1)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use '1/0', 'sub', or 'circ'.")

    n_bits = len(bit_indices)
    n_items = len(item_indices)

    max_cell_w = 1
    for r in range(n_items):
        for c in range(n_bits):
            max_cell_w = max(max_cell_w, _str_w(cell_str(r, c)))

    body_lines = []
    for row_i, item_idx in enumerate(item_indices):
        if n_bits > 0:
            cells = " ".join(_pad_w(cell_str(row_i, c), max_cell_w) for c in range(n_bits))
        else:
            cells = ""
        body_lines.append(f" [{cells}]↤{item_idx}")

    if n_bits == 0:
        header = " "
        sep = "⌟⌞"
        bottom = _pad_w("⌝", _str_w("⌝")) + "⌜"
    else:
        if is_bit_slice:
            start_l = _bit_letter(bit_indices[0])
            end_l = _bit_letter(bit_indices[-1]) if n_bits > 1 else start_l
            header = f" [{start_l}⇿{end_l}]"
        else:
            parts = " ".join(_pad_w(_bit_letter(b), max_cell_w) for b in bit_indices)
            header = "  " + parts + " "

        body = body_lines[0] if body_lines else f" [{' ' * (max_cell_w * n_bits + max(n_bits - 1, 0))}]↤"
        bracket_close_w = _find_w(body, ']')
        arrow_w = _find_w(body, '↤')

        if is_bit_slice:
            positions = {0: '⌟', 1: '⊩'}
            positions[bracket_close_w] = '⫣'
            positions[arrow_w] = '⌞'
            if bracket_close_w > 4:
                positions[3] = '⭠'
                positions[bracket_close_w - _char_w('⭢')] = '⭢'
            sep = _build_sep_at_w(arrow_w + 1, positions)
        else:
            positions = {0: '⌟', 1: '⌞'}
            for i in range(n_bits):
                cell_start = _str_w(' [') + i * (max_cell_w + 1)
                marker = '↧' if i == 0 or i == n_bits - 1 else '⥝'
                positions[cell_start] = marker
                positions[cell_start + _char_w(marker)] = '⌟'
                positions[cell_start + _char_w(marker) + 1] = '⌞'
            sep = _build_sep_at_w(_max_w_key(positions) + 1, positions)

        bottom = _pad_w("⌝", arrow_w) + "⌜"

    lines = [header, sep] + body_lines + [bottom]
    return '\n'.join(lines)
