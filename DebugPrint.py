from typing import List

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


def _superscript_letter(idx: int) -> str:
    letter = chr(ord('a') + idx) if idx < 26 else str(idx)
    return _SUPERSCRIPT.get(letter, letter)


def _subscript_number(n: int) -> str:
    return ''.join(_SUBSCRIPT_DIGITS[int(d)] for d in str(n))


def _circled_number(bit_idx: int, set: bool) -> str:
    if 0 <= bit_idx < 20:
        return _FILLED_CIRCLED[bit_idx] if set else _EMPTY_CIRCLED[bit_idx]
    return '●' if set else '○'


def slice_debug(view: SliceView, mode: str = "1/0") -> str:
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
            return str(val)
        elif mode == "sub":
            return _superscript_letter(bit_idx) + _subscript_number(item_idx)
        elif mode == "circ":
            return _superscript_letter(bit_idx) + _circled_number(bit_idx, val == 1)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use '1/0', 'sub', or 'circ'.")

    n_bits = len(bit_indices)
    n_items = len(item_indices)

    max_cell = 1
    for r in range(n_items):
        for c in range(n_bits):
            max_cell = max(max_cell, len(cell_str(r, c)))
    max_bit_label = max((len(str(b)) for b in bit_indices), default=1)
    cell_w = max(max_cell, max_bit_label) + 2

    max_item_label = max((len(str(i)) for i in item_indices), default=0)
    label_w = max(max_item_label, 0) + 1

    def pad(s: str) -> str:
        return s.center(cell_w)

    total_w = label_w + 1 + cell_w * n_bits
    lines = []

    if n_bits == 0:
        lines.append(' ' * label_w + '├' + '─' * (cell_w * n_bits) + '┤')
    elif is_bit_slice:
        start_b = bit_indices[0]
        end_b = bit_indices[-1]
        inner_w = cell_w * n_bits
        if start_b == end_b:
            core = str(start_b)
        else:
            core = f"{start_b}─ slice ─{end_b}"
        if len(core) > inner_w:
            core = f"{start_b}─{end_b}"
        top_label = core.center(inner_w, '─') if len(core) < inner_w else core[:inner_w]
        lines.append(' ' * label_w + '├' + top_label + '┤')
    else:
        header = ' ' * label_w + '│'
        for b in bit_indices:
            header += pad(str(b))
        header += '│'
        lines.append(header)

        sep = ' ' * label_w + '├'
        for _ in range(n_bits):
            half = cell_w // 2
            sep += '─' * half + '┴' + '─' * (cell_w - half - 1)
        sep += '┤'
        lines.append(sep)

    for row_i, item_idx in enumerate(item_indices):
        row = str(item_idx).rjust(label_w) + '│'
        for col_j in range(n_bits):
            row += pad(cell_str(row_i, col_j))
        row += '│'
        lines.append(row)

    return '\n'.join(lines)