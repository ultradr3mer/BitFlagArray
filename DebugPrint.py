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


def _bit_letter(idx: int) -> str:
    return chr(ord('a') + idx) if idx < 26 else str(idx)


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
            return f"{_bit_letter(bit_idx)}{val}"
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

    body_lines = []
    for row_i, item_idx in enumerate(item_indices):
        if n_bits > 0:
            cells = " ".join(cell_str(row_i, c).ljust(max_cell) for c in range(n_bits))
        else:
            cells = ""
        body_lines.append(f"[{cells}]↤{item_idx}")

    inner_w = max_cell * n_bits + max(n_bits - 1, 0) if n_bits > 0 else 0
    content_w = inner_w + 2

    if n_bits == 0:
        header = ""
        sep = "⌟⌞"
    elif is_bit_slice:
        start_l = _bit_letter(bit_indices[0])
        end_l = _bit_letter(bit_indices[-1]) if n_bits > 1 else start_l
        header = f"[{start_l}⇿{end_l}]"
        sep = f"⌟⊩ ⭠{' ' * (content_w - 6)}⭢ ⫣⌞" if content_w > 6 else f"⌟⊩⭠⭢⫣⌞"
    else:
        parts = " ".join(_bit_letter(b).ljust(max_cell) for b in bit_indices)
        header = " " + parts + " "
        markers = []
        for i in range(n_bits):
            markers.append("↧" if i == 0 or i == n_bits - 1 else "⥝")
        sep = "⌟⌞" + "⌟⌞".join(markers) + "⌟⌞"

    bottom = "⌝" + " " * (content_w + 2) + "⌜"

    lines = [header, sep] + body_lines + [bottom]
    return "\n".join(lines)
