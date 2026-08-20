from typing import Tuple, List

from BitFlagArray import SliceView, sliceTypes


def slice_debug(view: SliceView):
    def indices(key: sliceTypes, len: int) -> Tuple[List[int], bool]:
        if isinstance(key, slice):
            tpl = key.indices(len)
            return list(tpl), True
        else:
            return list(key), False
        
    item_indices, is_item_slice = indices(view.item_slice, view.get_item_count())    
    bit_indices, is_bit_slice = indices(view.bit_slice, view.get_bit_count())    

    # 2. Header bauen
    item_header_parts = []
    if is_item_slice:
        start = item_indices[0] if item_indices else 0
        end = item_indices[-1] if item_indices else 0
        item_header_parts.append(f"| {start}  -     -      {end}   |")
    else:
        item_header_parts.append(" ".join(f"_{i}" for i in item_indices))

    bit_header_parts = []
    if is_bit_slice:
        start = bit_indices[0] if bit_indices else 0
        end = bit_indices[-1] if bit_indices else 0
        bit_header_parts.append(f"| {start} - - {end} |")
    else:
        bit_header_parts.append("_")  # Nur ein Unterstrich für Arrays

    header = f"{''.join(item_header_parts)} {''.join(bit_header_parts)}"

    # 3. Body (Zeilen) bauen
    lines = [header]
    for b in bit_indices:
        # Item-Teil
        if is_item_slice:
            item_str = " ".join(f"{i}/{b}" for i in item_indices)
        else:
            # Bei Array-Auswahl auch das Format i/b nutzen, aber ohne Rahmen
            item_str = " ".join(f"{i}/{b}" for i in item_indices)

        # Bit-Teil (hier nur der aktuelle Bit-Index mit Unterstrich für Array)
        if is_bit_slice:
            bit_str = f"{b}"
        else:
            bit_str = f"_ {b}"

        # Kombiniere (mit etwas Padding, das du anpassen kannst)
        lines.append(f"{item_str}    {bit_str}")

    return "\n".join(lines)