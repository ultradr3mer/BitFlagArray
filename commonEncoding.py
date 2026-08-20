from typing import Iterable, Any

import numpy as np

from modelCompression.common import get_bit_count


def get_bitmask(length, start=0):
    return (np.power(2, length)-1)<<start
    # result = 0
    # for i in range(length):
    #     result |= 1 << i
    # return result

def trim_leading_zero(bits: Iterable):
    had_zero = 0
    for bit in bits:
        if bit == 0:
            had_zero += 1
        else:
            break
    had_zero -= 1
    if had_zero > 0:
        return bits[had_zero:]
    return bits

def get_bits(value: np.ndarray | int | str, count = None):
    if isinstance(value, str):
        return [1 if c == '1' else 0 for c in value]

    bit_count = int(count if count is not None and count > 0
                    else get_bit_count(np.max(value)))
    bit_range = np.arange(bit_count-1, -1, -1)

    if np.isscalar(value):
        return (value >> bit_range) & 1
    else:
        return (value.reshape((-1,1)) >> bit_range) & 1


def get_number(bits):
    return sum([2**i if b == 1 or b == '1' else 0 for i, b in enumerate(reversed(bits))])

def flip_enc(value):
    def gen(ary):
        last = '0' if isinstance(ary[0], str) else 0
        for v in ary:
            yield (v if v != 0 and v != 1
                        and v != '1' and v != '0'
                    else 1 if v != last else 0)
            last = v
    return list(gen(value))

def trim_trailing_zero(p):
    if sum(p) == 0:
        return [0, 0]
    zeros = 0
    while sum(p[-zeros-1:]) == 0:
        zeros += 1
    return p if zeros == 0 else p[:-zeros]

def symbol_to_str(sym: Iterable[Any] | np.ndarray[np.generic]) -> str:
    result = ''
    for s in sym:
        result += str(s)
    return result

def get_bit_flag_idx(value):
    return np.log2(value).astype(int) if value > 0 else 0

def bits_to_hex(bits: str) -> str:
    """Konvertiert einen Binärstring in einen Hexadezimalstring."""
    if not bits:
        return "0"

    # Auf ein Vielfaches von 4 auffüllen
    padding = (4 - len(bits) % 4) % 4
    padded_bits = bits + '0' * padding

    # Jeweils 4 Bits in ein Hex-Zeichen umwandeln
    return "".join(
        f"{int(padded_bits[i:i + 4], 2):X}"
        for i in range(0, len(padded_bits), 4)
    )


def hex_to_bits(hex_str: str, original_length: int = None) -> str:
    """Konvertiert einen Hexadezimalstring in einen Binärstring."""
    if not hex_str:
        return ""

    # Jedes Hex-Zeichen in genau 4 Bits umwandeln (mit führenden Nullen)
    bits = "".join(f"{int(c, 16):04b}" for c in hex_str)

    # Falls die Originallänge übergeben wurde, überschüssige Padding-Nullen abschneiden
    if original_length is not None and original_length < len(bits):
        return bits[:original_length]

    return bits

def build_bins_n_print(v, q):
    print(f"Max: {np.max(v)}, Avg: {np.average(v)}")
    bins = np.bincount(v)
    print(f"Distribution: {bins}")
    value_perc = np.array(np.percentile(v, q=q), dtype=np.uint32)
    value_perc_delta = value_perc - np.concat((np.zeros(1, dtype=np.uint32), value_perc[:-1]))
    print(f"Q{q}: {value_perc}->{value_perc_delta}")
    return value_perc_delta

def fmt_k_bits(b: int) -> str:
    size, u = b / 8, 0
    while size >= 1024 and u < 4:
        size /= 1024
        u += 1
    units = ["B", "KB", "MB", "GB", "TB"]
    return f"{b:,} bits ({size:.1f} {units[u]})"

# def arrange_bits(values, target_indices):
#     def calc_value_for_bit_idx(bit_idx):
#         return np.pow(2, bit_idx)
#     vl = calc_value_for_bit_idx
#
#     result = np.zeros_like(values)
#
#     for idx_to, idx_from in enumerate(target_indices):
#         result = result + vl(idx_to) * (vl(idx_from) & values > 0)
#
#     return result

def arrange_bits(values, take_indices):
    result = np.zeros_like(values)

    for idx_to, idx_from in enumerate(take_indices):
        result = result + (1 << idx_to) * ((values >> idx_from) & 1)

    return result
