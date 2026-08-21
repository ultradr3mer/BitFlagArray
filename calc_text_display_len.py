from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap

from common import get_type_for_scalar

#
# def get_y_and_heights(text_wrapped, dimensions, margin, font):
#     """Get the first vertical coordinate at which to draw text and the height of each line of text"""
#     # https://stackoverflow.com/a/46220683/9263761
#     ascent, descent = font.getmetrics()
#
#     # Calculate the height needed to draw each line of text (including its bottom margin)
#     line_heights = [
#         font.getmask(text_line).getbbox()[3] + descent + margin
#         for text_line in text_wrapped
#     ]
#     # The last line doesn't have a bottom margin
#     line_heights[-1] -= margin
#
#     # Total height needed
#     height_text = sum(line_heights)
#
#     # Calculate the Y coordinate at which to draw the first line of text
#     y = (dimensions[1] - height_text) // 2
#
#     # Return the first Y coordinate and a list with the height of each line
#     return (y, line_heights)

# FONT_PATH = "C:\\Windows\\Fonts"
FONT_FAMILY = "consola.ttf"
WIDTH = 600
HEIGHT = 300
FONT_SIZE = 13
# V_MARGIN =  40
# CHAR_LIMIT = 12
BG_COLOR = "black"
TEXT_COLOR = "white"

text = "This is centered text"

# full = Path(FONT_PATH).joinpath(FONT_FAMILY)
# Create the font
font = ImageFont.truetype(FONT_FAMILY, FONT_SIZE)
# New image based on the settings defined above
img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
# Interface to draw on the image
draw_interface = ImageDraw.Draw(img)


# Wrap the `text` string into a list of `CHAR_LIMIT`-character strings
# text_lines = wrap(text, CHAR_LIMIT)
# Get the first vertical coordinate at which to draw text and the height of each line of text
# y, line_heights = get_y_and_heights(
#     text_lines,
#     (WIDTH, HEIGHT),
#     V_MARGIN,
#     font
# )


def get_display_width(line: str):
    return font.getmask(line).getbbox()[2]


def get_table_info(row: str):  # text including ]
    px_w = get_display_width(row)
    char_w = len(row)
    return char_w, px_w

SPACE_CHAR_WIDTH = 8.75

def gen_slice(table_char_w: int, table_px_w: int):  # fact kann angepasst werden
    # um schneller ergebnisse zu haben oder gar nicht erst rechnen zu müssen. Das ist aktuell aber unklar
    def build_slice(s_count: int):
        side_spaces = s_count // 4
        center_space = s_count // 2
        center_space += (s_count - side_spaces - center_space)
        return f"⌟⊩{' ' * side_spaces}⭠{' ' * center_space}⭢{' ' * side_spaces}⫣"

    # ---- base ---
    base_str = build_slice(0)
    char_count_base = len(base_str)
    base_width = get_display_width(base_str)
    # ---- first shot ---

    space_count = int(np.round((table_px_w - base_width) / SPACE_CHAR_WIDTH))
    # calc with char: table_char_w - char_count_base
    text = build_slice(space_count)
    width = get_display_width(text)
    delta = width - base_width
    px_off = abs(table_px_w - width)
    print("Tried (spaces):", space_count,
          "Length changed:", delta,
          "Off by:", px_off,
          "Space char width:", delta / space_count)


table_row = " [a1 b0 c1 d0 e1 f0 g1 h0 i1 j0 k1 l0 m1 n0 o1 p0]"
inf = get_table_info(table_row)
gen_slice(*inf)
