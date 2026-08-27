from pathlib import Path

import numpy as np

from commonEncoding import get_bits


def build(columns):
    pass


if __name__ == "__main__":
    bits_to_take = 32
    num_possible = np.pow(2, bits_to_take)
    base = Path("bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint32)

        values, counts = np.unique(x, return_counts=True)

        bits = get_bits(values)

        build(bits)