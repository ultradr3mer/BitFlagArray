import numpy as np
import pytest

from clarautils import BitWriter, BitReader


def test_put_bool_msb_first():
    bw = BitWriter()
    bw.put(True)
    assert bw.get_bytes().tolist() == [0b1000_0000]


def test_put_int_with_length():
    bw = BitWriter()
    bw.put(5, length=3)
    bw.put(1, length=5)
    assert bw.get_bytes().tolist() == [0b1010_0001]


def test_put_bit_list():
    bw = BitWriter()
    bw.put(None, [1, 0, 1, 1])
    assert bw.get_bytes().tolist() == [0b1011_0000]


def test_zero_length_writes_nothing():
    bw = BitWriter()
    bw.put(7)
    assert bw.get_bytes().tolist() == []


def test_empty_writer_has_no_bytes():
    assert BitWriter().get_bytes().tolist() == []


def test_exact_byte_boundary():
    bw = BitWriter()
    bw.put(0xAB, length=8)
    assert bw.get_bytes().tolist() == [0xAB]
    bw.put(True)
    assert bw.get_bytes().tolist() == [0xAB, 0x80]


def test_multi_byte_value():
    bw = BitWriter()
    bw.put(0x1234, length=16)
    data = bw.get_bytes()
    assert data.dtype == np.uint8
    assert data.tolist() == [0x12, 0x34]


def test_demo_roundtrip():
    bw = BitWriter()
    bw.put(True)
    bw.put(2, length=5)
    bw.put(False)
    bw.put(12, length=5)
    data = bw.get_bytes()
    assert data.tolist() == [0b1000_1000, 0b1100_0000]

    br = BitReader(data)
    assert br.get_bool() is True
    assert br.get_int(5).value == 2
    assert br.get_bool() is False
    r = br.get_int(5)
    assert (r.value, r.bit_count) == (12, 5)


def test_random_roundtrip():
    rng = np.random.default_rng(0)
    fields = []
    bw = BitWriter()
    for _ in range(300):
        if rng.random() < 0.3:
            v = bool(rng.integers(2))
            bw.put(v)
            fields.append((v, None))
        else:
            n = int(rng.integers(1, 33))
            v = int(rng.integers(0, 1 << n))
            bw.put(v, length=n)
            fields.append((v, n))

    br = BitReader(bw.get_bytes())
    for v, n in fields:
        if n is None:
            assert br.get_bool() is v
        else:
            assert br.get_int(n).value == v


def test_reader_exhausted():
    br = BitReader(np.array([0xFF], np.uint8))
    br.get_int(8)
    with pytest.raises(StopIteration):
        br.get_bool()
