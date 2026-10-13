"""The run-length form a mask travels in between the editor and this server.

    "VMSK" · version u8 (=1) · width u32le · height u32le · runs...
    run = value u8 · length (unsigned LEB128)

It is the same bytes label-service reads, so the editor decodes a segmentation
result with the code it already has for its own masks.
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"VMSK"
VERSION = 1


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        low = value & 0x7F
        value >>= 7
        if value:
            out.append(low | 0x80)
        else:
            out.append(low)
            return bytes(out)


def encode_mask(mask: np.ndarray) -> bytes:
    """A 2-D array of small non-negative integers (0 = background) as run-length bytes."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    height, width = mask.shape
    flat = np.ascontiguousarray(mask, dtype=np.uint8).reshape(-1)
    out = bytearray(MAGIC + bytes([VERSION]) + struct.pack("<II", width, height))
    if flat.size == 0:
        return bytes(out)
    # Run boundaries: where the value changes from one pixel to the next.
    starts = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1))
    ends = np.concatenate((starts[1:], [flat.size]))
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        out.append(int(flat[start]))
        out += _varint(end - start)
    return bytes(out)


def decode_mask(data: bytes) -> np.ndarray:
    """The inverse; used by the tests, and handy for anyone scripting against the server."""
    if len(data) < 13 or data[:4] != MAGIC or data[4] != VERSION:
        raise ValueError("not a Visin mask")
    width, height = struct.unpack("<II", data[5:13])
    out = np.zeros(width * height, dtype=np.uint8)
    pos, filled = 13, 0
    while pos < len(data):
        value = data[pos]
        pos += 1
        length, shift = 0, 0
        while True:
            byte = data[pos]
            pos += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        out[filled : filled + length] = value
        filled += length
    if filled != out.size:
        raise ValueError("runs do not cover the image")
    return out.reshape(height, width)
