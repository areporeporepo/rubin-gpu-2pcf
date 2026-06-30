"""Tile compression — the mechanism FITS uses and cuPhoton accelerates.

FITS tile-compresses images: the array is cut into tiles, each tile is
compressed independently, so decompression is embarrassingly parallel. That
parallelism is exactly what a GPU exploits.

Real LSST uses RICE_1. nvCOMP has NO Rice codec (its codecs are
LZ4/Snappy/GDeflate/Deflate/Zstd/...), so a true GPU Rice decode needs a custom
CUDA kernel (what cuPhoton presumably wrote). Here we use **raw DEFLATE** tiles:
- CPU side: stdlib `zlib` (raw deflate, no header) -- runs anywhere.
- GPU side: nvCOMP "Deflate" -- consumes the same raw-deflate bytes.

Same wire format on both sides => an honest apples-to-apples decode comparison
of the *same parallel-tile mechanism*, just not the Rice codec specifically.
"""

from __future__ import annotations

import zlib

import numpy as np


def tile_boxes(shape, tile):
    """List of (y0, y1, x0, x1) tile bounds covering `shape`."""
    h, w = shape
    th, tw = tile
    return [
        (y, min(y + th, h), x, min(x + tw, w))
        for y in range(0, h, th)
        for x in range(0, w, tw)
    ]


def _deflate(data: bytes, level: int) -> bytes:
    """Raw DEFLATE (no zlib header/trailer) so nvCOMP Deflate can consume it."""
    co = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    return co.compress(data) + co.flush()


def _inflate(data: bytes) -> bytes:
    do = zlib.decompressobj(-zlib.MAX_WBITS)
    return do.decompress(data) + do.flush()


def compress_tiles(img, tile=(256, 256), level=6):
    """Cut `img` into tiles and raw-DEFLATE each. Returns (blobs, meta)."""
    boxes = tile_boxes(img.shape, tile)
    blobs, raw_sizes = [], []
    for (y0, y1, x0, x1) in boxes:
        raw = np.ascontiguousarray(img[y0:y1, x0:x1]).tobytes()
        raw_sizes.append(len(raw))
        blobs.append(_deflate(raw, level))
    meta = {
        "shape": tuple(img.shape),
        "dtype": img.dtype.str,
        "tile": tuple(tile),
        "boxes": boxes,
        "raw_sizes": raw_sizes,
    }
    return blobs, meta


def decompress_tiles_cpu(blobs, meta):
    """Reassemble the image from raw-DEFLATE tiles on the CPU (reference path)."""
    dtype = np.dtype(meta["dtype"])
    img = np.empty(meta["shape"], dtype=dtype)
    for blob, (y0, y1, x0, x1) in zip(blobs, meta["boxes"]):
        raw = _inflate(blob)
        tile = np.frombuffer(raw, dtype=dtype).reshape(y1 - y0, x1 - x0)
        img[y0:y1, x0:x1] = tile
    return img


def compression_ratio(blobs, meta) -> float:
    raw = sum(meta["raw_sizes"])
    comp = sum(len(b) for b in blobs)
    return raw / comp
